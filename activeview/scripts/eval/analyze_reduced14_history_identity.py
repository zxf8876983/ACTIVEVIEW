#!/usr/bin/env python3
"""Val-only diagnostics for recovering action identity from s0+s1 history.

Two small classifiers are trained on the reduced14 Train contexts: one sees
only the two posterior vectors and the other also sees the frozen 256-D
ST-GCN features.  Their Val beliefs are used only in an offline privileged
candidate selector whose candidate recognition is the archived true_logp.
Formal WM-E, Joint Revision and ST-GCN artifacts are never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.analyze_reduced14_history_belief_fusion import _belief_selector
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _orders,
    _terminal_predictions,
    _validate_rows_cache,
)


SEED = 42
NUM_CLASSES = 14
STGCN_FEATURE_DIM = 256
LOGP_DIM = NUM_CLASSES
HIDDEN_DIM = 256
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_identity"
PREVIOUS_SELECTOR_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_belief_fusion/result.json"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HistoryMLP(nn.Module):
    """Two-layer diagnostic classifier with a fixed 256-unit hidden layer."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _load_split(data_root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if split not in {"train", "val"}:
        raise ValueError("Test is locked for history identity diagnostics")
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(root / "stage_d/features" / f"{split}.jsonl")
    cache = _load_npz(root / "counterfactual_cache" / f"{split}.npz")
    _validate_rows_cache(rows, cache, split)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows):
        raise ValueError(f"{split} feature rows must explicitly carry policy_split={split}")
    expected = (len(rows), LOGP_DIM)
    for name in ("current_logp_s0", "current_logp_s1"):
        if cache[name].shape != expected or not np.isfinite(cache[name]).all():
            raise ValueError(f"invalid {name} shape/value: {cache[name].shape}")
    if cache.get("label_id", np.asarray([], dtype=np.int64)).shape not in {(0,), (len(rows),)}:
        raise ValueError(f"invalid cache label_id shape for {split}")
    if "label_id" in cache and not np.array_equal(
        np.asarray(cache["label_id"], dtype=np.int64),
        np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64),
    ):
        raise ValueError(f"{split} row/cache label IDs are not aligned")
    for row in rows:
        for name in ("s0_feature", "s1_feature"):
            feature = np.asarray(row[name], dtype=np.float32)
            if feature.shape != (STGCN_FEATURE_DIM + LOGP_DIM + 3,) or not np.isfinite(feature).all():
                raise ValueError(f"invalid {name} in episode {row['episode_id']}: {feature.shape}")
    return rows, cache


def _history_inputs(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    feature_s0 = np.stack([np.asarray(row["s0_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM] for row in rows])
    feature_s1 = np.stack([np.asarray(row["s1_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM] for row in rows])
    logp_s0 = np.asarray(cache["current_logp_s0"], dtype=np.float32)
    logp_s1 = np.asarray(cache["current_logp_s1"], dtype=np.float32)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    return {
        "posterior": np.concatenate([logp_s0, logp_s1], axis=1).astype(np.float32),
        "feature": np.concatenate([feature_s0, feature_s1, logp_s0, logp_s1], axis=1).astype(np.float32),
        "s1_belief": np.exp(logp_s1).astype(np.float32),
        "labels": labels,
    }


def _train_classifier(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
) -> tuple[HistoryMLP, dict[str, Any]]:
    _seed()
    model = HistoryMLP(int(train_inputs.shape[1])).to(device)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_inputs, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_labels, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    loss_history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for inputs, labels in loader:
            logits = model(inputs.to(device=device, dtype=torch.float32))
            loss = criterion(logits, labels.to(device=device, dtype=torch.long))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        loss_history.append(float(np.mean(losses)))
        print(f"history identity epoch {epoch}/{EPOCHS} loss={loss_history[-1]:.6f}", flush=True)
    model.eval()
    with torch.inference_mode():
        train_logits = model(torch.from_numpy(train_inputs).to(device=device, dtype=torch.float32))
        train_predictions = train_logits.argmax(dim=1).cpu().numpy()
    train_accuracy = float(np.mean(train_predictions == train_labels))
    return model, {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "train_accuracy": train_accuracy,
    }


def _predict(model: HistoryMLP, inputs: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits = model(torch.from_numpy(inputs).to(device=device, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
    return predictions.astype(np.int64), probabilities.astype(np.float32)


def _classification_metrics(
    predictions: Sequence[int], labels: Sequence[int], label_names: Sequence[str],
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    if predicted.shape != truth.shape:
        raise ValueError("prediction/label shapes differ")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, guess in zip(truth.tolist(), predicted.tolist()):
        if not 0 <= actual < NUM_CLASSES or not 0 <= guess < NUM_CLASSES:
            raise ValueError("class ID outside reduced14 range")
        matrix[actual, guess] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(label_names):
        tp = float(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted_count = int(matrix[:, class_id].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "class_id": class_id,
            "support": support,
            "accuracy": recall,
            "f1": f1,
        }
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(predicted == truth)) if truth.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _selector_metrics(
    beliefs: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    label_names: Sequence[str],
) -> dict[str, Any]:
    actions = _belief_selector(rows, cache, orders, beliefs)
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    terminal, positives = _terminal_predictions(actions, cache, index, rows)
    stay = sum(action is None for action in actions)
    labels = [int(row["label_id"]) for row in rows]
    return {
        "positive_action_hit_rate": float(np.mean(positives)) if positives else 0.0,
        "positive_action_hit_count": int(sum(positives)),
        "stay_rate": float(stay / len(actions)) if actions else 0.0,
        "action_counts": {"stay": stay, "move": len(actions) - stay},
        "terminal": _classification_metrics(terminal, labels, label_names),
    }


def _load_previous_selector(total: int) -> dict[str, Any]:
    if not PREVIOUS_SELECTOR_RESULT.is_file():
        raise FileNotFoundError(f"missing prior privileged diagnostics: {PREVIOUS_SELECTOR_RESULT}")
    payload = json.loads(PREVIOUS_SELECTOR_RESULT.read_text(encoding="utf-8"))
    if payload.get("test_used") is not False:
        raise ValueError("prior privileged diagnostics are not Val-only")
    methods = payload.get("selector_metrics", {})
    required = ("PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle")
    result: dict[str, Any] = {}
    for name in required:
        item = methods.get(name)
        if item is None or int(item.get("terminal", {}).get("count", -1)) != total:
            raise ValueError(f"missing or misaligned prior selector result: {name}")
        result[name] = item
    return result


def _read_label_names(data_root: Path) -> list[str]:
    path = data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-train/label_mapping.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if set(mapping.values()) != set(range(NUM_CLASSES)):
        raise ValueError(f"label mapping is not contiguous reduced14: {path}")
    return [str(name) for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = data_root.resolve()
    train_rows, train_cache = _load_split(data_root, "train")
    val_rows, val_cache = _load_split(data_root, "val")
    train_inputs = _history_inputs(train_rows, train_cache)
    val_inputs = _history_inputs(val_rows, val_cache)
    label_names = _read_label_names(data_root)
    val_labels = val_inputs["labels"]
    val_orders = _orders(data_root, val_rows)

    posterior_model, posterior_training = _train_classifier(
        train_inputs["posterior"], train_inputs["labels"], device,
    )
    feature_model, feature_training = _train_classifier(
        train_inputs["feature"], train_inputs["labels"], device,
    )
    posterior_predictions, posterior_belief = _predict(posterior_model, val_inputs["posterior"], device)
    feature_predictions, feature_belief = _predict(feature_model, val_inputs["feature"], device)
    s1_predictions = np.argmax(val_cache["current_logp_s1"], axis=1).astype(np.int64)

    classifier_metrics = {
        "S1_only_frozen_ST_GCN": _classification_metrics(s1_predictions, val_labels, label_names),
        "Posterior_history_MLP": _classification_metrics(posterior_predictions, val_labels, label_names),
        "Feature_history_MLP": _classification_metrics(feature_predictions, val_labels, label_names),
    }
    selector_metrics = {
        "S1_only_frozen_ST_GCN": _selector_metrics(val_inputs["s1_belief"], val_rows, val_cache, val_orders, label_names),
        "Posterior_history_MLP": _selector_metrics(posterior_belief, val_rows, val_cache, val_orders, label_names),
        "Feature_history_MLP": _selector_metrics(feature_belief, val_rows, val_cache, val_orders, label_names),
    }
    selector_metrics.update(_load_previous_selector(len(val_rows)))
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_HISTORY_IDENTITY",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows)},
        "label_mapping": {str(index): name for index, name in enumerate(label_names)},
        "classifier_metrics": classifier_metrics,
        "selector_metrics": selector_metrics,
        "training": {
            "Posterior_history_MLP": posterior_training,
            "Feature_history_MLP": feature_training,
        },
        "protocol": {
            "posterior_input_dim": LOGP_DIM * 2,
            "feature_input_dim": STGCN_FEATURE_DIM * 2 + LOGP_DIM * 2,
            "architecture": "Linear(input,256) -> GELU -> Linear(256,14)",
            "loss": "CrossEntropyLoss",
            "candidate_selector": "direct argmax over [Stay, legal candidates] using archived true_logp",
            "candidate_identity_source": "frozen candidate ordering; never true label",
            "candidate_true_logp_role": "privileged offline selector diagnostic only",
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_training": False,
            "true_logp_used_in_classifier_input": False,
            "true_logp_used_only_for_privileged_candidate_selector": True,
            "formal_checkpoint_modified": False,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    labels = result["label_mapping"]
    classifier = result["classifier_metrics"]
    selectors = result["selector_metrics"]
    lines = [
        "# Reduced14 History Action Identity (Val)",
        "",
        "This diagnostic trains two fixed two-layer MLPs on Train contexts and evaluates only the Val moving contexts. Formal WM-E, Joint Revision and ST-GCN checkpoints remain unchanged.",
        "",
        f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}.",
        "",
        "## Frozen ST-GCN / history classifiers",
        "",
        "Per-class `accuracy` is class recall (true positives divided by class support). Each row in `result.json` also contains the full 14x14 confusion matrix.",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for name in ("S1_only_frozen_ST_GCN", "Posterior_history_MLP", "Feature_history_MLP"):
        item = classifier[name]
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |")
    lines.extend([
        "",
        "## Privileged candidate selector",
        "",
        "The selector compares Stay and legal candidates directly. Candidate recognition is archived `true_logp` and is used only for this offline privileged diagnostic; classifier inputs never contain true_logp.",
        "",
        "| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("S1_only_frozen_ST_GCN", "Posterior_history_MLP", "Feature_history_MLP", "PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = selectors[name]
        lines.append(
            f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |"
        )
    posterior_gain = classifier["Posterior_history_MLP"]["accuracy"] - classifier["S1_only_frozen_ST_GCN"]["accuracy"]
    feature_gain = classifier["Feature_history_MLP"]["accuracy"] - classifier["S1_only_frozen_ST_GCN"]["accuracy"]
    selector_gain = selectors["Feature_history_MLP"]["terminal"]["accuracy"] - selectors["S1_only_frozen_ST_GCN"]["terminal"]["accuracy"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Posterior-history accuracy change versus S1-only: {posterior_gain:+.6f}; Feature-history change: {feature_gain:+.6f}.",
        f"Feature-history privileged selector terminal-accuracy change versus S1-only: {selector_gain:+.6f}.",
        "",
        "If Feature-history materially exceeds Posterior-history, frozen ST-GCN features retain identity information that is absent from the final posterior and a learned belief refiner would be justified. If both remain close to S1-only, the next direction should be information-seeking/disambiguation viewpoint selection rather than another history classifier.",
        "",
        "The privileged JR and GT-label Privileged JR rows are reused from the preceding Val-only diagnostic for comparison; they are not retrained by this script.",
        "",
        "Leakage audit: `test_used=false`; no Test path is loaded; Val is not used for training; no formal checkpoint is modified.",
        "",
        f"Classes ({len(labels)}): " + ", ".join(labels[str(index)] for index in range(NUM_CLASSES)),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 history action identity diagnostic")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    requested = torch.device(args.device)
    if requested.type != "cuda":
        raise RuntimeError("History identity diagnostic requires CUDA; CPU execution is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    analyze(args.data_root, requested)


if __name__ == "__main__":
    main()
