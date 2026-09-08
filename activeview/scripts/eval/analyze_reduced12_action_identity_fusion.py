#!/usr/bin/env python3
"""Train/Val-only reduced12 action-identity fusion ablation.

The posterior branches are analytic frozen baselines.  The four MLP branches
are independent small classifiers trained only on the existing moving Train
contexts and selected by moving Val Macro-F1.  No WM, JR, skeleton, RGB or
DINO artifact is changed or regenerated, and policy Test is never read.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.experiments.run_reduced12_action_belief_estimator import (
    _load_dino_store,
)
from activeview.scripts.experiments.run_reduced12_selector_ceiling_decomposition import (
    _cache,
    _metrics,
)


SEED = 42
NUM_CLASSES = 12
HISTORY_FEATURE_DIM = 256
DINO_DIM = 768
EPOCHS = 20
BATCH_SIZE = 512
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/action_identity_fusion_ablation"
)
LABEL_NAMES = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)


class _ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class _IdentityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError(f"expected [B,{self.input_dim}], got {tuple(features.shape)}")
        return self.network(features)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rank_metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    return _metrics(predictions, labels)


def _per_class(predictions: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float | int]]:
    matrix = np.bincount(
        labels * NUM_CLASSES + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    result: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(LABEL_NAMES):
        tp = float(matrix[index, index])
        actual = float(matrix[index, :].sum())
        predicted = float(matrix[:, index].sum())
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {
            "label_id": index,
            "count": int(actual),
            "recall": recall,
            "f1": f1,
        }
    return result


def _feature_rows(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    dino_lookup: Mapping[tuple[str, str, str, int], int],
    dino_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    values: dict[str, list[np.ndarray]] = {name: [] for name in ("feature", "feature_logp", "dino", "all")}
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in cache_index:
            raise ValueError(f"counterfactual cache missing episode: {episode_id}")
        index = cache_index[episode_id]
        features: list[np.ndarray] = []
        logps: list[np.ndarray] = []
        dino_means: list[np.ndarray] = []
        for feature_name, viewpoint_name, suffix in (
            ("s0_feature", "s0_viewpoint_id", "s0"),
            ("s1_feature", "s1_viewpoint_id", "s1"),
        ):
            feature = np.asarray(row[feature_name], dtype=np.float32)
            if feature.shape != (271,):
                raise ValueError(f"expected 271-D Stage-D feature, got {feature.shape}")
            logp = np.asarray(cache[f"current_logp_{suffix}"][index], dtype=np.float32)
            key = (
                str(row["scene_id"]),
                str(row["region"]),
                str(row["record_id"]),
                int(row[viewpoint_name]),
            )
            if key not in dino_lookup:
                raise ValueError(f"DINO cache missing visited key: {key}")
            seen.add(key)
            dino_means.append(np.asarray(dino_embeddings[dino_lookup[key]], dtype=np.float32).mean(axis=0))
            features.append(feature[:HISTORY_FEATURE_DIM])
            logps.append(logp)
        feature_delta = features[1] - features[0]
        logp_delta = logps[1] - logps[0]
        dino_delta = dino_means[1] - dino_means[0]
        values["feature"].append(np.concatenate([features[0], features[1], feature_delta]))
        values["feature_logp"].append(np.concatenate([
            features[0], logps[0], features[1], logps[1], feature_delta, logp_delta,
        ]))
        values["dino"].append(np.concatenate([dino_means[0], dino_means[1], dino_delta]))
        values["all"].append(np.concatenate([
            np.concatenate([features[0], logps[0], dino_means[0]]),
            np.concatenate([features[1], logps[1], dino_means[1]]),
            np.concatenate([feature_delta, logp_delta, dino_delta]),
        ]))
    arrays = {name: np.asarray(items, dtype=np.float32) for name, items in values.items()}
    arrays["unique_visited_observations"] = np.asarray([len(seen)], dtype=np.int64)
    return arrays


def _infer_model(
    model: _IdentityMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    normalized = ((features - mean) / std).astype(np.float32)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_branch(
    branch: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed()
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized_train = ((train_features - mean) / std).astype(np.float32)
    normalized_val = ((val_features - mean) / std).astype(np.float32)
    model = _IdentityMLP(train_features.shape[1]).to(device)
    loader = DataLoader(_ArrayDataset(normalized_train, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1 = -1.0
    best_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            logits = model(features.float().to(device))
            loss = nn.functional.cross_entropy(logits, labels.long().to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = _infer_model(
            model,
            val_features,
            mean,
            std,
            device=device,
            batch_size=batch_size,
        )
        val_predictions = np.argmax(val_logits, axis=1)
        val_metrics = _rank_metrics(val_predictions.tolist(), val_labels.tolist())
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(record)
        print(
            f"identity branch={branch} epoch={epoch}/{EPOCHS} "
            f"loss={record['train_loss']:.6f} val_acc={record['val_accuracy']:.6f} "
            f"val_f1={record['val_macro_f1']:.6f}",
            flush=True,
        )
        if (float(record["val_macro_f1"]), float(record["val_accuracy"])) > (best_f1, best_accuracy):
            best_f1 = float(record["val_macro_f1"])
            best_accuracy = float(record["val_accuracy"])
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_std": std,
                "input_dim": int(train_features.shape[1]),
                "hidden_dim": 256,
                "num_classes": NUM_CLASSES,
                "seed": SEED,
                "best_epoch": best_epoch,
                "branch": branch,
            }, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    logits = _infer_model(
        model,
        val_features,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )
    summary = {
        "branch": branch,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "hidden_dim": 256,
        "input_dim": int(train_features.shape[1]),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_macro_f1": best_f1,
        "final_train_loss": history[-1]["train_loss"],
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return logits, summary


def _metrics_bundle(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    base = _rank_metrics(predictions.tolist(), labels.tolist())
    return {
        **base,
        "per_class": _per_class(predictions, labels),
    }


def _posterior_baselines(
    train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_logp0 = np.asarray(train_cache["current_logp_s0"], dtype=np.float32)
    train_logp1 = np.asarray(train_cache["current_logp_s1"], dtype=np.float32)
    val_logp0 = np.asarray(val_cache["current_logp_s0"], dtype=np.float32)
    val_logp1 = np.asarray(val_cache["current_logp_s1"], dtype=np.float32)
    train_p0, train_p1 = np.exp(train_logp0), np.exp(train_logp1)
    val_p0, val_p1 = np.exp(val_logp0), np.exp(val_logp1)
    train_confidence = np.max(train_p0, axis=1) >= np.max(train_p1, axis=1)
    val_confidence = np.max(val_p0, axis=1) >= np.max(val_p1, axis=1)
    return {
        "S1-only": (train_p1, val_p1),
        "Mean posterior": (0.5 * (train_p0 + train_p1), 0.5 * (val_p0 + val_p1)),
        "Product-of-Evidence": (
            np.exp(train_logp0 + train_logp1),
            np.exp(val_logp0 + val_logp1),
        ),
        "Confidence-select": (
            np.where(train_confidence[:, None], train_p0, train_p1),
            np.where(val_confidence[:, None], val_p0, val_p1),
        ),
    }


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    order = result["method_order"]
    methods = result["methods"]
    lines = [
        "# Reduced12 action identity fusion ablation",
        "",
        "Train/Val only on the existing moving contexts. No policy Test was read; no WM/JR/skeleton/RGB/DINO artifact was modified or regenerated.",
        "",
        "## Val Moving comparison",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAccuracy vs S1 | ΔMacro-F1 vs S1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        value = methods[name]
        lines.append(
            f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} | "
            f"{result['relative_to_s1_pp'][name]['accuracy_pp']:.3f} | "
            f"{result['relative_to_s1_pp'][name]['macro_f1_pp']:.3f} |"
        )
    lines.extend([
        "",
        "## Focus actions",
        "",
        "| Method | bend Recall/F1 | stumble Recall/F1 | knock Recall/F1 | touching face Recall/F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in order:
        per_class = methods[name]["per_class"]
        focus_values = []
        for label in ("bend", "stumble", "knock", "touching face"):
            focus_values.append(f"{per_class[label]['recall']:.3f}/{per_class[label]['f1']:.3f}")
        lines.append(f"| {name} | " + " | ".join(focus_values) + " |")
    lines.extend(["", "## Scientific interpretation", ""])
    lines.append(result["scientific_conclusion"]["posterior_fusion"])
    lines.append(result["scientific_conclusion"]["feature_source"])
    lines.append(result["scientific_conclusion"]["dino_effect"])
    lines.append(result["scientific_conclusion"]["best_deployable"])
    lines.append(result["scientific_conclusion"]["identity_ceiling"])
    lines.extend([
        "",
        "## Protocol",
        "",
        "- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)",
        "- MLP: two layers, hidden=256, GELU, CrossEntropy, seed=42, 20 epochs, best moving-Val Macro-F1",
        "- normalization: Train-only mean/std for each MLP branch",
        "- `test_used=false`; no Test path or artifact was read",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device, output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rgb_root = data_root / "datasets/policy_reduced12_eight_placement_v1_rgb_restored"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    train_cache = _cache(rgb_root / "counterfactual_cache/train.npz")
    val_cache = _cache(rgb_root / "counterfactual_cache/val.npz")
    train_ids = {str(value) for value in train_cache["episode_ids"].tolist()}
    val_ids = {str(value) for value in val_cache["episode_ids"].tolist()}
    if train_ids != {str(row["episode_id"]) for row in train_rows}:
        raise ValueError("Train rows and counterfactual cache IDs are not aligned")
    if val_ids != {str(row["episode_id"]) for row in val_rows}:
        raise ValueError("Val rows and counterfactual cache IDs are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root)
    train_modalities = _feature_rows(train_rows, train_cache, dino_lookup, dino_embeddings)
    val_modalities = _feature_rows(val_rows, val_cache, dino_lookup, dino_embeddings)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    if train_modalities["feature"].shape[0] != len(train_rows) or val_modalities["feature"].shape[0] != len(val_rows):
        raise ValueError("modalities and rows are not aligned")

    method_order = [
        "S1-only", "Mean posterior", "Product-of-Evidence", "Confidence-select",
        "ST-GCN feature-only MLP", "ST-GCN feature + logp MLP", "DINO-only MLP",
        "Current all-feature MLP",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    method_logits: dict[str, np.ndarray] = {}
    training: dict[str, Any] = {}
    analytic = _posterior_baselines(train_cache, val_cache)
    for name, (_, val_probability) in analytic.items():
        method_logits[name] = np.log(np.clip(val_probability, 1e-12, 1.0))
    branch_inputs = {
        "ST-GCN feature-only MLP": "feature",
        "ST-GCN feature + logp MLP": "feature_logp",
        "DINO-only MLP": "dino",
        "Current all-feature MLP": "all",
    }
    checkpoint_root = data_root / "checkpoints/activeview_reduced12_action_identity_fusion_ablation"
    for method, modality in branch_inputs.items():
        logits, summary = _train_branch(
            _safe_name(method),
            train_modalities[modality],
            train_labels,
            val_modalities[modality],
            val_labels,
            device=device,
            checkpoint=checkpoint_root / f"{_safe_name(method)}.pth",
            summary_path=output_dir / f"training_{_safe_name(method)}.json",
            batch_size=batch_size,
        )
        method_logits[method] = logits
        training[method] = summary

    metrics: dict[str, dict[str, Any]] = {}
    confusion: dict[str, np.ndarray] = {}
    s1_predictions = np.argmax(method_logits["S1-only"], axis=1)
    for method in method_order:
        predictions = np.argmax(method_logits[method], axis=1)
        metrics[method] = _metrics_bundle(predictions, val_labels)
        confusion[_safe_name(method)] = np.bincount(
            val_labels * NUM_CLASSES + predictions,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES).astype(np.int64)
    np.savez_compressed(output_dir / "confusion_matrices.npz", **confusion)
    s1 = metrics["S1-only"]
    relative: dict[str, dict[str, float]] = {}
    for method in method_order:
        relative[method] = {
            "accuracy_pp": 100.0 * (float(metrics[method]["accuracy"]) - float(s1["accuracy"])),
            "macro_f1_pp": 100.0 * (float(metrics[method]["macro_f1"]) - float(s1["macro_f1"])),
        }
    mlp_order = list(branch_inputs)
    best_method = max(method_order, key=lambda name: (float(metrics[name]["macro_f1"]), float(metrics[name]["accuracy"])))
    feature_only = metrics["ST-GCN feature-only MLP"]
    feature_logp = metrics["ST-GCN feature + logp MLP"]
    dino_only = metrics["DINO-only MLP"]
    all_feature = metrics["Current all-feature MLP"]
    posterior_best = max((metrics[name] for name in analytic), key=lambda value: (float(value["macro_f1"]), float(value["accuracy"])))
    posterior_best_name = max(analytic, key=lambda name: (float(metrics[name]["macro_f1"]), float(metrics[name]["accuracy"])))
    if float(posterior_best["macro_f1"]) > float(s1["macro_f1"]):
        posterior_conclusion = f"Simple posterior fusion helps: {posterior_best_name} is the best analytic branch and improves S1-only by {relative[posterior_best_name]['macro_f1_pp']:.3f} pp Macro-F1."
    else:
        posterior_conclusion = "No analytic posterior fusion branch improves S1-only Macro-F1."
    feature_conclusion = f"ST-GCN feature-only reaches {feature_only['accuracy']:.6f}/{feature_only['macro_f1']:.6f}; adding logp changes it to {feature_logp['accuracy']:.6f}/{feature_logp['macro_f1']:.6f}."
    if float(all_feature["macro_f1"]) > float(feature_logp["macro_f1"]):
        dino_conclusion = f"DINO is complementary in the all-feature branch (all-feature Macro-F1 {all_feature['macro_f1']:.6f} vs feature+logp {feature_logp['macro_f1']:.6f}); DINO-only is {dino_only['macro_f1']:.6f}."
    else:
        dino_conclusion = f"DINO does not improve the all-feature branch over feature+logp (all-feature Macro-F1 {all_feature['macro_f1']:.6f} vs {feature_logp['macro_f1']:.6f}); DINO-only is {dino_only['macro_f1']:.6f}."
    conclusion = {
        "posterior_fusion": posterior_conclusion,
        "feature_source": feature_conclusion,
        "dino_effect": dino_conclusion,
        "best_deployable": f"The best Val-moving branch is {best_method} with Accuracy/Macro-F1 {metrics[best_method]['accuracy']:.6f}/{metrics[best_method]['macro_f1']:.6f}; this is a single-seed Val comparison, not Test evidence.",
        "identity_ceiling": f"Current all-feature MLP reaches {all_feature['accuracy']:.6f}/{all_feature['macro_f1']:.6f}; compare this with the prior approximately 55% identity result without claiming generalization beyond Val.",
    }
    result = {
        "experiment_id": "REDUCED12_ACTION_IDENTITY_FUSION_ABLATION",
        "status": "COMPLETED",
        "test_used": False,
        "training_performed": True,
        "population": {
            "train_moving_contexts": len(train_rows),
            "val_moving_contexts": len(val_rows),
            "train_unique_visited_observations": int(train_modalities["unique_visited_observations"][0]),
            "val_unique_visited_observations": int(val_modalities["unique_visited_observations"][0]),
        },
        "label_names": list(LABEL_NAMES),
        "method_order": method_order,
        "methods": metrics,
        "focus_actions": {
            label: {method: metrics[method]["per_class"][label] for method in method_order}
            for label in ("bend", "stumble", "knock", "touching face")
        },
        "relative_to_s1_pp": relative,
        "training": training,
        "dino_cache_summary": dino_summary,
        "scientific_conclusion": conclusion,
        "protocol": {
            "loss": "CrossEntropy",
            "mlp": "Linear(input,256) -> GELU -> Linear(256,12)",
            "epochs": EPOCHS,
            "seed": SEED,
            "selection": "best moving-Val Macro-F1 then Accuracy",
            "normalization": "Train-only mean/std per branch",
            "wm_or_jr_modified": False,
        },
        "leakage_flags": {
            "test_used": False,
            "test_paths_read": False,
            "future_candidate_rgb_used": False,
            "future_candidate_dino_used": False,
            "wm_modified": False,
            "jr_modified": False,
            "skeleton_regenerated": False,
            "rgb_regenerated": False,
            "dino_regenerated": False,
        },
        "artifacts": {
            "per_class_metrics": str((output_dir / "per_class_metrics.json").resolve()),
            "confusion_matrices": str((output_dir / "confusion_matrices.npz").resolve()),
        },
    }
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps({"label_names": list(LABEL_NAMES), "methods": metrics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    result = run(args.data_root, torch.device(args.device), args.output_dir, batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"]}, indent=2))


if __name__ == "__main__":
    main()
