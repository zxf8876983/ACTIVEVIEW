#!/usr/bin/env python3
"""Val diagnostic for a GT-label-conditioned privileged Joint Revision.

The diagnostic model receives the archived candidate ``true_logp`` and a
ground-truth 14-way label one-hot in addition to the existing current-state
features.  It is trained on Train only and evaluated on Val moving contexts;
no formal checkpoint is replaced and Test is intentionally inaccessible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.joint_revision.model import JointRevision, tie_argmax
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _JRDataset,
    _build_examples,
    _load_npz,
    _orders,
    _terminal_predictions,
    _validate_rows_cache,
)


SEED = 42
NUM_CLASSES = 14
VIEW_COUNT = 32
EPOCHS = 20
BATCH_SIZE = 512
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/gt_label_privileged_jr"
PREVIOUS_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/selector_bottleneck/result.json"


class GTLabelPrivilegedJR(JointRevision):
    """Existing JR architecture with a diagnostic 14-D label input."""

    def __init__(self) -> None:
        super().__init__(num_classes=NUM_CLASSES)
        self.current_dim = 2 * NUM_CLASSES + 6 + NUM_CLASSES
        self.current_projector = nn.Sequential(nn.Linear(self.current_dim, 128), nn.GELU())


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _with_label_input(arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    current, candidates, mask, fallback, positive, labels = arrays
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[labels]
    return (np.concatenate([current, one_hot], axis=1), candidates, mask, fallback, positive, labels)


def _train_gt(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], device: torch.device,
) -> tuple[GTLabelPrivilegedJR, dict[str, Any]]:
    _seed()
    privileged_cache = dict(cache)
    privileged_cache["imagined_logp"] = np.asarray(cache["true_logp"], dtype=np.float32)
    arrays, stats = _build_examples(rows, privileged_cache, orders)
    arrays = _with_label_input(arrays)
    loader = DataLoader(_JRDataset(arrays), batch_size=BATCH_SIZE, shuffle=True)
    model = GTLabelPrivilegedJR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            fallback = batch["fallback"].long().to(device)
            positive = batch["positive"].float().to(device)
            labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask)
            valid_scores = scores.masked_fill(~mask, -1e9)
            all_lse = torch.logsumexp(valid_scores, dim=1)
            positive_mask = positive.bool() & mask
            positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1)
            has_positive = positive_mask.any(dim=1)
            main = torch.where(
                has_positive,
                all_lse - positive_lse,
                nn.functional.cross_entropy(valid_scores, fallback, reduction="none"),
            )
            bce_all = nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none")
            bce = (bce_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            target = labels.unsqueeze(1).expand(-1, posterior.size(1))
            post_all = nn.functional.cross_entropy(
                posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none"
            ).reshape_as(scores)
            post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (main + 0.25 * bce + 0.05 * post).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        print(f"GT-label privileged JR epoch {epoch}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    model.eval()
    return model, {**stats, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "final_loss": history[-1], "loss_history": history}


def _current_stats(cache: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    logp0 = np.asarray(cache["current_logp_s0"][index], dtype=np.float32)
    logp1 = np.asarray(cache["current_logp_s1"][index], dtype=np.float32)
    probs0, probs1 = np.exp(logp0), np.exp(logp1)
    return np.asarray(
        [
            *probs0,
            *probs1,
            -np.sum(probs0 * logp0),
            -np.sum(probs1 * logp1),
            np.max(probs0),
            np.max(probs1),
            np.sort(probs1)[-1] - np.sort(probs1)[-2],
            np.sort(probs0)[-1] - np.sort(probs0)[-2],
        ],
        dtype=np.float32,
    )


def _select_gt(
    model: GTLabelPrivilegedJR, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], device: torch.device,
) -> list[int | None]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    selected: list[int | None] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        label = int(row["label_id"])
        current = np.asarray(cache["current_logp_s1"][i], dtype=np.float32)
        values = [np.concatenate([current, np.zeros(9, dtype=np.float32), [1.0]])]
        values.extend(
            np.concatenate([cache["true_logp"][i, candidate], cache["candidate_descriptor"][i, candidate], [0.0]])
            for candidate in candidates
        )
        current_augmented = np.concatenate([_current_stats(cache, i), np.eye(NUM_CLASSES, dtype=np.float32)[label]])
        tensor = torch.zeros((1, max(31, len(values)), model.candidate_dim), dtype=torch.float32)
        tensor[0, : len(values)] = torch.from_numpy(np.asarray(values, dtype=np.float32))
        mask = torch.zeros((1, tensor.shape[1]), dtype=torch.bool)
        mask[0, : len(values)] = True
        with torch.inference_mode():
            scores, _ = model(
                torch.from_numpy(current_augmented[None]).to(device),
                tensor.to(device),
                mask.to(device),
            )
        choice = tie_argmax(scores[0, : len(values)].cpu().numpy())
        selected.append(None if choice == 0 else candidates[choice - 1])
    return selected


def _method_from_previous(payload: Mapping[str, Any], name: str, total: int) -> dict[str, Any]:
    item = payload[name]
    return {
        "positive_action_hit_rate": float(item["positive_hit_rate"]),
        "stay_rate": float(item["action_counts"]["stay"] / total),
        "action_counts": item["action_counts"],
        "terminal": item["terminal"],
        "source": str(PREVIOUS_RESULT.resolve()),
    }


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    data_root = data_root.resolve()
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    train_cache = _load_npz(root / "counterfactual_cache/train.npz")
    val_cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate_rows_cache(train_rows, train_cache, "train")
    _validate_rows_cache(val_rows, val_cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("GT-label diagnostic requires explicit Val rows")
    train_orders = _orders(data_root, train_rows)
    val_orders = _orders(data_root, val_rows)
    model, training = _train_gt(train_rows, train_cache, train_orders, device)
    actions = _select_gt(model, val_rows, val_cache, val_orders, device)
    predictions, positives = _terminal_predictions(
        actions,
        val_cache,
        {str(value): i for i, value in enumerate(val_cache["episode_ids"].tolist())},
        val_rows,
    )
    labels = [int(row["label_id"]) for row in val_rows]
    from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import _metrics
    gt_result = {
        "positive_action_hit_rate": float(np.mean(positives)),
        "positive_action_hit_count": int(sum(positives)),
        "stay_rate": float(sum(action is None for action in actions) / len(actions)),
        "action_counts": {"stay": sum(action is None for action in actions), "move": sum(action is not None for action in actions)},
        "terminal": _metrics(predictions, labels),
    }
    previous = json.loads(PREVIOUS_RESULT.read_text(encoding="utf-8"))
    if previous.get("test_used") is not False:
        raise ValueError("previous selector diagnostics are not Val-only")
    methods = {
        "NormalJR": _method_from_previous(previous, "normal_jr", len(val_rows)),
        "PrivilegedJR": _method_from_previous(previous, "privileged_jr", len(val_rows)),
        "GTLabelPrivilegedJR": gt_result,
        "SafeOracle": _method_from_previous(previous, "safe_oracle", len(val_rows)),
    }
    gt = methods["GTLabelPrivilegedJR"]["terminal"]
    privileged = methods["PrivilegedJR"]["terminal"]
    normal = methods["NormalJR"]["terminal"]
    safe = methods["SafeOracle"]["terminal"]
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_GT_LABEL_PRIVILEGED_JR",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows)},
        "methods": methods,
        "gaps": {
            "privileged_jr_to_gt_label_privileged_jr": {
                "accuracy": float(gt["accuracy"] - privileged["accuracy"]),
                "macro_f1": float(gt["macro_f1"] - privileged["macro_f1"]),
            },
            "gt_label_privileged_jr_to_safe_oracle": {
                "accuracy": float(safe["accuracy"] - gt["accuracy"]),
                "macro_f1": float(safe["macro_f1"] - gt["macro_f1"]),
            },
            "normal_jr_to_privileged_jr": {
                "accuracy": float(privileged["accuracy"] - normal["accuracy"]),
                "macro_f1": float(privileged["macro_f1"] - normal["macro_f1"]),
            },
        },
        "training": training,
        "protocol": {
            "taxonomy": "reduced14_kneel",
            "candidate_input": "archived true_logp substituted for imagined_logp",
            "extra_input": "ground-truth label one-hot, 14-D",
            "architecture": "JointRevision with current projector input 48-D; candidate/Transformer heads unchanged",
            "terminal_recognition": "archived true candidate/current logp through frozen ST-GCN",
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_training": False,
            "formal_checkpoints_modified": False,
            "true_logp_used_as_privileged_diagnostic_input": True,
            "ground_truth_label_used_as_privileged_diagnostic_input": True,
            "true_logp_used_for_candidate_identity": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["methods"]
    gaps = result["gaps"]
    gt_gap = gaps["gt_label_privileged_jr_to_safe_oracle"]["accuracy"]
    if abs(float(gt_gap)) < 0.05:
        verdict = "GT-label Privileged JR is close to SafeOracle; the dominant remaining limitation is action belief/identity inference."
    else:
        verdict = "GT-label Privileged JR remains clearly below SafeOracle; JR selector/objective capacity still has a substantial independent bottleneck."
    lines = [
        "# Reduced14 GT-label Privileged JR (Val)",
        "",
        "This is a diagnostic-only model trained on Train contexts and evaluated on Val moving contexts. It receives archived candidate true_logp plus a 14-D ground-truth label one-hot; no formal checkpoint is changed.",
        "",
        "| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("NormalJR", "PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = methods[name]
        lines.append(f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |")
    lines.extend(
        [
            "",
            f"Privileged JR → GT-label Privileged JR: ΔAccuracy={gaps['privileged_jr_to_gt_label_privileged_jr']['accuracy']:+.6f}, ΔMacro-F1={gaps['privileged_jr_to_gt_label_privileged_jr']['macro_f1']:+.6f}.",
            f"GT-label Privileged JR → SafeOracle gap: Accuracy={gaps['gt_label_privileged_jr_to_safe_oracle']['accuracy']:.6f}, Macro-F1={gaps['gt_label_privileged_jr_to_safe_oracle']['macro_f1']:.6f}.",
            f"Normal JR → Privileged JR: ΔAccuracy={gaps['normal_jr_to_privileged_jr']['accuracy']:+.6f}, ΔMacro-F1={gaps['normal_jr_to_privileged_jr']['macro_f1']:+.6f}.",
            "",
            "## Interpretation",
            "",
            verdict,
            "The comparison is not a new formal method: archived recognition and the GT label are privileged inputs used only to locate the ceiling. A large Normal→Privileged gain indicates that candidate recognition/belief is important; a remaining GT-label→SafeOracle gap indicates selector/objective limitations after identity inference is removed.",
            "",
            "Leakage audit: `test_used=false`; Train was used only for the diagnostic fit, Val only for evaluation, and no formal WM-E/JR/ST-GCN checkpoint was modified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 GT-label privileged JR diagnostic")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GT-label privileged JR requires GPU")
    analyze(args.data_root, device)


if __name__ == "__main__":
    main()
