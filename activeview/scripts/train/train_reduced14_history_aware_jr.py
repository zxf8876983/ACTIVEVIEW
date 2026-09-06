#!/usr/bin/env python3
"""Train the reduced14 history-aware Multi-positive Joint Revision.

This is the first formal JR-only method extension.  WM-E and ST-GCN remain
frozen.  The new checkpoint is selected on Val terminal Accuracy after each
Train epoch; Test is intentionally inaccessible from this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.joint_revision.history_aware import HistoryAwareJointRevision
from activeview.methods.joint_revision.model import JointRevision, select_actions
from activeview.scripts.train.train_reduced14_joint_revision import _examples, _orders
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _metrics,
    _terminal_predictions,
    _validate_rows_cache,
)


SEED = 42
NUM_CLASSES = 14
STGCN_FEATURE_DIM = 256
HISTORY_INPUT_DIM = 2 * STGCN_FEATURE_DIM + 2 * NUM_CLASSES
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_ID = 0.2
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_aware_jr_v1"
CHECKPOINT_NAME = "history_aware_joint_revision_best.pth"
PRIOR_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_belief_fusion/result.json"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _HistoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: tuple[np.ndarray, ...]) -> None:
        self.current, self.candidates, self.mask, self.fallback, self.positive, self.labels, self.history = arrays

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current": torch.from_numpy(self.current[index]),
            "candidates": torch.from_numpy(self.candidates[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "fallback": torch.tensor(self.fallback[index]),
            "positive": torch.from_numpy(self.positive[index]),
            "label": torch.tensor(self.labels[index]),
            "history": torch.from_numpy(self.history[index]),
        }


def _history_inputs(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> np.ndarray:
    feature_s0 = np.stack([np.asarray(row["s0_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM] for row in rows])
    feature_s1 = np.stack([np.asarray(row["s1_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM] for row in rows])
    logp_s0 = np.asarray(cache["current_logp_s0"], dtype=np.float32)
    logp_s1 = np.asarray(cache["current_logp_s1"], dtype=np.float32)
    history = np.concatenate([feature_s0, feature_s1, logp_s0, logp_s1], axis=1)
    if history.shape != (len(rows), HISTORY_INPUT_DIM) or not np.isfinite(history).all():
        raise ValueError(f"invalid history input shape/value: {history.shape}")
    return history


def _loss(
    scores: torch.Tensor,
    posterior: torch.Tensor,
    identity_logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = scores.device
    mask = batch["mask"].bool().to(device)
    fallback = batch["fallback"].long().to(device)
    positive = batch["positive"].float().to(device)
    labels = batch["label"].long().to(device)
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
    jr_loss = (main + 0.25 * bce + 0.05 * post).mean()
    identity_loss = nn.functional.cross_entropy(identity_logits, labels)
    return jr_loss + LAMBDA_ID * identity_loss, jr_loss.detach(), identity_loss.detach()


def _classification_metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(truth * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0
        recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"count": int(truth.size), "accuracy": float(np.mean(pred == truth)), "macro_f1": float(np.mean(f1))}


def _select_history_actions(
    model: HistoryAwareJointRevision,
    arrays: tuple[np.ndarray, ...],
    rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]],
    device: torch.device,
) -> tuple[list[int | None], np.ndarray]:
    current, candidates, mask, _fallback, _positive, _labels, history = arrays
    actions: list[int | None] = []
    identity_predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(rows))
            scores, _posterior, identity_logits = model(
                torch.from_numpy(current[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(candidates[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(mask[start:stop]).to(device=device, dtype=torch.bool),
                torch.from_numpy(history[start:stop]).to(device=device, dtype=torch.float32),
            )
            choice = scores.masked_fill(~torch.from_numpy(mask[start:stop]).to(device), -1e9).argmax(dim=1).cpu().numpy()
            identity_predictions.append(identity_logits.argmax(dim=1).cpu().numpy())
            for offset, selected in enumerate(choice.tolist()):
                order = list(orders[str(rows[start + offset]["episode_id"])])
                actions.append(None if selected == 0 else int(order[selected - 1]))
    return actions, np.concatenate(identity_predictions, axis=0)


def _method_metrics(
    actions: Sequence[int | None], rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    predictions, positives = _terminal_predictions(actions, cache, index, rows)
    stay = sum(action is None for action in actions)
    return {
        "positive_action_hit_rate": float(np.mean(positives)),
        "positive_action_hit_count": int(sum(positives)),
        "stay_rate": float(stay / len(actions)),
        "action_counts": {"stay": stay, "move": len(actions) - stay},
        "terminal": _metrics(predictions, [int(row["label_id"]) for row in rows]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prior_methods(total: int) -> dict[str, Any]:
    payload = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    if payload.get("test_used") is not False:
        raise ValueError("prior comparison result is not Val-only")
    methods = payload.get("selector_metrics", {})
    result: dict[str, Any] = {}
    for name in ("PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = methods.get(name)
        if item is None or int(item["terminal"]["count"]) != total:
            raise ValueError(f"missing or misaligned prior method: {name}")
        result[name] = item
    return result


def _load_normal_jr(
    data_root: Path, val_rows: Sequence[Mapping[str, Any]], val_cache: Mapping[str, np.ndarray],
    val_orders: Mapping[str, Sequence[int]], device: torch.device,
) -> dict[str, Any]:
    checkpoint = data_root / "checkpoints/activeview_reduced14_eight_placement_v1/joint_revision_multi_positive.pth"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = JointRevision(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload["state_dict"]))
    model.eval()
    actions = select_actions(model, val_cache, val_rows, val_orders, budget="ALL_LEGAL", device=device)
    return _method_metrics(actions, val_rows, val_cache)


def train_and_evaluate(data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    root = data_root.resolve() / "datasets/policy_reduced14_kneel_eight_placement_v1"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    train_cache = _load_npz(root / "counterfactual_cache/train.npz")
    val_cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate_rows_cache(train_rows, train_cache, "train")
    _validate_rows_cache(val_rows, val_cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val rows must explicitly carry policy_split=val")
    train_orders = _orders(data_root, train_rows)
    val_orders = _orders(data_root, val_rows)
    train_base, train_stats = _examples(train_rows, train_cache, train_orders)
    val_base, _ = _examples(val_rows, val_cache, val_orders)
    train_history = _history_inputs(train_rows, train_cache)
    val_history = _history_inputs(val_rows, val_cache)
    train_arrays = (*train_base, train_history)
    val_arrays = (*val_base, val_history)
    loader = DataLoader(_HistoryDataset(train_arrays), batch_size=BATCH_SIZE, shuffle=True)
    model = HistoryAwareJointRevision(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint = data_root.resolve() / "checkpoints/activeview_reduced14_eight_placement_v1" / CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_key = (-np.inf, -np.inf)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_losses: list[float] = []
        jr_losses: list[float] = []
        identity_losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            history_input = batch["history"].float().to(device)
            scores, posterior, identity_logits = model(current, candidates, mask, history_input)
            loss, jr_loss, identity_loss = _loss(scores, posterior, identity_logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_losses.append(float(loss.detach().cpu()))
            jr_losses.append(float(jr_loss.cpu()))
            identity_losses.append(float(identity_loss.cpu()))
        actions, identity_predictions = _select_history_actions(model, val_arrays, val_rows, val_orders, device)
        terminal = _method_metrics(actions, val_rows, val_cache)
        val_identity = _classification_metrics(identity_predictions, val_base[-1])
        epoch_stats = {
            "epoch": epoch,
            "loss": float(np.mean(total_losses)),
            "jr_loss": float(np.mean(jr_losses)),
            "identity_loss": float(np.mean(identity_losses)),
            "val_accuracy": float(terminal["terminal"]["accuracy"]),
            "val_macro_f1": float(terminal["terminal"]["macro_f1"]),
            "val_identity_accuracy": float(val_identity["accuracy"]),
            "val_identity_macro_f1": float(val_identity["macro_f1"]),
        }
        history.append(epoch_stats)
        print(
            f"history-aware JR epoch {epoch}/{EPOCHS} loss={epoch_stats['loss']:.6f} "
            f"val_acc={epoch_stats['val_accuracy']:.6f} val_id={epoch_stats['val_identity_accuracy']:.6f}",
            flush=True,
        )
        key = (epoch_stats["val_accuracy"], epoch_stats["val_macro_f1"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "epoch": epoch,
                    "lambda_id": LAMBDA_ID,
                    "history_input_dim": HISTORY_INPUT_DIM,
                    "val_terminal": terminal["terminal"],
                    "val_identity": val_identity,
                },
                checkpoint,
            )

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    actions, identity_predictions = _select_history_actions(model, val_arrays, val_rows, val_orders, device)
    labels = val_base[-1]
    history_method = _method_metrics(actions, val_rows, val_cache)
    identity_metrics = _classification_metrics(identity_predictions, labels)
    normal_method = _load_normal_jr(data_root.resolve(), val_rows, val_cache, val_orders, device)
    current_predictions = np.argmax(val_cache["current_logp_s1"], axis=1)
    frozen_method = {
        "positive_action_hit_rate": float(np.mean(current_predictions == labels)),
        "positive_action_hit_count": int(np.sum(current_predictions == labels)),
        "stay_rate": 1.0,
        "action_counts": {"stay": len(val_rows), "move": 0},
        "terminal": _metrics(current_predictions, labels),
    }
    methods = {
        "FrozenStageCv0": frozen_method,
        "NormalMultiPositiveJR": normal_method,
        "HistoryAwareMultiPositiveJR": history_method,
        **_load_prior_methods(len(val_rows)),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_HISTORY_AWARE_JR_V1",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows)},
        "methods": methods,
        "history_identity_head": identity_metrics,
        "training": {
            "contexts": len(train_rows),
            "positive_contexts": train_stats["positive_contexts"],
            "multi_positive_contexts": train_stats["multi_positive_contexts"],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lambda_id": LAMBDA_ID,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_key[0],
            "best_val_macro_f1": best_key[1],
            "loss_history": history,
            "final_total_loss": history[-1]["loss"],
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "protocol": {
            "history_input": "s0/s1 256-D ST-GCN features + s0/s1 14-D logp",
            "history_encoder": "Linear(540,256) -> GELU -> Linear(256,128) -> GELU",
            "identity_head": "Linear(128,14)",
            "jr_architecture": "existing Multi-positive JointRevision with history latent concatenated to current token",
            "loss": "L_JR + 0.2 * CrossEntropy(refined_action_logits, label)",
            "candidate_budget": "ALL_LEGAL",
            "terminal_recognition": "real archived skeleton through frozen ST-GCN",
            "checkpoint_selection": "Val terminal Accuracy, then Macro-F1",
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_training": False,
            "formal_wm_e_modified": False,
            "formal_stgcn_modified": False,
            "old_jr_checkpoint_modified": False,
            "true_future_recognition_as_model_input": False,
            "candidate_identity_changed": False,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["methods"]
    baseline = methods["NormalMultiPositiveJR"]["terminal"]
    proposed = methods["HistoryAwareMultiPositiveJR"]["terminal"]
    identity = result["history_identity_head"]
    lines = [
        "# History-aware Multi-positive JR v1 (Val)",
        "",
        "The JR branch adds a 540→256→128 history identity encoder and a 14-way auxiliary identity head. WM-E, ST-GCN, taxonomy, split and the existing Multi-positive JR checkpoint remain frozen.",
        "",
        f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}.",
        "",
        "| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("FrozenStageCv0", "NormalMultiPositiveJR", "HistoryAwareMultiPositiveJR", "PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = methods[name]
        lines.append(f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |")
    lines.extend([
        "",
        f"History identity head: Accuracy={identity['accuracy']:.6f}, Macro-F1={identity['macro_f1']:.6f}.",
        f"History-aware JR minus normal JR: ΔAccuracy={proposed['accuracy'] - baseline['accuracy']:+.6f}, ΔMacro-F1={proposed['macro_f1'] - baseline['macro_f1']:+.6f}.",
        f"Normal JR→Privileged JR Accuracy gap: {methods['PrivilegedJR']['terminal']['accuracy'] - baseline['accuracy']:+.6f}; history-aware JR→Privileged JR gap: {methods['PrivilegedJR']['terminal']['accuracy'] - proposed['accuracy']:+.6f}.",
        "",
        "Interpretation: the history identity branch is useful if it exceeds the normal JR on the frozen Val protocol and its identity head is materially above S1-only. Any improvement here is a JR-only Train/Val result; no formal Test claim is made.",
        "",
        "Leakage audit: `test_used=false`; no Test artifact/path is accessed, Val is used only for checkpoint selection/evaluation, and the old WM-E/JR/ST-GCN artifacts are unchanged.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reduced14 history-aware Multi-positive JR on Train and select on Val")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("History-aware JR requires CUDA; CPU fallback is disabled")
    train_and_evaluate(args.data_root, device)


if __name__ == "__main__":
    main()
