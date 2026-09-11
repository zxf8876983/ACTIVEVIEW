#!/usr/bin/env python3
"""Audit a lightweight classifier on the matched reduced12 Stage-D s1 view.

The frozen reduced12 ST-GCN representation is read from the existing Stage-D
feature cache.  Only the new two-layer classifier is trained; no candidate,
future observation, policy Test, or perception artifact is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint


SEED = 42
NUM_CLASSES = 12
FEATURE_DIM = 256
S1_CACHE_DIM = 271
MAX_EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
REFERENCE_ACCURACY = 0.454265873015873
REFERENCE_MACRO_F1 = 0.44478185161607287
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
STAGE_D_RELATIVE = POLICY_RELATIVE / "stage_d/features"
CHECKPOINT_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
HEAD_RUNTIME_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "single_view_classifier_adaptation/learned_head_best.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "single_view_classifier_adaptation"
)
OFFLINE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def _load_split(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"empty Stage-D feature file: {path}")
    features: list[np.ndarray] = []
    cached_logp: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        vector = np.asarray(row["s1_feature"], dtype=np.float32)
        if vector.shape != (S1_CACHE_DIM,):
            raise ValueError(f"unexpected s1_feature shape {vector.shape} in {path}")
        label = int(row["label_id"])
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"invalid reduced12 label {label}")
        if "s1_viewpoint_id" not in row:
            raise ValueError("Stage-D row lacks the matched s1 viewpoint identity")
        features.append(vector[:FEATURE_DIM])
        cached_logp.append(vector[FEATURE_DIM : FEATURE_DIM + NUM_CLASSES])
        labels.append(label)
    return {
        "rows": rows,
        "features": np.stack(features).astype(np.float32),
        "cached_logp": np.stack(cached_logp).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "accuracy": recall,
            "recall": recall,
            "f1": f1,
        }
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


class LightweightClassifier(nn.Module):
    """The only trainable component in this audit."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _evaluate_head(
    model: LightweightClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.from_numpy(features[start : start + BATCH_SIZE]).to(
                device, non_blocking=True
            )
            logits_parts.append(model(batch).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0).astype(np.float32)
    return _classification(labels, np.argmax(logits, axis=1)), logits


def _train_head(
    train: Mapping[str, Any],
    val: Mapping[str, Any],
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
) -> tuple[LightweightClassifier, dict[str, Any]]:
    _seed()
    model = LightweightClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_set = TensorDataset(
        torch.from_numpy(train["features"]), torch.from_numpy(train["labels"])
    )
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_features = val["features"]
    val_labels = val["labels"]
    best_f1 = -float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(torch.from_numpy(val_features).to(device)).cpu()
            val_loss = float(nn.functional.cross_entropy(
                val_logits, torch.from_numpy(val_labels)
            ).item())
        val_predictions = np.argmax(val_logits.numpy(), axis=1)
        val_metrics = _classification(val_labels, val_predictions)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"[single-view-head] epoch={epoch:02d}/{MAX_EPOCHS} "
            f"train_loss={record['train_loss']:.6f} "
            f"val_loss={val_loss:.6f} val_acc={record['val_accuracy']:.6f} "
            f"val_f1={record['val_macro_f1']:.6f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1 + 1e-8:
            best_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_dim": FEATURE_DIM,
                    "hidden_dim": 256,
                    "num_classes": NUM_CLASSES,
                    "epoch": epoch,
                    "seed": SEED,
                },
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary: dict[str, Any] = {
        "train_contexts": len(train["labels"]),
        "val_contexts": len(val["labels"]),
        "epochs_max": MAX_EPOCHS,
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_loss": best_val_loss,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "architecture": "Linear(256,256) -> GELU -> Linear(256,12)",
        "checkpoint": str(checkpoint.resolve()),
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model, summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spot_check(
    data_root: Path,
    val: Mapping[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Read one archived s1 skeleton to verify cache/checkpoint alignment."""
    row = val["rows"][0]
    archive = (
        data_root / OFFLINE_RELATIVE / str(row["scene_id"]) /
        str(row["region"]) / f"{row['record_id']}.npz"
    )
    if not archive.exists():
        raise FileNotFoundError(f"matched Stage-D archive is missing: {archive}")
    with np.load(archive, allow_pickle=False) as loaded:
        skeletons = np.asarray(loaded["skeleton"], dtype=np.float32)
        viewpoint_ids = np.asarray(loaded["viewpoint_ids"], dtype=np.int64)
    matches = np.flatnonzero(viewpoint_ids == int(row["s1_viewpoint_id"]))
    if matches.size != 1:
        raise ValueError("s1 viewpoint is not unique in the archived record")
    model, _ = load_checkpoint(checkpoint, NUM_CLASSES, str(device))
    skeleton = torch.from_numpy(skeletons[int(matches[0]) : int(matches[0]) + 1]).to(device)
    with torch.inference_mode():
        feature = model.forward_features(skeleton).cpu().numpy()[0]
        logp = torch.log_softmax(model.fc(torch.from_numpy(feature[None]).to(device)), dim=1)
        logp = logp.cpu().numpy()[0]
    cached_feature = np.asarray(val["features"][0], dtype=np.float32)
    cached_logp = np.asarray(val["cached_logp"][0], dtype=np.float32)
    return {
        "archive": str(archive.resolve()),
        "record_id": str(row["record_id"]),
        "scene_id": str(row["scene_id"]),
        "s1_viewpoint_id": int(row["s1_viewpoint_id"]),
        "feature_max_abs_error": float(np.max(np.abs(feature - cached_feature))),
        "logp_max_abs_error": float(np.max(np.abs(logp - cached_logp))),
        "checkpoint_sha256": _file_sha256(checkpoint),
    }


def _analysis(result: Mapping[str, Any]) -> str:
    frozen = result["metrics"]["Frozen original head"]
    learned = result["metrics"]["Learned single-view head"]
    delta_acc = result["delta_accuracy"]
    delta_f1 = result["delta_macro_f1"]
    learned_acc = float(learned["accuracy"])
    if learned_acc <= 0.49:
        decision = "失败（Accuracy ≤49%）：按预注册淘汰标准停止该方向。"
    elif learned_acc <= 0.52:
        decision = "弱信号（49–52%）：提升不足，不继续复杂化。"
    elif learned_acc < 0.53:
        decision = "超过52%门槛，但提升有限；仅保留为候选，不扩展方法。"
    else:
        decision = "达到有效候选区间（≥53%）；本轮仅记录结果，不自动扩展。"
    lines = [
        "# Reduced12 Single-view Classifier Adaptation Audit",
        "",
        "## Matched protocol",
        "",
        f"Train uses {result['protocol']['train_contexts']} Stage-D contexts and Moving Val uses {result['protocol']['moving_val_contexts']} contexts. Both use the exact cached `s1_viewpoint_id` observation from `stage_d/features/{{train,val}}.jsonl`; no s0, Stay, candidate, or alternate viewpoint is mixed in.",
        "",
        f"The frozen reduced12 ST-GCN checkpoint is `{result['protocol']['stgcn_checkpoint']}`. Its 256-D penultimate feature is the only input to the learned head. The read-only archive spot check reports feature max error {result['spot_check']['feature_max_abs_error']:.3e} and logp max error {result['spot_check']['logp_max_abs_error']:.3e}.",
        "",
        "## Classifier and training",
        "",
        "Train-only head: `Linear(256,256) -> GELU -> Linear(256,12)` with cross-entropy, AdamW, seed 42, and at most 20 epochs. The best checkpoint is selected by Moving-Val Macro-F1 and stored outside Git.",
        f"Selected epoch: {result['training']['selected_epoch']}; best Val loss: {result['training']['best_val_loss']:.6f}; best Val Macro-F1: {result['training']['best_val_macro_f1']:.6f}.",
        "",
        "## Moving Val result",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Frozen original head | {frozen['accuracy']:.6f} | {frozen['macro_f1']:.6f} |",
        f"| Learned single-view head | {learned['accuracy']:.6f} | {learned['macro_f1']:.6f} |",
        "",
        f"Learned minus Frozen: ΔAccuracy={delta_acc * 100:+.3f}pp, ΔMacro-F1={delta_f1 * 100:+.3f}pp.",
        f"Decision: {decision}",
        "",
        "Per-class metrics and the 12×12 confusion matrices are included in `result.json` and `per_class_metrics.json` to check that any gain is not confined to one action.",
        "",
        "## Leakage and boundaries",
        "",
        "Inference input is only the frozen ST-GCN `s1` feature. GT label, future candidate observation/skeleton/RGB/logits, hard predicted action, geometry, policy Test, and new perception data are not used. No active view selection or continuous fusion is evaluated.",
        "",
        "```text",
        "policy_test_used=false",
        "stgcn_encoder_frozen=true",
        "new_data_generated=false",
        "single_view_s1_only=true",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    _seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_root = data_root / STAGE_D_RELATIVE
    train = _load_split(stage_root / "train.jsonl")
    val = _load_split(stage_root / "val.jsonl")
    if len(val["labels"]) != 10080:
        raise ValueError(f"expected 10080 Moving Val contexts, found {len(val['labels'])}")
    if any(str(row.get("policy_split")) != split
           for split, data in (("train", train), ("val", val))
           for row in data["rows"]):
        raise ValueError("Stage-D split marker mismatch")
    checkpoint = data_root / CHECKPOINT_RELATIVE
    if not checkpoint.exists():
        raise FileNotFoundError(f"frozen ST-GCN checkpoint not found: {checkpoint}")
    head_checkpoint = data_root / HEAD_RUNTIME_RELATIVE
    model, training = _train_head(
        train, val, device, head_checkpoint, output_dir / "training_summary.json"
    )
    learned_metrics, _ = _evaluate_head(model, val["features"], val["labels"], device)
    cached_predictions = np.argmax(val["cached_logp"], axis=1)
    frozen_metrics = _classification(val["labels"], cached_predictions)
    if (abs(frozen_metrics["accuracy"] - REFERENCE_ACCURACY) > 1e-9 or
            abs(frozen_metrics["macro_f1"] - REFERENCE_MACRO_F1) > 1e-9):
        raise RuntimeError(
            "FrozenStageCv0 matched-baseline gate failed before learned-head "
            f"evaluation: {frozen_metrics['accuracy']:.12f}/"
            f"{frozen_metrics['macro_f1']:.12f}"
        )
    spot_check = _spot_check(data_root, val, checkpoint, device)
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_SINGLE_VIEW_CLASSIFIER_ADAPTATION",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.time() - started,
        },
        "protocol": {
            "train_contexts": int(len(train["labels"])),
            "moving_val_contexts": int(len(val["labels"])),
            "train_file": str((stage_root / "train.jsonl").resolve()),
            "moving_val_file": str((stage_root / "val.jsonl").resolve()),
            "viewpoint": "Stage-D s1_viewpoint_id (matched FrozenStageCv0)",
            "feature_source": "s1_feature[:256] cached frozen ST-GCN penultimate feature",
            "original_classifier_source": "s1_feature[256:268] cached log_softmax",
            "stgcn_checkpoint": str(checkpoint.resolve()),
            "stgcn_checkpoint_sha256": _file_sha256(checkpoint),
            "normalization": "inherited from archived Stage-D frozen ST-GCN features",
            "policy_test_used": False,
        },
        "feature_dim": FEATURE_DIM,
        "classifier": "Linear(256,256) -> GELU -> Linear(256,12)",
        "training": training,
        "spot_check": spot_check,
        "metrics": {
            "Frozen original head": frozen_metrics,
            "Learned single-view head": learned_metrics,
        },
        "delta_accuracy": float(learned_metrics["accuracy"] - frozen_metrics["accuracy"]),
        "delta_macro_f1": float(learned_metrics["macro_f1"] - frozen_metrics["macro_f1"]),
        "leakage_audit": {
            "inference_uses_only_s1_frozen_feature": True,
            "gt_label_used_at_inference": False,
            "future_candidate_observation_used_at_inference": False,
            "future_candidate_skeleton_used_at_inference": False,
            "future_candidate_feature_or_logp_used_at_inference": False,
            "hard_predicted_action_used_at_inference": False,
            "policy_test_used": False,
            "new_rgb_or_skeleton_generated": False,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps({
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "num_classes": NUM_CLASSES,
            "classifier": result["classifier"],
            "viewpoint": result["protocol"]["viewpoint"],
            "train_file": result["protocol"]["train_file"],
            "moving_val_file": result["protocol"]["moving_val_file"],
            "policy_test_used": False,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps({name: value["per_class"] for name, value in result["metrics"].items()},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = _require_cuda(args.device)
    result = run(args.output_dir.resolve(), args.data_root.resolve(), device)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "moving_val_contexts": result["protocol"]["moving_val_contexts"],
        "frozen": result["metrics"]["Frozen original head"],
        "learned": result["metrics"]["Learned single-view head"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
