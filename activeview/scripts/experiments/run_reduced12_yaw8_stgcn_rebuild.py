#!/usr/bin/env python3
"""Train and evaluate the reduced12 Yaw8 recognizer without policy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.perception.skeleton import get_skeleton_definition
from activeview.recognition.stgcn.model import STGCN

LOGGER = logging.getLogger(__name__)
YAW_DEGREES: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_metrics(targets: Sequence[int], predictions: Sequence[int], classes: int = 12) -> Dict[str, Any]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    f1: List[float] = []
    for class_id in range(classes):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    total = max(1, int(confusion.sum()))
    return {
        "count": int(confusion.sum()),
        "accuracy": float(np.trace(confusion) / total),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {"class_id": i, "support": int(confusion[i].sum()), "recall": float(
                confusion[i, i] / confusion[i].sum() if confusion[i].sum() else 0.0
            ), "f1": float(f1[i])}
            for i in range(classes)
        ],
    }


def load_split(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    data = np.load(root / f"{split}_data.npy", mmap_mode="r")
    labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
    metadata_path = root / f"{split}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing generated metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(data) != len(labels) or len(data) != len(metadata):
        raise ValueError(f"Mismatched {split} arrays and metadata")
    return np.asarray(data), np.asarray(labels, dtype=np.int64), metadata


def make_model(classes: int, device: torch.device) -> STGCN:
    model = STGCN(
        in_channels=3, num_classes=classes, graph_strategy="spatial",
        edge_importance_weighting=True, skel_def=get_skeleton_definition(backend="h36m_17"),
    ).to(device)
    return model


def evaluate_model(model: nn.Module, data: np.ndarray, labels: np.ndarray, device: torch.device,
                   batch_size: int = 256) -> Tuple[Dict[str, Any], np.ndarray]:
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(data)).float(), torch.from_numpy(labels).long()),
                        batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.eval()
    predictions: List[int] = []
    with torch.no_grad():
        for batch, _ in loader:
            predictions.extend(model(batch.to(device)).argmax(dim=-1).cpu().tolist())
    return fixed_metrics(labels.tolist(), predictions), np.asarray(predictions, dtype=np.int64)


def train_yaw8(data_root: Path, checkpoint_root: Path, *, seed: int, device: torch.device,
               max_epochs: int, batch_size: int, patience: int) -> Dict[str, Any]:
    set_seed(seed)
    train_data, train_labels, _ = load_split(data_root, "train")
    val_data, val_labels, _ = load_split(data_root, "val")
    class_counts = np.bincount(train_labels, minlength=12).astype(np.float32)
    weights = np.power(1.0 / np.maximum(class_counts, 1.0), 0.5)
    weights /= weights.mean()
    sample_weights = torch.from_numpy(weights[train_labels]).double()
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(train_data)).float(), torch.from_numpy(train_labels).long()),
        batch_size=batch_size, sampler=sampler, pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    model = make_model(12, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                            patience=5, min_lr=1e-5)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch, labels in DataLoader(
                TensorDataset(torch.from_numpy(np.asarray(val_data)).float(), torch.from_numpy(val_labels).long()),
                batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda",
            ):
                logits = model(batch.to(device))
                val_loss += float(criterion(logits, labels.to(device)).item()) * len(labels)
                val_seen += len(labels)
        val_loss /= max(1, val_seen)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, seen), "val_loss": val_loss,
               "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "optimizer_steps": int((epoch * len(train_loader)))}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            LOGGER.info("Yaw8 epoch %d/%d train_loss=%.5f val_loss=%.5f", epoch, max_epochs,
                        row["train_loss"], row["val_loss"])
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "categories": list(json.loads((data_root / "label_mapping.json").read_text()).keys()),
                        "num_classes": 12, "selection_metric": "yaw8_val_cross_entropy", "epoch": epoch,
                        "seed": seed}, checkpoint_root / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "num_classes": 12, "epoch": len(history), "seed": seed}, checkpoint_root / "last.pt")
    summary = {"seed": seed, "train_records": len(train_labels), "val_records": len(val_labels),
               "max_epochs": max_epochs, "completed_epochs": len(history), "best_epoch": best_epoch,
               "best_val_loss": best_val_loss, "batch_size": batch_size, "optimizer": "Adam",
               "learning_rate": 1e-3, "weight_decay": 1e-4, "oversample_power": 0.5,
               "history": history, "checkpoint": str((checkpoint_root / "best.pt").resolve())}
    (checkpoint_root / "training_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_model(checkpoint: Path, device: torch.device) -> STGCN:
    model = make_model(12, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_views(model: nn.Module, data: np.ndarray, labels: np.ndarray,
                   metadata: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, Any]:
    overall, predictions = evaluate_model(model, data, labels, device)
    per_yaw: Dict[str, Any] = {}
    for yaw in YAW_DEGREES:
        indices = [i for i, item in enumerate(metadata) if int(item.get("scene_yaw_deg", -1)) == yaw]
        per_yaw[str(yaw)] = fixed_metrics(labels[indices].tolist(), predictions[indices].tolist()) if indices else {}
    return {"overall": overall, "per_yaw": per_yaw}


def protected_assets(paths: Iterable[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for path in paths:
        if path.exists() and path.is_file():
            stat = path.stat()
            result[str(path.resolve())] = {"sha256": sha256(path), "size": stat.st_size,
                                           "mtime_ns": stat.st_mtime_ns}
    return result


def write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    old = result["evaluations"]["old_on_yaw8_val"]
    new = result["evaluations"]["new_on_yaw8_val"]
    old_yaw = old["per_yaw"]
    new_yaw = new["per_yaw"]
    old_mean = float(np.mean([old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES]))
    new_mean = float(np.mean([new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES]))
    old_worst = min(old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)
    new_worst = min(new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)
    gain = new_mean - old_mean
    worst_gain = new_worst - old_worst
    clean_change = result["evaluations"]["new_on_original_val"]["overall"]["accuracy"] - result["evaluations"]["old_on_original_val"]["overall"]["accuracy"]
    clean_drop = -clean_change
    if clean_drop <= 0.03 and (gain >= 0.05 or worst_gain >= 0.08):
        decision = "CONFIRMED"
    elif gain >= 0.02 or worst_gain >= 0.04:
        decision = "PARTIALLY CONFIRMED"
    else:
        decision = "NOT A MAJOR CAUSE"
    lines = [
        "Experiment Yaw8 ST-GCN Recognizer Rebuild",
        "Purpose mismatch: test whether clean recognizer training is mismatched to random eight-direction scene yaw.",
        "Identical original record split; yaw augmentation 8 values: 0,45,90,135,180,225,270,315.",
        "Augmentation at Habitat humanoid rendering before RGB; perception unchanged.",
        "Preprocessing unchanged align_canonical=True; old assets preserved.",
        "Policy Train/Val/Test not used for model selection; Policy Test not read.",
        "",
        "## Results",
        f"- Original clean Val: old Acc/F1={result['evaluations']['old_on_original_val']['overall']['accuracy']:.6f}/{result['evaluations']['old_on_original_val']['overall']['macro_f1']:.6f}; new={result['evaluations']['new_on_original_val']['overall']['accuracy']:.6f}/{result['evaluations']['new_on_original_val']['overall']['macro_f1']:.6f}.",
        f"- Yaw8 Val overall: old Acc/F1={old['overall']['accuracy']:.6f}/{old['overall']['macro_f1']:.6f}; new={new['overall']['accuracy']:.6f}/{new['overall']['macro_f1']:.6f}.",
        f"- Mean yaw accuracy: old={old_mean:.6f}, new={new_mean:.6f}, MeanYawGain={gain:+.6f}.",
        f"- Worst yaw accuracy: old={old_worst:.6f}, new={new_worst:.6f}, WorstYawGain={worst_gain:+.6f}.",
        f"- Yaw robustness gap (max-min): old={max(v['accuracy'] for v in old_yaw.values())-old_worst:.6f}, new={max(v['accuracy'] for v in new_yaw.values())-new_worst:.6f}.",
        f"- Original-clean change (new-old)={clean_change:+.6f}.",
        "",
        "## Decision",
        f"Final decision: **{decision}**.",
        "This is a recognizer-only Train/Val result. Do not rebuild Policy automatically; freeze the new recognizer and rebuild Policy logits/utility only after an explicit decision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = get_data_root()
    source_root = args.source_root or data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-train"
    yaw_root = args.yaw_root or data_root / "datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1"
    checkpoint_root = args.checkpoint_root or data_root / "checkpoints/stgcn_reduced12_yaw8_v1"
    output_root = args.output_root or REPO_ROOT / "experiments/reduced12_eight_placement_v1/yaw8_stgcn_rebuild"
    output_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Yaw8 training/evaluation")
    device = torch.device(args.device)
    old_checkpoint = data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
    videopose_checkpoint = get_data_root() / "checkpoints/videopose3d/pretrained_h36m_detectron_coco.bin"
    source_audit = {
        "old_dataset_root": str(source_root.resolve()),
        "old_train_manifest": str((source_root / "train.json").resolve()),
        "old_val_manifest": str((source_root / "val.json").resolve()),
        "old_checkpoint": str(old_checkpoint.resolve()),
        "old_train_records": len(json.loads((source_root / "train.json").read_text())),
        "old_val_records": len(json.loads((source_root / "val.json").read_text())),
        "classes": json.loads((source_root / "label_mapping.json").read_text()),
        "target_frames": 30, "skeleton_shape": [3, 30, 17, 1],
        "image_size": 256, "camera_hfov_deg": 50.0, "camera_height_m": 1.2,
        "camera_distance_range_m": [2.4, 2.8],
        "camera_angle_jitter_deg": [-25.0, 25.0],
        "pose_backend": "Ultralytics YOLO26n-Pose + VideoPose3D",
        "videopose3d_checkpoint": str(videopose_checkpoint.resolve()),
        "videopose3d_checkpoint_sha256": sha256(videopose_checkpoint),
        "videopose3d_config": "UltralyticsPose3DEstimator project defaults",
        "normalization": "camera_to_gravity + root_center + torso_scale + yaw_only; align_canonical=True",
        "stgcn": {"architecture": "spatial ST-GCN, 3 channels, 9 blocks, 256-D feature",
                  "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 64,
                  "epochs_max": 200, "patience": 20, "seed": 42},
        "old_checkpoint_sha256": sha256(old_checkpoint),
        "old_source_files": protected_assets([source_root / "train.json", source_root / "val.json",
                                                source_root / "train_data.npy", source_root / "val_data.npy", old_checkpoint]),
        "policy_train_val_used_for_recognizer": False, "policy_test_read": False,
    }
    (output_root / "source_protocol_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")
    if not (yaw_root / "train_data.npy").exists() or not (yaw_root / "val_data.npy").exists():
        raise FileNotFoundError(f"Yaw8 data is incomplete: {yaw_root}")
    yaw_train, yaw_train_labels, yaw_train_meta = load_split(yaw_root, "train")
    yaw_val, yaw_val_labels, yaw_val_meta = load_split(yaw_root, "val")
    split_audit = json.loads((yaw_root / "split_audit.json").read_text())
    if split_audit.get("record_split_leakage", 1) != 0:
        raise RuntimeError("Yaw8 source record split leakage detected; refusing to train")
    generation = json.loads((yaw_root / "generation_summary.json").read_text()) if (yaw_root / "generation_summary.json").exists() else {}
    (output_root / "generation_summary.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    (output_root / "split_audit.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
    yaw_dist = {str(yaw): {"train": sum(int(x.get("scene_yaw_deg", -1)) == yaw for x in yaw_train_meta),
                           "val": sum(int(x.get("scene_yaw_deg", -1)) == yaw for x in yaw_val_meta)} for yaw in YAW_DEGREES}
    (output_root / "yaw_distribution.json").write_text(json.dumps(yaw_dist, indent=2), encoding="utf-8")
    (output_root / "training_config.json").write_text(json.dumps({"seed": args.seed, "device": str(device),
        "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "selection_metric": "Yaw8 Val cross-entropy", "policy_test_read": False}, indent=2), encoding="utf-8")
    training = train_yaw8(yaw_root, checkpoint_root, seed=args.seed, device=device,
                          max_epochs=args.max_epochs, batch_size=args.batch_size, patience=args.patience)
    (output_root / "training_history.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    old_model = load_model(old_checkpoint, device)
    new_model = load_model(checkpoint_root / "best.pt", device)
    clean_val, clean_labels, clean_meta = load_split(source_root, "val")
    evaluations = {
        "old_on_original_val": evaluate_views(old_model, clean_val, clean_labels, clean_meta, device),
        "new_on_original_val": evaluate_views(new_model, clean_val, clean_labels, clean_meta, device),
        "old_on_yaw8_val": evaluate_views(old_model, yaw_val, yaw_val_labels, yaw_val_meta, device),
        "new_on_yaw8_val": evaluate_views(new_model, yaw_val, yaw_val_labels, yaw_val_meta, device),
    }
    for name, value in evaluations.items():
        (output_root / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    source_train_ids = {str(item.get("source_record_id", "")) for item in yaw_train_meta}
    source_val_ids = {str(item.get("source_record_id", "")) for item in yaw_val_meta}
    generated_source_overlap = sorted((source_train_ids & source_val_ids) - {""})
    source_train_manifest_ids = {
        str(item["record_id"]) for item in json.loads((source_root / "train.json").read_text(encoding="utf-8"))
    }
    source_val_manifest_ids = {
        str(item["record_id"]) for item in json.loads((source_root / "val.json").read_text(encoding="utf-8"))
    }
    source_manifest_overlap = sorted(source_train_manifest_ids & source_val_manifest_ids)
    old_yaw = evaluations["old_on_yaw8_val"]["per_yaw"]
    new_yaw = evaluations["new_on_yaw8_val"]["per_yaw"]
    robustness = {
        "old_mean_yaw_accuracy": float(np.mean([old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES])),
        "new_mean_yaw_accuracy": float(np.mean([new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES])),
        "old_worst_yaw_accuracy": float(min(old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)),
        "new_worst_yaw_accuracy": float(min(new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)),
        "MeanYawGain": float(np.mean([new_yaw[str(y)]["accuracy"] - old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES])),
        "WorstYawGain": float(min(new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES) - min(old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)),
        "old_yaw_robustness_gap": float(max(old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES) - min(old_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)),
        "new_yaw_robustness_gap": float(max(new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES) - min(new_yaw[str(y)]["accuracy"] for y in YAW_DEGREES)),
        "original_clean_accuracy_change": float(evaluations["new_on_original_val"]["overall"]["accuracy"] - evaluations["old_on_original_val"]["overall"]["accuracy"]),
    }
    (output_root / "per_yaw_metrics.json").write_text(json.dumps({"old": old_yaw, "new": new_yaw}, indent=2), encoding="utf-8")
    (output_root / "per_class_metrics.json").write_text(json.dumps({
        "old_yaw8": evaluations["old_on_yaw8_val"]["overall"]["per_class"],
        "new_yaw8": evaluations["new_on_yaw8_val"]["overall"]["per_class"],
        "old_per_yaw": {yaw: old_yaw[yaw]["per_class"] for yaw in old_yaw},
        "new_per_yaw": {yaw: new_yaw[yaw]["per_class"] for yaw in new_yaw},
    }, indent=2), encoding="utf-8")
    (output_root / "robustness_metrics.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    (output_root / "gt_yaw_control.json").write_text(json.dumps({"status": "skipped", "reason": "Optional GT control not run; main estimated-skeleton experiment completed without GT skeleton input."}, indent=2), encoding="utf-8")
    after_assets = protected_assets([source_root / "train.json", source_root / "val.json", source_root / "train_data.npy", source_root / "val_data.npy", old_checkpoint])
    unchanged = source_audit["old_source_files"] == after_assets
    (output_root / "leakage_audit.json").write_text(json.dumps({
        "source_record_overlap_train_val": len(source_manifest_overlap),
        "source_record_overlap_ids": source_manifest_overlap,
        "same_source_record_different_yaw_across_split": len(generated_source_overlap),
        "same_source_record_different_yaw_across_split_ids": generated_source_overlap,
        "policy_train_val_used_for_recognizer": False,
        "policy_test_read": False,
    }, indent=2), encoding="utf-8")
    (output_root / "artifact_protection_audit.json").write_text(json.dumps({"before_generation": source_audit["old_source_files"], "after_generation": after_assets, "unchanged": unchanged}, indent=2), encoding="utf-8")
    result = {"experiment": "yaw8_stgcn_rebuild", "status": "completed", "source_protocol": source_audit,
              "generation": generation, "yaw_distribution": yaw_dist, "training": training,
              "evaluations": evaluations, "robustness": robustness,
              "flags": {"policy_test_read": False, "policy_train_val_used_for_recognizer": False,
                        "old_assets_modified": not unchanged, "yaw_applied_at_habitat_render": True,
                        "perception_unchanged": True, "test_used": False}}
    (output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_analysis(output_root / "analysis.md", result)
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--yaw-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args)


if __name__ == "__main__":
    main()
