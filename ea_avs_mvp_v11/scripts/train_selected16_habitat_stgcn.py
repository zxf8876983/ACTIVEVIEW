#!/usr/bin/env python3
"""Train an estimated-skeleton ST-GCN on selected16 pure-color Habitat data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> Dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    f1_values: List[float] = []
    recalls: List[float] = []
    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    targets: List[int] = []
    predictions: List[int] = []
    with torch.no_grad():
        for batch, labels in loader:
            logits = model(batch.to(device))
            loss_sum += float(criterion(logits, labels.to(device)).item()) * len(labels)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.tolist())
    result = metrics(targets, predictions, num_classes)
    result["loss"] = loss_sum / max(len(targets), 1)
    return result


def train(
    *, data_root: Path, checkpoint: Path, max_epochs: int, patience: int,
    min_delta: float, batch_size: int, learning_rate: float, weight_decay: float,
    seed: int, device_name: str, oversample_power: float,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    mapping = json.loads((data_root / "label_mapping.json").read_text(encoding="utf-8"))
    categories = [label for label, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    arrays = {split: np.load(data_root / f"{split}_data.npy") for split in ("train", "val")}
    labels = {split: np.load(data_root / f"{split}_labels.npy") for split in ("train", "val")}
    shape = arrays["train"].shape
    if len(shape) != 5 or shape[1] != 3 or shape[3:] != (17, 1):
        raise ValueError(f"Unexpected tensor shape: {shape}")
    for split in arrays:
        if arrays[split].shape[1:] != shape[1:] or len(arrays[split]) != len(labels[split]):
            raise ValueError(f"Inconsistent {split} tensor/label shape")
        if not np.isfinite(arrays[split]).all():
            raise ValueError(f"{split} tensor contains NaN or Inf")
    train_ds = TensorDataset(torch.from_numpy(arrays["train"]).float(), torch.from_numpy(labels["train"]).long())
    val_ds = TensorDataset(torch.from_numpy(arrays["val"]).float(), torch.from_numpy(labels["val"]).long())
    model = STGCN(in_channels=3, num_classes=len(categories), graph_strategy="spatial", edge_importance_weighting=True, skel_def=get_skeleton_definition(backend="h36m_17")).to(device)
    counts = np.bincount(labels["train"], minlength=len(categories)).astype(np.float32)
    class_weights = np.sqrt(len(labels["train"]) / np.maximum(counts, 1.0))
    class_weights /= np.mean(class_weights)
    if oversample_power < 0.0:
        raise ValueError("oversample_power must be non-negative")
    sampling_weights = np.power(1.0 / np.maximum(counts, 1.0), oversample_power).astype(np.float64)
    sampling_weights /= np.mean(sampling_weights)
    sample_weights = torch.from_numpy(sampling_weights[labels["train"]]).double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, patience // 4), min_lr=1e-5)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_epoch = max_epochs
    LOGGER.info("Training selected16 estimated skeletons: train=%d val=%d device=%s", len(train_ds), len(val_ds), device)
    LOGGER.info("Categories=%s; counts=%s", categories, counts.astype(int).tolist())
    LOGGER.info("Oversampling enabled: sampling_power=%.3f, sampling_weights=%s", oversample_power, sampling_weights.tolist())
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch, batch_labels in train_loader:
            batch, batch_labels = batch.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_labels)
            correct += int((logits.argmax(dim=-1) == batch_labels).sum().item())
            seen += len(batch_labels)
        val_metrics = _evaluate(model, val_loader, criterion, device, len(categories))
        scheduler.step(val_metrics["macro_f1"])
        epoch_metrics = {
            "epoch": float(epoch), "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1), "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)
        score = val_metrics["macro_f1"]
        if score > best_score + min_delta:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save({
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "categories": categories, "label_mapping": mapping, "num_classes": len(categories),
                "skeleton_backend": "h36m_17", "input_shape": list(shape[1:]), "frozen": True,
                "best_epoch": best_epoch, "best_val_macro_f1": best_score,
                "train_samples": len(train_ds), "val_samples": len(val_ds),
                "train_class_counts": counts.astype(int).tolist(), "class_weights": class_weights.tolist(),
                "oversampling": {
                    "enabled": True,
                    "power": oversample_power,
                    "replacement": True,
                    "samples_per_epoch": len(train_ds),
                    "class_sampling_weights": sampling_weights.tolist(),
                },
                "seed": seed, "train_data_sha256": file_sha256(data_root / "train_data.npy"),
                "val_data_sha256": file_sha256(data_root / "val_data.npy"),
                "skeleton_preprocessing": "RGB -> Ultralytics/VideoPose3D -> camera_to_gravity + root_center + torso_scale + yaw_only",
                "protocol": "selected16 pure-color Habitat Train/Val; validation selects frozen ST-GCN; Habitat active-view evaluation is separate",
            }, checkpoint)
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            LOGGER.info("Epoch %d/%d train_loss=%.4f train_acc=%.4f val_acc=%.4f val_macro_f1=%.4f stale=%d/%d", epoch, max_epochs, epoch_metrics["train_loss"], epoch_metrics["train_accuracy"], epoch_metrics["val_accuracy"], epoch_metrics["val_macro_f1"], stale_epochs, patience)
        if stale_epochs >= patience:
            stopped_epoch = epoch
            LOGGER.info("Early stopping at epoch %d; best epoch=%d val_macro_f1=%.4f", epoch, best_epoch, best_score)
            break
    result = {"checkpoint": str(checkpoint), "categories": categories, "train_samples": len(train_ds), "val_samples": len(val_ds), "best_epoch": best_epoch, "stopped_epoch": stopped_epoch, "best_val_macro_f1": best_score, "history": history}
    (checkpoint.parent / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    runtime_root = get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=runtime_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed")
    parser.add_argument("--checkpoint", type=Path, default=runtime_root / "checkpoints" / "stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled" / "stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oversample-power", type=float, default=0.5, help="Sampling weight is count^(-power); 0.5 is tempered inverse-frequency oversampling")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(data_root=args.data_root, checkpoint=args.checkpoint, max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed, device_name=args.device, oversample_power=args.oversample_power)


if __name__ == "__main__":
    main()
