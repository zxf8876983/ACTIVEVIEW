#!/usr/bin/env python3
"""Train the diagnostic-only Stage C-v3 future-perception teacher on Train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_dataset import RecordBalancedSampler
from activeview.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup
from activeview.active_view.stage_c_losses import stage_c_loss
from activeview.active_view.stage_c_v3_teacher import (
    FutureTeacherDataset,
    collate_future_teacher,
    load_teacher_stats,
    predict_teacher_dataset,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import FuturePerceptionTeacherMLP, count_parameters


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def train(
    *, cache_root: Path, output_dir: Path, device_name: str, batch_size: int,
    episodes_per_record: int, max_epochs: int, patience: int, lr: float,
    weight_decay: float, seed: int, stage_b_root: Path | None = None,
) -> Dict[str, Any]:
    _seed(seed)
    summary_path = cache_root / "stage_c_v3_teacher_summary.json"
    summary = _load(summary_path)
    stats = load_teacher_stats(cache_root / "teacher_feature_stats.json")
    train_set = FutureTeacherDataset(Path(summary["feature_files"]["train"]), stats)
    val_set = FutureTeacherDataset(Path(summary["feature_files"]["val"]), stats)
    sampler = RecordBalancedSampler(train_set.rows, episodes_per_record=episodes_per_record, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=collate_future_teacher, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_future_teacher, num_workers=0)
    if stage_b_root is None:
        stage_b_root = cache_root.parent.parent / "stage_b"
    stage_b_val = load_stage_b_lookup(stage_b_root / "utility_labels/val.jsonl")
    mapping = json.loads(Path(summary["label_mapping"]).read_text(encoding="utf-8"))
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = FuturePerceptionTeacherMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[Dict[str, float | int]] = []
    checkpoint = output_dir / "future_perception_teacher_best.pth"
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(
                batch["current_feature"].to(device),
                batch["candidate_geometry"].to(device),
                batch["future_candidate_perception"].to(device),
                batch["candidate_mask"].to(device),
            )
            losses = stage_c_loss(
                predicted,
                batch["utility_targets"].to(device),
                batch["candidate_mask"].to(device),
                lambda_reg=1.0,
                lambda_rank=1.0,
                tau=0.5,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(losses["total"].detach().cpu())
            steps += 1
        val_rows = predict_teacher_dataset(model, val_loader, stage_b_val, device)
        val_metrics = evaluate_predictions(val_rows, categories)
        val_score = float(val_metrics["recognition"]["StageC"]["macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": total / max(steps, 1),
            "val_macro_f1": val_score,
            "val_mean_regret": float(val_metrics["decision_regret"]["mean"]),
            "val_p90_regret": float(val_metrics["decision_regret"]["p90"]),
        })
        if val_score > best_metric + 1e-12:
            best_metric = val_score
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "future_perception_teacher",
                "epoch": epoch,
                "teacher_summary_sha256": file_sha256(summary_path),
                "config": {
                    "batch_size": batch_size, "episodes_per_record": episodes_per_record,
                    "max_epochs": max_epochs, "patience": patience, "lr": lr,
                    "weight_decay": weight_decay, "seed": seed,
                },
            }, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    result = {
        "model_type": "future_perception_teacher",
        "diagnostic_only": True,
        "deployable_policy": False,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "teacher_summary_sha256": file_sha256(summary_path),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("parameter_count", "selected_epoch", "best_val_macro_f1")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes-per-record", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        cache_root=args.cache_root, output_dir=args.output_dir, device_name=args.device,
        batch_size=args.batch_size, episodes_per_record=args.episodes_per_record,
        max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        stage_b_root=args.stage_b_root,
    )


if __name__ == "__main__":
    main()
