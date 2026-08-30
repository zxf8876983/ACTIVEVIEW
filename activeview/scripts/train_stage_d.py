#!/usr/bin/env python3
"""Train EXP014's second-step ranker on Stage D Train and select on Val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_losses import stage_c_loss
from activeview.active_view.stage_d_dataset import StageDDataset, StageDRecordBalancedSampler, collate_stage_d, load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, predict_second_step_dataset, summarize_trajectory_rows
from activeview.active_view.stage_d_policy import SequentialObservationRanker, schema_metadata
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def train(
    *, cache_root: Path, stage_b_root: Path, output_dir: Path, checkpoint_source: Path,
    label_mapping: Path, device_name: str, batch_size: int, episodes_per_record: int,
    max_epochs: int, patience: int, lr: float, weight_decay: float, seed: int,
) -> dict[str, Any]:
    _seed(seed)
    summary_path = cache_root / "stage_d_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    train_set = StageDDataset(Path(summary["feature_files"]["train"]), stats)
    val_set = StageDDataset(Path(summary["feature_files"]["val"]), stats)
    sampler = StageDRecordBalancedSampler(train_set.rows, episodes_per_record=episodes_per_record, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=collate_stage_d, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_stage_d, num_workers=0)
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(Path(summary["source_stage_c_v0_predictions"]["val"]))
    categories = _categories(label_mapping)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = SequentialObservationRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = output_dir / "sequential_observation_ranker_best.pth"
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = {"total": 0.0, "regression": 0.0, "ranking": 0.0}
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(
                batch["s0_feature"].to(device), batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device), batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            )
            losses = stage_c_loss(
                predicted, batch["utility_targets"].to(device), batch["candidate_mask"].to(device),
                lambda_reg=1.0, lambda_rank=1.0, tau=0.5, lambda_gap=0.0,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            steps += 1
        second_predictions = predict_second_step_dataset(model, val_loader, device)
        cache_rows = val_set.rows
        trajectories = build_stage_d_trajectories(stage_b_val, v0_val, cache_rows, second_predictions)
        val_metrics = summarize_trajectory_rows(trajectories, categories)
        score = float(val_metrics["recognition"]["macro_f1"])
        history.append({"epoch": epoch, "train_loss": totals["total"] / max(steps, 1), "train_regression_loss": totals["regression"] / max(steps, 1), "train_ranking_loss": totals["ranking"] / max(steps, 1), "val_metrics": val_metrics, "lr": optimizer.param_groups[0]["lr"]})
        if score > best_metric + 1e-12:
            best_metric = score
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "model_type": SequentialObservationRanker.model_type, "epoch": epoch,
                "cache_summary_sha256": file_sha256(summary_path), "cache_file_sha256": summary["feature_file_sha256"],
                "cache_stats_sha256": summary["feature_stats_sha256"], "checkpoint_source_sha256": file_sha256(checkpoint_source),
                "config": {"batch_size": batch_size, "episodes_per_record": episodes_per_record, "max_epochs": max_epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay, "seed": seed, "lambda_reg": 1.0, "lambda_rank": 1.0, "tau": 0.5, "lambda_gap": 0.0},
            }, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    result = {
        "protocol": "ACTIVEVIEW Stage D EXP014 Train-to-Val sequential policy training",
        "model_type": SequentialObservationRanker.model_type, "model_schema": schema_metadata(),
        "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_selection_metric": "Val final trajectory Macro-F1", "cache_root": str(cache_root.resolve()),
        "cache_summary_sha256": file_sha256(summary_path), "cache_file_sha256": summary["feature_file_sha256"], "cache_stats_sha256": summary["feature_stats_sha256"],
        "stgcn_checkpoint": str(checkpoint_source.resolve()), "stgcn_checkpoint_sha256": file_sha256(checkpoint_source),
        "eligible_train_records": len(sampler.groups), "eligible_train_episodes": len(train_set), "eligible_val_episodes": len(val_set),
        "sampler": {"mode": "record_balanced", "episodes_per_record": episodes_per_record},
        "loss": {"regression": "SmoothL1", "ranking": "stay-inclusive soft-target cross-entropy", "lambda_reg": 1.0, "lambda_rank": 1.0, "tau": 0.5, "lambda_gap": 0.0},
        "test_used": False, "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint-source", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--label-mapping", type=Path, required=True)
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
    if args.batch_size <= 0 or args.episodes_per_record <= 0 or args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("batch, sampler, epochs and patience must be positive")
    print(json.dumps(train(cache_root=args.cache_root, stage_b_root=args.stage_b_root, output_dir=args.output_dir, checkpoint_source=args.checkpoint_source, label_mapping=args.label_mapping, device_name=args.device, batch_size=args.batch_size, episodes_per_record=args.episodes_per_record, max_epochs=args.max_epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
