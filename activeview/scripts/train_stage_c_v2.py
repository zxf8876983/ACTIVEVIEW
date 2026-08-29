#!/usr/bin/env python3
"""Train one Stage C-v2 policy on Train and select by Val Macro-F1."""

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

from activeview.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup
from activeview.active_view.stage_c_losses import stage_c_loss
from activeview.active_view.stage_c_v2_dataset import StageCV2Dataset, V2RecordBalancedSampler, collate_stage_c_v2, load_v2_statistics
from activeview.active_view.stage_c_v2_evaluation import forward_policy, predict_dataset_v2
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import build_utility_predictor, count_parameters
from activeview.core.paths import get_data_root


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _categories(mapping_path: Path) -> list[str]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def train_model(*, feature_root: Path, stage_b_root: Path, output_dir: Path, model_type: str, device_name: str, batch_size: int, episodes_per_record: int, max_epochs: int, patience: int, lr: float, weight_decay: float, lambda_reg: float, lambda_rank: float, tau: float, seed: int) -> Dict[str, Any]:
    _seed(seed)
    feature_summary_path = feature_root / "stage_c_v2_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    stats = load_v2_statistics(feature_root / "stage_c_v2_feature_stats.json")
    geometry_dim = int(feature_summary["schema"]["candidate_geometry_dim"])
    train_set = StageCV2Dataset(feature_root, "train", stats=stats)
    val_set = StageCV2Dataset(feature_root, "val", stats=stats)
    sampler = V2RecordBalancedSampler(train_set.rows, episodes_per_record=episodes_per_record, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=collate_stage_c_v2, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_stage_c_v2, num_workers=0)
    stage_b_val = load_stage_b_lookup(stage_b_root / "utility_labels/val.jsonl")
    categories = _categories(Path(feature_summary["label_mapping"]))
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_utility_predictor(model_type, geometry_dim=geometry_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[Dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = {"total": 0.0, "regression": 0.0, "ranking": 0.0}
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = forward_policy(model, model_type, batch, device)
            mask = batch["candidate_mask"].to(device)
            losses = stage_c_loss(
                predicted, batch["utility_targets"].to(device), mask,
                lambda_reg=lambda_reg, lambda_rank=lambda_rank, tau=tau,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            steps += 1
        val_rows = predict_dataset_v2(model, model_type, val_loader, stage_b_val, device)
        val_metrics = evaluate_predictions(val_rows, categories)
        val_score = float(val_metrics["recognition"]["StageC"]["macro_f1"])
        scheduler.step(val_score)
        history.append({"epoch": epoch, "train_loss": totals["total"] / max(steps, 1), "train_regression_loss": totals["regression"] / max(steps, 1), "train_ranking_loss": totals["ranking"] / max(steps, 1), "val_metrics": val_metrics, "lr": optimizer.param_groups[0]["lr"]})
        if val_score > best_metric + 1e-12:
            best_metric = val_score
            best_epoch = epoch
            stale = 0
            checkpoint = output_dir / f"{model_type}_best.pth"
            torch.save({
                "model_state_dict": model.state_dict(), "model_type": model_type, "epoch": epoch,
                "feature_summary_sha256": file_sha256(feature_summary_path),
                "feature_file_sha256": feature_summary["feature_file_sha256"],
                "feature_stats_sha256": feature_summary["feature_stats_sha256"],
                "config": {"batch_size": batch_size, "episodes_per_record": episodes_per_record, "max_epochs": max_epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay, "lambda_reg": lambda_reg, "lambda_rank": lambda_rank, "tau": tau, "seed": seed, "geometry_dim": geometry_dim},
            }, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    checkpoint = output_dir / f"{model_type}_best.pth"
    summary = {
        "stage": "C-v2", "model_type": model_type, "parameter_count": count_parameters(model),
        "device": str(device), "selected_epoch": best_epoch, "max_epochs": max_epochs,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_selection_metric": "Val StageC recognition Macro-F1", "feature_root": str(feature_root.resolve()),
        "feature_summary_sha256": file_sha256(feature_summary_path), "feature_file_sha256": feature_summary["feature_file_sha256"],
        "feature_stats_sha256": feature_summary["feature_stats_sha256"], "candidate_geometry_dim": geometry_dim,
        "sampler": {"mode": "record_balanced", "episodes_per_record_per_epoch": episodes_per_record},
        "loss": {"regression": "SmoothL1", "ranking": "stay-inclusive soft-target cross-entropy", "lambda_reg": lambda_reg, "lambda_rank": lambda_rank, "tau": tau},
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model_type": model_type, "parameter_count": summary["parameter_count"], "selected_epoch": best_epoch}, ensure_ascii=False))
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("joint_aware_set_ranker", "candidate_conditioned_attention", "skeleton_policy_transformer"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes-per-record", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(feature_root=args.feature_root, stage_b_root=args.stage_b_root, output_dir=args.output_dir, model_type=args.model_type, device_name=args.device, batch_size=args.batch_size, episodes_per_record=args.episodes_per_record, max_epochs=args.max_epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay, lambda_reg=args.lambda_reg, lambda_rank=args.lambda_rank, tau=args.tau, seed=args.seed)


if __name__ == "__main__":
    main()
