#!/usr/bin/env python3
"""Train Stage C Utility predictors with Episode-level record-balanced sampling."""

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

from activeview.active_view.stage_c_dataset import EpisodeFeatureDataset, RecordBalancedSampler, collate_episode_batch, load_feature_statistics
from activeview.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup, predict_dataset
from activeview.active_view.stage_c_losses import stage_c_loss
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import build_utility_predictor, count_parameters
from activeview.core.paths import get_data_root


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_model(*, feature_root: Path, stage_b_root: Path, output_dir: Path, model_type: str, device_name: str, batch_size: int, episodes_per_record: int, max_epochs: int, patience: int, lr: float, weight_decay: float, lambda_reg: float, lambda_rank: float, tau: float, lambda_gap: float = 0.0, tau_gap: float = 1.0, max_gap_weight: float = 10.0, seed: int) -> Dict[str, Any]:
    _seed(seed)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    feature_summary_path = feature_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    feature_file_sha256 = feature_summary["feature_file_sha256"]
    feature_stats_sha256 = feature_summary["feature_stats_sha256"]
    train_set = EpisodeFeatureDataset(feature_root / "features/train.jsonl", **stats)
    val_set = EpisodeFeatureDataset(feature_root / "features/val.jsonl", **stats)
    sampler = RecordBalancedSampler(train_set.rows, episodes_per_record=episodes_per_record, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=collate_episode_batch, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_episode_batch, num_workers=0)
    mapping = json.loads((feature_root.parent.parent / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed" / "label_mapping.json").read_text(encoding="utf-8"))
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    stage_b_lookup = load_stage_b_lookup(stage_b_root / "utility_labels/val.jsonl")
    model = build_utility_predictor(model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf"); best_epoch = 0; stale = 0; history: list[Dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train(); loss_sums = {"total": 0.0, "regression": 0.0, "ranking": 0.0, "gap_ranking": 0.0}; steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["current_feature"].to(device), batch["candidate_geometry"].to(device), batch["candidate_mask"].to(device))
            losses = stage_c_loss(predicted, batch["utility_targets"].to(device), batch["candidate_mask"].to(device), lambda_reg=lambda_reg, lambda_rank=lambda_rank, tau=tau, lambda_gap=lambda_gap, tau_gap=tau_gap, max_gap_weight=max_gap_weight)
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for key in loss_sums:
                loss_sums[key] += float(losses[key].detach().cpu())
            steps += 1
        val_rows = predict_dataset(model, val_loader, stage_b_lookup, device)
        val_metrics = evaluate_predictions(val_rows, categories)
        val_score = float(val_metrics["recognition"]["StageC"]["macro_f1"])
        scheduler.step(val_score)
        history.append({"epoch": epoch, "train_loss": loss_sums["total"] / max(steps, 1), "train_regression_loss": loss_sums["regression"] / max(steps, 1), "train_ranking_loss": loss_sums["ranking"] / max(steps, 1), "train_gap_ranking_loss": loss_sums["gap_ranking"] / max(steps, 1), "val_metrics": val_metrics, "lr": optimizer.param_groups[0]["lr"]})
        improved = val_score > best_metric + 1e-12
        if improved:
            best_metric = val_score; best_epoch = epoch; stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "model_type": model_type, "epoch": epoch,
                "feature_summary_sha256": file_sha256(feature_summary_path),
                "feature_file_sha256": feature_file_sha256, "feature_stats_sha256": feature_stats_sha256,
                "config": {"batch_size": batch_size, "episodes_per_record": episodes_per_record, "lambda_reg": lambda_reg, "lambda_rank": lambda_rank, "tau": tau, "lambda_gap": lambda_gap, "tau_gap": tau_gap, "max_gap_weight": max_gap_weight, "seed": seed},
            }, output_dir / f"{model_type}_best.pth")
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {"stage": "C", "model_type": model_type, "parameter_count": count_parameters(model), "device": str(device), "max_epochs": max_epochs, "selected_epoch": best_epoch, "checkpoint": str((output_dir / f"{model_type}_best.pth").resolve()), "checkpoint_sha256": file_sha256(output_dir / f"{model_type}_best.pth"), "checkpoint_selection_metric": "Val StageC recognition Macro-F1", "feature_summary": str(feature_summary_path.resolve()), "feature_summary_sha256": file_sha256(feature_summary_path), "feature_file_sha256": feature_file_sha256, "feature_stats_sha256": feature_stats_sha256, "sampler": {"mode": "record_balanced", "episodes_per_record_per_epoch": episodes_per_record}, "loss": {"lambda_reg": lambda_reg, "lambda_rank": lambda_rank, "tau": tau, "lambda_gap": lambda_gap, "tau_gap": tau_gap, "max_gap_weight": max_gap_weight, "regression": "SmoothL1", "ranking": "stay-inclusive soft-target cross-entropy", "gap_ranking": "gap-weighted soft pairwise ranking"}, "history": history}
    (output_dir / f"{model_type}_training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--output-dir", type=Path, default=data_root / "checkpoints/stage_c")
    parser.add_argument("--model-type", choices=("pairwise_mlp", "set_ranker"), default="set_ranker")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes-per-record", type=int, default=16); parser.add_argument("--max-epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--lambda-reg", type=float, default=1.0); parser.add_argument("--lambda-rank", type=float, default=1.0); parser.add_argument("--tau", type=float, default=0.5); parser.add_argument("--lambda-gap", type=float, default=0.0); parser.add_argument("--tau-gap", type=float, default=1.0); parser.add_argument("--max-gap-weight", type=float, default=10.0); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_model(feature_root=args.feature_root, stage_b_root=args.stage_b_root, output_dir=args.output_dir, model_type=args.model_type, device_name=args.device, batch_size=args.batch_size, episodes_per_record=args.episodes_per_record, max_epochs=args.max_epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay, lambda_reg=args.lambda_reg, lambda_rank=args.lambda_rank, tau=args.tau, lambda_gap=args.lambda_gap, tau_gap=args.tau_gap, max_gap_weight=args.max_gap_weight, seed=args.seed)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
