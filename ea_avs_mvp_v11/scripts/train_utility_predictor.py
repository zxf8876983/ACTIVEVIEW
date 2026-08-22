#!/usr/bin/env python3
"""
ACTIVEVIEW v11.3 Viewpoint Utility Predictor Training & Evaluation Script
========================================================================

职责：
    1. 自动构建或加载 v11_utility_dataset 监督数据集；
    2. 训练轻量级 ViewpointUtilityPredictorNet MLP 神经网络；
    3. 计算回归损失 (MSE) 与排序指标 (Spearman Rank Correlation)；
    4. 保存最优权重到 checkpoints/ 目录；
    5. 在测试集上运行 4 大基准选择策略评测 (Random vs Nearest vs Ours vs Oracle)；
    6. 输出评测指标汇总报告。
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.models.utility_predictor import ViewpointUtilityPredictorNet
from ea_avs_mvp_v11.active_view.utility_dataset import UtilityDatasetBuilder
from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_selection import evaluate_viewpoint_selection_benchmarks
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_utility_predictor")


class UtilityDataset(Dataset):
    """PyTorch Dataset 包装。"""

    def __init__(self, records: List[Dict[str, Any]]):
        self.features = np.array([r["features"] for r in records], dtype=np.float32)
        self.targets = np.array([r["target_utility"] for r in records], dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(feats)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(feats)
    return total_loss / len(loader.dataset)


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            preds = model(feats)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(math.sqrt(mse))

    # Spearman 秩相关系数
    spearman_corr, _ = spearmanr(preds_arr, targets_arr)
    spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "spearman_rho": round(spearman_val, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ACTIVEVIEW v11.3 Viewpoint Utility Predictor")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Utility dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--rebuild_dataset", action="store_true", help="Force rebuild utility dataset")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    data_root = get_data_root()
    util_dir = Path(args.dataset_dir) if args.dataset_dir else (data_root / "v11_utility_dataset")

    # 1. 构建或加载 Utility Dataset
    if args.rebuild_dataset or not (util_dir / "train.json").exists():
        builder = UtilityDatasetBuilder(data_root=data_root)
        builder.build_utility_dataset(output_dir=util_dir)

    with open(util_dir / "train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(util_dir / "val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open(util_dir / "test.json", "r", encoding="utf-8") as f:
        test_records = json.load(f)

    logger.info("Loaded Utility Dataset from %s: Train=%d, Val=%d, Test=%d",
                util_dir, len(train_records), len(val_records), len(test_records))

    train_loader = DataLoader(UtilityDataset(train_records), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(UtilityDataset(val_records), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(UtilityDataset(test_records), batch_size=args.batch_size, shuffle=False)

    in_dim = len(train_records[0]["features"])
    model = ViewpointUtilityPredictorNet(in_dim=in_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 2. 模型训练与早停监控
    best_val_spearman = -1.0
    best_val_metrics = {}
    best_model_state = None

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (data_root / "checkpoints" / "v11_utility")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "utility_predictor_best.pth"

    logger.info("Starting training for %d epochs...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_metrics(model, val_loader, device)

        if val_metrics["spearman_rho"] > best_val_spearman:
            best_val_spearman = val_metrics["spearman_rho"]
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "val_metrics": val_metrics,
                "in_dim": in_dim,
            }, best_ckpt_path)

        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info("Epoch [%02d/%02d] - Train Loss: %.6f | Val MSE: %.6f, MAE: %.6f, Spearman Rho: %.4f (Best: %.4f)",
                        epoch, args.epochs, train_loss, val_metrics["mse"], val_metrics["mae"],
                        val_metrics["spearman_rho"], best_val_spearman)

    # 3. 加载最优模型并在测试集上评测
    logger.info("Loading best model checkpoint from %s...", best_ckpt_path)
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_metrics(model, test_loader, device)

    logger.info("=================================================================")
    logger.info("  Utility Predictor Training & Evaluation Summary               ")
    logger.info("=================================================================")
    logger.info("  Best Val Metrics:  MSE=%.6f, MAE=%.6f, RMSE=%.6f, Spearman=%.4f",
                best_val_metrics["mse"], best_val_metrics["mae"], best_val_metrics["rmse"], best_val_metrics["spearman_rho"])
    logger.info("  Test Set Metrics:  MSE=%.6f, MAE=%.6f, RMSE=%.6f, Spearman=%.4f",
                test_metrics["mse"], test_metrics["mae"], test_metrics["rmse"], test_metrics["spearman_rho"])
    logger.info("=================================================================")

    # 4. 执行测试集 4 大策略对比实验
    predictor_service = ViewpointUtilityPredictor(model_path=best_ckpt_path, in_dim=in_dim, device=str(device))
    benchmark_res = evaluate_viewpoint_selection_benchmarks(test_records, predictor_service, seed=args.seed)

    logger.info("=================================================================")
    logger.info("  ACTIVE VIEWPOINT SELECTION BENCHMARK COMPARISON (Test Set)    ")
    logger.info("=================================================================")
    logger.info("  Instances Evaluated: %d", benchmark_res["num_test_instances"])
    logger.info("  ---------------------------------------------------------------")
    logger.info("  Strategy            | Entropy H(v) | Gain ΔH | Oracle Gap | Top-1 Acc | Nav Cost")
    logger.info("  --------------------+--------------+---------+------------+-----------+---------")
    for policy, s in benchmark_res["policy_summary"].items():
        logger.info("  %-19s | %12.4f | %7.4f | %10.4f | %8.2f%% | %8.4f",
                    policy, s["mean_selected_entropy"], s["mean_entropy_reduction"],
                    s["mean_oracle_gap"], s["top1_accuracy"] * 100, s["mean_navigation_cost"])
    logger.info("=================================================================")

    # 5. 保存评测结果汇总
    results_payload = {
        "training_args": vars(args),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "benchmark_summary": benchmark_res["policy_summary"],
        "checkpoint_path": str(best_ckpt_path),
    }

    results_file = util_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Saved evaluation results to: %s", results_file)


if __name__ == "__main__":
    main()
