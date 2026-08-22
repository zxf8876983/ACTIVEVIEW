#!/usr/bin/env python3
"""
ACTIVEVIEW v11.3 Utility Predictor & Selection Visualizer —— utility_visualizer.py
================================================================================

职责：
    1. 绘制 Ground Truth Utility vs Predicted Utility 散点拟合图 (含 Spearman 相关系数标注)；
    2. 绘制 4 大主动视角选择策略对比柱状图 (Random vs Nearest vs Ours vs Oracle)；
    3. 绘制决策信息收益与移动代价帕累托权衡图 (Entropy Reduction vs Navigation Cost)；
    4. 输出高分辨率科研图表至 outputs/v11_visualization/utility_prediction_evaluation.png。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_selection import evaluate_viewpoint_selection_benchmarks
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("utility_visualizer")


def generate_utility_prediction_plots(
    dataset_dir: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """生成 v11.3 效用预测与基准选择多子图科研可视化。"""
    data_root = get_data_root()
    d_dir = Path(dataset_dir) if dataset_dir else (data_root / "v11_utility_dataset")
    ckpt_p = Path(checkpoint_path) if checkpoint_path else (data_root / "checkpoints" / "v11_utility" / "utility_predictor_best.pth")

    test_file = d_dir / "test.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_file}")

    with open(test_file, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    # 1. 预测测试集效用
    predictor = ViewpointUtilityPredictor(model_path=ckpt_p, in_dim=len(test_records[0]["features"]))

    feats = np.array([r["features"] for r in test_records], dtype=np.float32)
    targets = np.array([r["target_utility"] for r in test_records], dtype=np.float32)

    with torch.no_grad():
        preds = predictor.model(torch.tensor(feats).to(predictor.device)).cpu().numpy().flatten()

    spearman_corr, _ = spearmanr(preds, targets)

    # 2. 评测 4 大基准选择策略
    bench_results = evaluate_viewpoint_selection_benchmarks(test_records, predictor, seed=42)
    summary = bench_results["policy_summary"]

    # 3. 绘图 (1 行 3 列专业图表)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.28, top=0.88, bottom=0.14, left=0.06, right=0.96)

    # --- 子图 1: 真实效用 vs 预测效用散点图 ---
    ax1 = axes[0]
    ax1.scatter(targets, preds, alpha=0.35, color="#2563eb", edgecolors="none", s=24, label="Test Samples")
    # 理想对角线
    min_v, max_v = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.8, label="Ideal Fit (y = x)")
    # 拟合趋势线
    z = np.polyfit(targets, preds, 1)
    p_fn = np.poly1d(z)
    x_line = np.linspace(min_v, max_v, 100)
    ax1.plot(x_line, p_fn(x_line), color="#059669", linewidth=2.0, label=f"Trend (Slope={z[0]:.2f})")

    ax1.set_title(f"Utility Prediction Accuracy\n(Spearman $\\rho$ = {spearman_corr:.4f}, MAE = {np.mean(np.abs(preds-targets)):.4f})",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Ground Truth Utility $U(v) = H_{curr} - H_{cand}$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Utility $\\hat{U}(v)$", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    # --- 子图 2: 4 大策略动作识别平均熵与 Oracle Gap 对比 ---
    ax2 = axes[1]
    policies = ["random", "nearest", "utility_predictor", "oracle"]
    policy_labels = ["Random\n(Baseline)", "Nearest\n(Distance)", "Utility Predictor\n(Ours)", "Oracle\n(Upper Bound)"]
    colors = ["#94a3b8", "#f59e0b", "#10b981", "#6366f1"]

    entropies = [summary[p]["mean_selected_entropy"] for p in policies]
    gaps = [summary[p]["mean_oracle_gap"] for p in policies]

    x_pos = np.arange(len(policies))
    width = 0.38

    bars1 = ax2.bar(x_pos - width/2, entropies, width, label="Selected Entropy $H(v)$", color=colors, alpha=0.85, edgecolor="black")
    bars2 = ax2.bar(x_pos + width/2, gaps, width, label="Oracle Gap $(H - H^*)$", color="#dc2626", alpha=0.6, hatch="//", edgecolor="black")

    for bar in bars1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_title("Active View Selection Quality\n(Lower Entropy & Oracle Gap is Better)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(policy_labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Shannon Entropy / Gap (nats)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right", frameon=True)

    # --- 子图 3: 熵降低增益 vs 移动代价帕累托分布 ---
    ax3 = axes[2]
    for p, label, color in zip(policies, ["Random", "Nearest", "Ours (Utility)", "Oracle"], colors):
        gain = summary[p]["mean_entropy_reduction"]
        cost = summary[p]["mean_navigation_cost"]
        top1 = summary[p]["top1_accuracy"] * 100
        ax3.scatter(cost, gain, s=top1 * 3.5 + 80, color=color, edgecolors="black", linewidth=1.5, zorder=5, label=f"{label} (Top-1: {top1:.1f}%)")
        ax3.annotate(f"{label}\n(Top-1: {top1:.1f}%)", (cost, gain), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax3.set_title("Information Gain vs Navigation Cost Tradeoff", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Mean Navigation Cost (meters)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Entropy Reduction $\\Delta H$ (nats)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", frameon=True)

    # 保存图片
    out_p = Path(output_path) if output_path else (repo_root / "outputs" / "v11_visualization" / "utility_prediction_evaluation.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()

    logger.info("Saved utility prediction evaluation plot to: %s", out_p.resolve())
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Generate ACTIVEVIEW v11.3 Utility Predictor Visualizations")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Utility dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output plot image path")
    args = parser.parse_args()

    generate_utility_prediction_plots(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
