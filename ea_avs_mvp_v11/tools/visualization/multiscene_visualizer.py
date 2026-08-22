#!/usr/bin/env python3
"""
多场景主动视角数据集与策略评测科学可视化工具 —— multiscene_visualizer.py (v11.3)
========================================================================

职责：
    1. 生成多场景人体与机器人空间分布图 (Human Placement & Robot Start Distribution)；
    2. 生成候选视角极坐标不确定度衰减热力图 (Polar Viewpoint Uncertainty Heatmap)；
    3. 生成 4 大策略测试集 Benchmark 对比图 (Entropy, Gain, Oracle Gap, Nav Cost)；
    4. 输出高分辨率出版级科研图表到 outputs/v11_visualization/。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.core.paths import get_data_root, get_repo_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiscene_visualizer")


def plot_multiscene_dataset_distribution(
    dataset_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    """绘制多场景数据分布图。"""
    data_root = get_data_root()
    d_dir = Path(dataset_dir) if dataset_dir else (data_root / "v11_multiscene_viewpoint_dataset")
    meta_file = d_dir / "episodes_metadata.json"

    if not meta_file.exists():
        logger.warning("Episodes metadata not found at %s. Skipping plot.", meta_file)
        return

    with open(meta_file, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. 场景与动作类别分布
    scenes = [ep["scene_id"] for ep in episodes]
    actions = [ep["action_label"] for ep in episodes]
    unique_scenes, s_counts = np.unique(scenes, return_counts=True)

    ax1 = axes[0, 0]
    bars = ax1.bar(unique_scenes, s_counts, color=["#2b5c8f", "#418ab3", "#6baed6"], edgecolor="black", width=0.55)
    ax1.set_title("Multi-Scene Episode Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Episode Count", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{int(b.get_height())}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 2. 人体与机器人空间位置平面投影
    ax2 = axes[0, 1]
    hx = [ep["human_placement"]["human_position"][0] for ep in episodes]
    hz = [ep["human_placement"]["human_position"][2] for ep in episodes]
    rx = [ep["current_viewpoint"]["position"][0] for ep in episodes]
    rz = [ep["current_viewpoint"]["position"][2] for ep in episodes]

    ax2.scatter(hx, hz, c="#e74c3c", marker="o", s=40, alpha=0.75, label="Human Positions (Randomized)", edgecolors="black", linewidths=0.5)
    ax2.scatter(rx, rz, c="#2ecc71", marker="^", s=45, alpha=0.65, label="Robot Starts (2m~8m Dist)", edgecolors="black", linewidths=0.5)
    ax2.set_title("Human & Robot Spatial Placement Map", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X Coordinate (m)", fontsize=11)
    ax2.set_ylabel("Z Coordinate (m)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. 初始视距与方位角分布
    ax3 = axes[1, 0]
    dists = [ep["current_viewpoint"]["distance_to_human"] for ep in episodes]
    ax3.hist(dists, bins=15, color="#f39c12", edgecolor="black", alpha=0.85)
    ax3.set_title("Robot Initial Observation Distance Distribution", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Initial Distance to Human (m)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. 候选视点平均后验熵极坐标衰减
    ax4 = axes[1, 1]
    samples_meta_file = d_dir / "metadata.json"
    if samples_meta_file.exists():
        with open(samples_meta_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        angles = [s["viewpoint"]["angle"] for s in samples]
        entropies = [s["entropy"] for s in samples]
        u_angles = np.unique(angles)
        mean_ents = [np.mean([entropies[i] for i in range(len(samples)) if angles[i] == a]) for a in u_angles]

        ax4.plot(u_angles, mean_ents, marker="o", linewidth=2.5, color="#8e44ad", label="Mean Posterior Entropy")
        ax4.set_title("Viewpoint Angle vs Posterior Recognition Entropy", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Observation Polar Angle (deg)", fontsize=11)
        ax4.set_ylabel("Entropy H(v) (nats)", fontsize=11)
        ax4.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
        ax4.set_xticklabels(["0°(Front)", "45°", "90°(Side)", "135°", "180°(Back)", "225°", "270°(Side)", "315°"])
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()

    out_file = Path(output_path) if output_path else (repo_root / "outputs" / "v11_visualization" / "multiscene_active_view_distribution.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    logger.info("Saved multi-scene distribution plot to: %s", out_file)


def plot_multiscene_benchmark_results(
    results_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    """绘制多场景策略 Benchmark 科学对比图。"""
    data_root = get_data_root()
    res_f = Path(results_file) if results_file else (data_root / "v11_multiscene_utility_dataset" / "evaluation_results.json")

    if not res_f.exists():
        logger.warning("Evaluation results not found at %s. Skipping plot.", res_f)
        return

    with open(res_f, "r", encoding="utf-8") as f:
        res = json.load(f)

    summary = res["benchmark_summary"]
    policies = list(summary.keys())
    labels = ["Random\n(Baseline)", "Nearest\n(Distance)", "Utility Predictor\n(Ours)", "Oracle\n(Upper Bound)"]
    colors = ["#95a5a6", "#e67e22", "#27ae60", "#2980b9"]

    entropies = [summary[p]["mean_selected_entropy"] for p in policies]
    gains = [summary[p]["mean_entropy_reduction"] for p in policies]
    nav_costs = [summary[p]["mean_navigation_cost"] for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. 选后平均熵
    ax1 = axes[0]
    bars1 = ax1.bar(labels, entropies, color=colors, edgecolor="black", width=0.55)
    ax1.set_title("Selected Viewpoint Uncertainty H(v)\n(Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Shannon Entropy (nats)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. 信息增益
    ax2 = axes[1]
    bars2 = ax2.bar(labels, gains, color=colors, edgecolor="black", width=0.55)
    ax2.set_title("Active Perception Information Gain ΔH\n(Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Entropy Reduction ΔH (nats)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. 导航代价
    ax3 = axes[2]
    bars3 = ax3.bar(labels, nav_costs, color=colors, edgecolor="black", width=0.55)
    ax3.set_title("Average Navigation Movement Cost\n(Lower Movement)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Euclidean Distance (m)", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}m",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    out_file = Path(output_path) if output_path else (repo_root / "outputs" / "v11_visualization" / "multiscene_utility_evaluation.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    logger.info("Saved multi-scene benchmark plot to: %s", out_file)


def main():
    logger.info("Generating multi-scene scientific visualizations...")
    plot_multiscene_dataset_distribution()
    plot_multiscene_benchmark_results()


if __name__ == "__main__":
    main()
