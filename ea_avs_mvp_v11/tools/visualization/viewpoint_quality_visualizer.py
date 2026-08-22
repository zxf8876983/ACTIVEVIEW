#!/usr/bin/env python3
"""
视点识别质量与不确定度热力图可视化工具 —— viewpoint_quality_visualizer.py
========================================================================

职责：
    1. 生成角度-距离不确定度热力图 (Angle-Distance Entropy Heatmap):
       - 横轴: 水平方位角 θ (0°, 45°, 90°, ..., 315°)
       - 纵轴: 观察距离 r (1.5m, 2.0m, 2.5m, 3.0m)
       - 颜色深度: 平均预测信息熵 H(p)
    2. 生成动作专属性视点识别质量雷达图 (Action-Specific Viewpoint Profiles):
       - 展现 6 大动作类别在不同观察视角下的识别质量与信息熵响应特征
    3. 输出高质量科研论文插图 (PNG)。
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viewpoint_quality_visualizer")

ACTION_CLASSES = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]


def visualize_viewpoint_quality(
    dataset_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    加载 Viewpoint Quality Dataset 并生成综合质量分析图表。
    """
    data_root = get_data_root()
    d_dir = Path(dataset_dir) if dataset_dir else (data_root / "v11_viewpoint_dataset")
    out_p = Path(output_path) if output_path else (Path("outputs/v11_visualization/viewpoint_quality_heatmap.png"))
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = d_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at: {metadata_path}. Please run dataset generation first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info("Loaded %d samples from %s", len(samples), metadata_path)

    # 1. 构建角度-距离热力图矩阵 (8 角度 x 4 距离)
    angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    distances = [1.5, 2.0, 2.5, 3.0]

    heatmap_entropy = np.zeros((len(distances), len(angles)), dtype=np.float32)
    heatmap_acc = np.zeros((len(distances), len(angles)), dtype=np.float32)
    heatmap_count = np.zeros((len(distances), len(angles)), dtype=np.int32)

    # 按动作类别收集角度熵响应
    action_angle_entropy: Dict[str, Dict[float, List[float]]] = {
        act: {ang: [] for ang in angles} for act in ACTION_CLASSES
    }

    for s in samples:
        vp = s["viewpoint"]
        ang = float(vp["angle"])
        dist = float(vp["distance"])
        ent = float(s["entropy"])
        acc = 1.0 if s["is_correct"] else 0.0
        act = s["action_label"]

        if ang in angles and dist in distances:
            a_idx = angles.index(ang)
            d_idx = distances.index(dist)
            heatmap_entropy[d_idx, a_idx] += ent
            heatmap_acc[d_idx, a_idx] += acc
            heatmap_count[d_idx, a_idx] += 1

        if act in action_angle_entropy and ang in action_angle_entropy[act]:
            action_angle_entropy[act][ang].append(ent)

    # 计算平均值
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_entropy_matrix = np.where(heatmap_count > 0, heatmap_entropy / np.maximum(heatmap_count, 1), np.nan)
        avg_acc_matrix = np.where(heatmap_count > 0, (heatmap_acc / np.maximum(heatmap_count, 1)) * 100.0, np.nan)

    # 2. 绘制科研多子图可视化 (1x2 布局)
    fig, (ax_heat, ax_act) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

    # -------------------------------------------------------------
    # 子图 1: 角度-距离信息熵热力图 (Angle-Distance Entropy Heatmap)
    # -------------------------------------------------------------
    cmap = plt.cm.get_cmap("YlOrRd")
    im = ax_heat.imshow(avg_entropy_matrix, cmap=cmap, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Prediction Shannon Entropy H(p)", fontsize=11, fontweight="bold")

    ax_heat.set_xticks(range(len(angles)))
    ax_heat.set_xticklabels([f"{int(a)}°" for a in angles], fontsize=10)
    ax_heat.set_yticks(range(len(distances)))
    ax_heat.set_yticklabels([f"{d:.1f}m" for d in distances], fontsize=10)

    ax_heat.set_xlabel("Horizontal Azimuth Angle θ (degrees)", fontsize=11, fontweight="bold")
    ax_heat.set_ylabel("Observation Radius r (meters)", fontsize=11, fontweight="bold")
    ax_heat.set_title("ACTIVEVIEW v11.2: Viewpoint Quality Heatmap (Entropy H(p))", fontsize=12, fontweight="bold", pad=12)

    # 在单元格内标注数值 (熵值 + 准确率)
    for i in range(len(distances)):
        for j in range(len(angles)):
            val_ent = avg_entropy_matrix[i, j]
            val_acc = avg_acc_matrix[i, j]
            if not np.isnan(val_ent):
                color = "white" if val_ent > np.nanmean(avg_entropy_matrix) * 1.1 else "black"
                ax_heat.text(j, i, f"H={val_ent:.3f}\nAcc={val_acc:.1f}%",
                             ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------
    # 子图 2: 动作专属性视点质量曲线 (Action-Specific Quality Profiles)
    # -------------------------------------------------------------
    ax_act.set_title("Action-Specific Viewpoint Uncertainty Profiles H(p)(θ)", fontsize=12, fontweight="bold", pad=12)
    ax_act.set_xlabel("Observation Azimuth Angle θ (degrees)", fontsize=11, fontweight="bold")
    ax_act.set_ylabel("Average Uncertainty Entropy H(p)", fontsize=11, fontweight="bold")
    ax_act.grid(True, linestyle="--", alpha=0.5)

    colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#E74C3C", "#1ABC9C"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for idx, act in enumerate(ACTION_CLASSES):
        mean_ents = [
            float(np.mean(action_angle_entropy[act][ang])) if action_angle_entropy[act][ang] else 0.0
            for ang in angles
        ]
        ax_act.plot(angles, mean_ents, label=act, color=colors[idx], marker=markers[idx],
                    linewidth=2.0, markersize=6.5)

    ax_act.set_xticks(angles)
    ax_act.set_xticklabels([f"{int(a)}°" for a in angles], fontsize=10)
    ax_act.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(out_p, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved viewpoint quality visualization to: %s", out_p.resolve())
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Visualize Viewpoint Quality Dataset Heatmap")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to v11_viewpoint_dataset directory")
    parser.add_argument("--output", type=str, default="outputs/v11_visualization/viewpoint_quality_heatmap.png", help="Output PNG path")
    args = parser.parse_args()

    visualize_viewpoint_quality(dataset_dir=args.dataset_dir, output_path=args.output)


if __name__ == "__main__":
    main()
