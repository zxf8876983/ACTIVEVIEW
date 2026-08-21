"""
动作感知视点对比可视化绘图工具 —— action_comparison_plotter.py
=========================================================

功能：
    1. 生成同一场景下不同动作的最佳视角分布与偏角对比；
    2. 生成 v8 (纯几何) vs v9 (动作感知) 视角迁移对比图；
    3. 生成打分组成分解图 (Q_geom vs Delta_Q vs Q_total)。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def plot_action_comparison_figure(
    comparison_data: Dict[str, Any],
    output_image_path: Union[str, Path],
) -> Optional[Path]:
    """生成动作对比可视化多子图 (包含视角迁移、得分分解与指标覆盖)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return None

    out_p = Path(output_image_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    actions_dict = comparison_data.get("actions", {})
    if not actions_dict:
        return None

    act_names = list(actions_dict.keys())
    v8_scores = [actions_dict[a]["geometry_best_viewpoint"]["action_total_score"] for a in act_names]
    v9_scores = [actions_dict[a]["action_conditioned_viewpoint"]["action_total_score"] for a in act_names]
    v8_geom_base = [actions_dict[a]["geometry_best_viewpoint"]["geometry_score"] for a in act_names]
    v9_deltas = [actions_dict[a]["action_conditioned_viewpoint"].get("action_delta", 0.0) for a in act_names]
    v8_angles = [actions_dict[a]["geometry_best_viewpoint"]["viewing_angle_deg"] for a in act_names]
    v9_angles = [actions_dict[a]["action_conditioned_viewpoint"]["viewing_angle_deg"] for a in act_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(act_names))
    bar_w = 0.35

    # 子图 1: v8 vs v9 综合得分 Q(v|a) 对比
    ax1 = axes[0]
    rects1 = ax1.bar(x - bar_w/2, v8_scores, bar_w, label="v8 Geometry Best", color="#4A90E2", alpha=0.85)
    rects2 = ax1.bar(x + bar_w/2, v9_scores, bar_w, label="v9 Action-Conditioned", color="#E94E77", alpha=0.85)
    ax1.set_title("Total Score Q(v|A): v8 vs v9", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in act_names])
    ax1.set_ylabel("Score [0.0 - 1.0]")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 子图 2: 最佳视角观察偏角 (Viewing Angle) 迁移
    ax2 = axes[1]
    ax2.plot(x, v8_angles, marker="o", linewidth=2.5, label="v8 Geometry (0° Frontal)", color="#4A90E2")
    ax2.plot(x, v9_angles, marker="s", linewidth=2.5, label="v9 Action-Conditioned", color="#E94E77")
    ax2.set_title("Optimal Viewing Angle Shift (deg)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in act_names])
    ax2.set_ylabel("Viewing Angle (deg relative to front)")
    ax2.set_ylim(-10, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 子图 3: v9 得分组成分解 (Q_geom vs Delta_Q)
    ax3 = axes[2]
    ax3.bar(x, v8_geom_base, bar_w * 1.2, label="w_geom * Q_geom", color="#50E3C2", alpha=0.85)
    ax3.bar(x, [0.4 * d for d in v9_deltas], bar_w * 1.2, bottom=[0.6 * g for g in v8_geom_base],
            label="w_act * Delta_Q", color="#F5A623", alpha=0.85)
    ax3.set_title("Score Decomposition (Q_geom + Delta_Q)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([a.upper() for a in act_names])
    ax3.set_ylabel("Weighted Contribution")
    ax3.set_ylim(0.0, 1.0)
    ax3.legend(loc="lower right")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close(fig)
    logger.info("Saved action comparison visualization to: %s", out_p)
    return out_p
