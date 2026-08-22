"""
ACTIVEVIEW v11.4 闭环主动感知科研可视化引擎 —— closed_loop_visualizer.py
======================================================================

职责：
    1. 绘制包含 5 大核心科研维度的闭环主动感知全景图：
       - (A) 2D 机器人导航轨迹与中间路径点 (Robot Trajectories & Waypoints)
       - (B) 初始视点 vs 策略选中视点极坐标分布 (Initial vs Selected Polar Distribution)
       - (C) 导航前后不确定度熵对比 (Entropy Before vs After: H_initial vs H_after)
       - (D) 多策略导航距离与能耗对比 (Navigation Distance & Path Cost)
       - (E) 导航效率与闭环成功率 (Navigation Efficiency & Success Rate)
    2. 输出高清 publication-ready 论文图表 (PNG, 300 DPI)。
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

from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("closed_loop_visualizer")

# 统一出版级学术配色
COLORS = {
    "random": "#95a5a6",           # 灰色
    "nearest": "#e67e22",          # 橙色
    "fixed_front": "#9b59b6",      # 紫色
    "utility_predictor": "#2ecc71",# 鲜绿色 (Ours)
    "oracle": "#3498db",           # 蓝色
    "initial": "#e74c3c",          # 红色
}

LABELS = {
    "random": "Random View",
    "nearest": "Nearest View (Distance)",
    "fixed_front": "Fixed Front View",
    "utility_predictor": "Utility Predictor (Ours)",
    "oracle": "Oracle (Upper Bound)",
}


class ClosedLoopVisualizer:
    """闭环主动感知科研可视化器。"""

    def __init__(self, data_root: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.data_root = data_root if data_root else get_data_root()
        self.output_dir = output_dir if output_dir else (repo_root / "outputs" / "v11_visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_closed_loop_evaluation(
        self,
        dataset_dir: Optional[Path] = None,
        save_name: str = "closed_loop_active_perception_evaluation.png",
    ) -> Path:
        """生成 5 子图全景学术可视化图。"""
        d_dir = dataset_dir if dataset_dir else (self.data_root / "v11_closed_loop_dataset")
        eval_f = d_dir / "evaluation_results.json"
        meta_f = d_dir / "episodes_metadata.json"

        if not eval_f.exists():
            raise FileNotFoundError(f"Evaluation results not found at: {eval_f}")

        with open(eval_f, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        summary = eval_data["benchmark_summary"]

        episodes = []
        if meta_f.exists():
            with open(meta_f, "r", encoding="utf-8") as f:
                episodes = json.load(f)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig = plt.figure(figsize=(20, 12), dpi=300)

        # -------------------------------------------------------------
        # Subplot 1: 机器人 2D 导航轨迹 (Robot Navigation 2D Trajectories)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title("(a) Robot Navigation Trajectories & Waypoints", fontsize=12, fontweight="bold", pad=10)

        # 绘制人体中心与视线朝向
        ax1.scatter([0.0], [0.0], color="red", s=140, marker="*", label="Human (Target)", zorder=5)
        ax1.arrow(0.0, 0.0, 0.0, 0.6, head_width=0.15, head_length=0.15, fc="red", ec="red", zorder=5)

        # 绘制代表性 Episode 的各策略轨迹
        if episodes:
            sample_ep = episodes[0]
            hx, _, hz = sample_ep["human_placement"]["human_position"]

            for p_name in ["random", "nearest", "fixed_front", "utility_predictor"]:
                p_res = sample_ep["policy_results"][p_name]
                wpts = np.array(p_res["trajectory"]["waypoints"])
                # 相对人体中心坐标化
                rel_x = wpts[:, 0] - hx
                rel_z = wpts[:, 2] - hz
                ax1.plot(rel_x, rel_z, "-o", color=COLORS[p_name], label=LABELS[p_name],
                         linewidth=2.0, markersize=4, alpha=0.85)

        ax1.set_xlabel("X (m, Relative to Human)", fontsize=10)
        ax1.set_ylabel("Z (m, Relative to Human)", fontsize=10)
        ax1.set_xlim(-6.0, 6.0)
        ax1.set_ylim(-6.0, 6.0)
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Subplot 2: 初始视点 vs 策略选中视点极坐标分布 (Polar Viewpoint Distribution)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2, polar=True)
        ax2.set_title("(b) Viewpoint Distribution: Initial vs Selected", fontsize=12, fontweight="bold", pad=15)

        if episodes:
            init_angles = [math.radians(ep["robot_initial_viewpoint"]["angle_to_human"]) for ep in episodes[:50]]
            init_dists = [ep["robot_initial_viewpoint"]["distance_to_human"] for ep in episodes[:50]]
            ax2.scatter(init_angles, init_dists, color=COLORS["initial"], s=25, alpha=0.6, label="Initial Robots", marker="x")

            for p_name in ["nearest", "utility_predictor"]:
                p_angles = [math.radians(ep["policy_results"][p_name]["selected_viewpoint"]["angle"]) for ep in episodes[:50]]
                p_dists = [ep["policy_results"][p_name]["selected_viewpoint"]["distance"] for ep in episodes[:50]]
                ax2.scatter(p_angles, p_dists, color=COLORS[p_name], s=40, alpha=0.75, label=f"{LABELS[p_name]}")

        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_ylim(0.0, 7.5)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=8, framealpha=0.9)

        # -------------------------------------------------------------
        # Subplot 3: 导航前后不确定度熵对比 (Entropy Before vs After)
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title("(c) Action Uncertainty Reduction (Shannon Entropy)", fontsize=12, fontweight="bold", pad=10)

        policies = ["random", "nearest", "fixed_front", "utility_predictor", "oracle"]
        x_pos = np.arange(len(policies))
        width = 0.35

        h_before = [summary[p]["mean_entropy_before"] for p in policies]
        h_after = [summary[p]["mean_entropy_after"] for p in policies]

        b1 = ax3.bar(x_pos - width/2, h_before, width, label="Initial H_before", color="#e74c3c", alpha=0.85)
        b2 = ax3.bar(x_pos + width/2, h_after, width, label="Post-Nav H_after", color=[COLORS[p] for p in policies], alpha=0.9)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([p.replace("_", "\n").title() for p in policies], fontsize=9)
        ax3.set_ylabel("Shannon Entropy (nats)", fontsize=10)
        ax3.set_ylim(0.0, 0.8)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        for rect in b2:
            h = rect.get_height()
            ax3.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # -------------------------------------------------------------
        # Subplot 4: 导航移动距离对比 (Navigation Distance)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title("(d) Mean Navigation Distance (Travel Cost)", fontsize=12, fontweight="bold", pad=10)

        dists = [summary[p]["mean_navigation_distance_m"] for p in policies]
        bars_dist = ax4.bar(x_pos, dists, width=0.55, color=[COLORS[p] for p in policies], alpha=0.9, edgecolor="black", linewidth=0.8)

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([p.replace("_", "\n").title() for p in policies], fontsize=9)
        ax4.set_ylabel("Navigation Distance (meters)", fontsize=10)
        ax4.set_ylim(0.0, 6.5)
        ax4.grid(True, linestyle="--", alpha=0.5)

        for rect in bars_dist:
            h = rect.get_height()
            ax4.annotate(f"{h:.2f} m", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # -------------------------------------------------------------
        # Subplot 5: 导航效率对比 (Navigation Efficiency: Gain / Distance)
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title("(e) Active Perception Efficiency (ΔH / Distance)", fontsize=12, fontweight="bold", pad=10)

        efficiencies = [summary[p]["mean_navigation_efficiency"] for p in policies]
        bars_eff = ax5.bar(x_pos, efficiencies, width=0.55, color=[COLORS[p] for p in policies], alpha=0.9, edgecolor="black", linewidth=0.8)

        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([p.replace("_", "\n").title() for p in policies], fontsize=9)
        ax5.set_ylabel("Efficiency (nats / meter)", fontsize=10)
        ax5.set_ylim(0.0, max(efficiencies) * 1.3)
        ax5.grid(True, linestyle="--", alpha=0.5)

        for rect in bars_eff:
            h = rect.get_height()
            ax5.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # -------------------------------------------------------------
        # Subplot 6: 闭环信息增益与成功率总结 (Information Gain & Success Summary)
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title("(f) Information Gain (ΔH) & Success Rate", fontsize=12, fontweight="bold", pad=10)

        gains = [summary[p]["mean_entropy_reduction"] for p in policies]
        bars_gain = ax6.bar(x_pos, gains, width=0.55, color=[COLORS[p] for p in policies], alpha=0.9, edgecolor="black", linewidth=0.8)

        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([p.replace("_", "\n").title() for p in policies], fontsize=9)
        ax6.set_ylabel("Entropy Gain ΔH (nats)", fontsize=10)
        ax6.set_ylim(0.0, 0.8)
        ax6.grid(True, linestyle="--", alpha=0.5)

        for rect in bars_gain:
            h = rect.get_height()
            ax6.annotate(f"+{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.suptitle("ACTIVEVIEW v11.4 Closed-Loop Active Perception Scientific Benchmark",
                     fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved closed-loop scientific visualization to: %s", out_path)
        return out_path
