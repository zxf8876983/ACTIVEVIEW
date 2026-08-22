#!/usr/bin/env python3
"""
主动候选视点与可行性过滤可视化工具 —— viewpoint_visualizer.py
============================================================

职责：
    1. 在 2D 鸟瞰图 (Top-Down Bird's-Eye View) 与极坐标雷达图中可视化展示：
       - 人体目标位置 (Human Target)
       - 机器人当前位置 (Robot Current Position)
       - 原始候选视点 (Raw Candidates: 8 方位角 x 4 视距 = 32 个)
       - 被过滤的无效视点 (Filtered Candidates: 不可行走 / 不可达 / 视线遮挡)
       - 最终可行观察视点 (Feasible Viewpoints: 附带朝向人体箭头与导航代价标签)
    2. 生成高清可视化图片 (PNG)，用于人工检查与科研论文插图。
"""

import argparse
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

from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.active_view.visibility_checker import VisibilityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viewpoint_visualizer")


def visualize_candidate_viewpoints(
    human_position: List[float] = [0.0, 0.0, 0.0],
    robot_position: List[float] = [2.0, 0.0, 3.5],
    obstacles: Optional[List[Dict[str, Any]]] = None,
    forbidden_boxes: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    生成候选视点与过滤状态的综合可视化图表。
    """
    out_p = Path(output_path) if output_path else (Path("outputs/v11_visualization/candidate_viewpoints.png"))
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 默认添加示例障碍物 (家具/立柱/墙体)
    if obstacles is None:
        obstacles = [
            {"name": "Wall_Obstacle", "min": [0.8, -0.5, 0.5], "max": [2.2, 2.5, 1.8]},
        ]
    if forbidden_boxes is None:
        forbidden_boxes = [
            {"name": "Forbidden_Zone", "min": [0.7, -1.0, 0.4], "max": [2.3, 1.0, 1.9]},
        ]

    # 1. 生成候选视点
    generator = CandidateViewGenerator()
    raw_candidates = generator.generate(human_position, robot_current_position=robot_position)

    # 2. 过滤候选视点
    vis_checker = VisibilityChecker(obstacles=obstacles)
    habitat_filter = HabitatViewFilter(
        visibility_checker=vis_checker,
        nav_bounds={"min": [-6, -2, -6], "max": [6, 2, 6], "forbidden_boxes": forbidden_boxes},
    )
    feasible_viewpoints = habitat_filter.filter_viewpoints(
        candidates=raw_candidates,
        human_position=human_position,
        robot_current_position=robot_position,
    )

    # 3. 绘制专业科研可视化图表 (2D Top-Down + Polar Radar Grid)
    fig, (ax_2d, ax_polar) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=200)

    hx, hy, hz = human_position
    rx, ry, rz = robot_position

    # -------------------------------------------------------------
    # 子图 1: 2D 鸟瞰图 (Top-Down Plane: X-Z)
    # -------------------------------------------------------------
    ax_2d.set_title("ACTIVEVIEW v11.1: Candidate Viewpoint & Feasibility Filtering (Top-Down X-Z)", fontsize=13, fontweight="bold", pad=12)
    ax_2d.set_xlabel("X Axis (meters)", fontsize=11)
    ax_2d.set_ylabel("Z Axis (meters)", fontsize=11)
    ax_2d.grid(True, linestyle="--", alpha=0.5)
    ax_2d.set_aspect("equal", "box")

    # 绘制同心采样圆环 (1.5m, 2.0m, 2.5m, 3.0m)
    for dist in generator.distances:
        circle = plt.Circle((hx, hz), dist, color="navy", fill=False, linestyle=":", alpha=0.35, linewidth=1.2)
        ax_2d.add_patch(circle)
        ax_2d.text(hx + dist * 0.707, hz + dist * 0.707, f"r={dist}m", fontsize=8, color="navy", alpha=0.7)

    # 绘制障碍物区域
    for obs in obstacles:
        b_min, b_max = obs["min"], obs["max"]
        rect = plt.Rectangle((b_min[0], b_min[2]), b_max[0] - b_min[0], b_max[2] - b_min[2],
                             color="#E74C3C", alpha=0.30, hatch="//", label="Physical Obstacle / Occlusion")
        ax_2d.add_patch(rect)
        ax_2d.text((b_min[0] + b_max[0]) / 2, (b_min[2] + b_max[2]) / 2, obs.get("name", "Obstacle"),
                   fontsize=9, color="#922B21", ha="center", va="center", fontweight="bold")

    # 绘制机器人当前位置
    ax_2d.scatter(rx, rz, color="#2980B9", marker="s", s=140, edgecolor="black", linewidth=1.5, zorder=5, label=f"Robot Start ({rx:.1f}, {rz:.1f})")

    # 绘制人体目标位置
    ax_2d.scatter(hx, hz, color="#C0392B", marker="*", s=280, edgecolor="black", linewidth=1.5, zorder=6, label=f"Human Target ({hx:.1f}, {hz:.1f})")

    # 绘制各类视点
    feasible_ids = {v.id for v in feasible_viewpoints}

    for vp in raw_candidates:
        vx, vz = vp.position[0], vp.position[2]
        if vp.id in feasible_ids:
            # 可行视点: 绿色圆点 + 朝向人体的箭头
            ax_2d.scatter(vx, vz, color="#27AE60", marker="o", s=90, edgecolor="#145A32", linewidth=1.2, zorder=4)
            # 计算朝向人体的方向小箭头
            arrow_dx = (hx - vx) * 0.25 / vp.distance
            arrow_dz = (hz - vz) * 0.25 / vp.distance
            ax_2d.arrow(vx, vz, arrow_dx, arrow_dz, head_width=0.08, head_length=0.08, fc="#1E8449", ec="#1E8449", zorder=4)
        else:
            # 不可行视点: 灰色叉号
            ax_2d.scatter(vx, vz, color="#95A5A6", marker="x", s=70, linewidth=1.5, zorder=3)

    # 绘制图例虚拟点
    ax_2d.scatter([], [], color="#27AE60", marker="o", s=90, label=f"Feasible Viewpoint ({len(feasible_viewpoints)} / {len(raw_candidates)})")
    ax_2d.scatter([], [], color="#95A5A6", marker="x", s=70, label=f"Filtered Viewpoint ({len(raw_candidates) - len(feasible_viewpoints)} / {len(raw_candidates)})")

    ax_2d.legend(loc="upper left", framealpha=0.92, fontsize=9)

    # -------------------------------------------------------------
    # 子图 2: 极坐标视点分布与方位角雷达图 (Polar Viewpoint Radar)
    # -------------------------------------------------------------
    ax_polar.set_title("Polar Candidate Grid & Filter Status", fontsize=13, fontweight="bold", pad=12)
    ax_polar.set_xlabel("Horizontal Azimuth Angle θ (degrees)", fontsize=11)
    ax_polar.set_ylabel("Observation Radius r (meters)", fontsize=11)
    ax_polar.grid(True, linestyle=":", alpha=0.6)

    # 绘制角度-距离网格矩阵
    for vp in raw_candidates:
        ang = vp.angle
        r = vp.distance
        if vp.id in feasible_ids:
            ax_polar.scatter(ang, r, color="#27AE60", marker="o", s=110, edgecolor="#145A32", linewidth=1.5, zorder=4)
            ax_polar.text(ang, r + 0.08, f"cost={vp.navigation_cost:.1f}m", fontsize=7, color="#196F3D", ha="center")
        else:
            reason = "Nav" if not vp.is_navigable else ("Reach" if not vp.is_reachable else "Occluded")
            ax_polar.scatter(ang, r, color="#E74C3C", marker="x", s=90, linewidth=1.5, zorder=3)
            ax_polar.text(ang, r + 0.08, reason, fontsize=7, color="#78281F", ha="center")

    ax_polar.set_xticks(generator.angles)
    ax_polar.set_yticks(generator.distances)
    ax_polar.set_ylim(1.2, 3.4)

    plt.tight_layout()
    plt.savefig(out_p, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved viewpoint visualization to: %s", out_p.resolve())
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Visualize Candidate Viewpoints and Filtering")
    parser.add_argument("--output", type=str, default="outputs/v11_visualization/candidate_viewpoints.png", help="Output PNG path")
    parser.add_argument("--human_x", type=float, default=0.0)
    parser.add_argument("--human_z", type=float, default=0.0)
    parser.add_argument("--robot_x", type=float, default=2.0)
    parser.add_argument("--robot_z", type=float, default=3.5)
    args = parser.parse_args()

    visualize_candidate_viewpoints(
        human_position=[args.human_x, 0.0, args.human_z],
        robot_position=[args.robot_x, 0.0, args.robot_z],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
