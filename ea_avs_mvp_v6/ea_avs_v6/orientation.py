"""
人体朝向与相对观察方向模块 —— orientation.py
==============================================

功能：
    处理人体朝向建模以及候选视角相对人体的方位关系与高斯朝向评分。
"""

import numpy as np
from .geometry import normalize_angle


def human_yaw_to_forward(human_yaw: float) -> np.ndarray:
    """将人体朝向角转换为前向向量 (yaw=0 -> +Z)。"""
    return np.array([np.sin(human_yaw), 0.0, np.cos(human_yaw)], dtype=np.float64)


def compute_view_direction_from_human(
    human_pos: np.ndarray,
    view_pos: np.ndarray,
) -> np.ndarray:
    """计算从人体指向候选视角的方向向量。"""
    vec = np.array(view_pos, dtype=np.float64) - np.array(human_pos, dtype=np.float64)
    vec[1] = 0.0
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def compute_relative_view_angle(
    human_pos: np.ndarray,
    human_yaw: float,
    view_pos: np.ndarray,
) -> float:
    """计算候选视角相对于人体正面的偏角 [-π, π]。"""
    human_forward = human_yaw_to_forward(human_yaw)
    view_dir = compute_view_direction_from_human(human_pos, view_pos)

    cross_y = human_forward[0] * view_dir[2] - human_forward[2] * view_dir[0]
    dot = human_forward[0] * view_dir[0] + human_forward[2] * view_dir[2]

    relative_angle = float(np.arctan2(cross_y, dot))
    return relative_angle


def compute_orientation_score(relative_angle: float, config: dict) -> float:
    """根据相对角度计算朝向适配得分（侧前方约 45° 最优）。"""
    orient_cfg = config["orientation_score"]
    preferred_deg = orient_cfg["preferred_angle_deg"]
    sigma_deg = orient_cfg["sigma_deg"]

    preferred_rad = np.deg2rad(preferred_deg)
    sigma_rad = np.deg2rad(sigma_deg)

    abs_angle = abs(relative_angle)
    score = float(np.exp(-((abs_angle - preferred_rad) ** 2) / (2 * sigma_rad ** 2)))
    return score
