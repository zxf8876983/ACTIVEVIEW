"""
人体朝向与相对观察方向模块 —— orientation.py
==============================================

功能：
    处理人体朝向建模以及候选视角相对人体的方位关系。

v3.0 核心概念：
    - human_yaw：人体正面朝向（弧度制），yaw=0 表示人体面向 +Z
    - relative_view_angle：候选视角相对人体正面的偏角
      * 0° —— 人体正前方
      * ±45° —— 侧前方（最优观察角度）
      * ±90° —— 正侧面
      * ±180° —— 正后方（最差）

朝向评分设计：
    侧前方 45° 得分最高（便于同时观察面部表情和身体动作），
    正前方次之，侧方可接受，背面较差。
"""

import numpy as np

from .geometry import normalize_angle


def human_yaw_to_forward(human_yaw: float) -> np.ndarray:
    """将人体朝向角转换为前向向量。

    参数：
        human_yaw: 人体朝向角（弧度制）。

    返回：
        shape=(3,) 的单位向量，表示人体正面方向。
    """
    return np.array([np.sin(human_yaw), 0.0, np.cos(human_yaw)])


def compute_view_direction_from_human(
    human_pos: np.ndarray,
    view_pos: np.ndarray,
) -> np.ndarray:
    """计算从人体指向候选视角的方向向量。

    参数：
        human_pos: 人体脚底中心位置，shape=(3,)。
        view_pos: 候选视角位置，shape=(3,)。

    返回：
        shape=(3,) 的单位方向向量（从人体指向视角）。
    """
    vec = view_pos - human_pos
    # 忽略 Y 轴高度差，只考虑 X-Z 平面方向
    vec[1] = 0.0
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def compute_relative_view_angle(
    human_pos: np.ndarray,
    human_yaw: float,
    view_pos: np.ndarray,
) -> float:
    """计算候选视角相对于人体正面的偏角。

    参数：
        human_pos: 人体脚底中心位置，shape=(3,)。
        human_yaw: 人体朝向角（弧度制）。
        view_pos: 候选视角位置，shape=(3,)。

    返回：
        相对偏角（弧度制），范围 [-π, π]。
        * 0 —— 候选点在人体正前方
        * ±π/2 —— 候选点在人体正侧面
        * ±π —— 候选点在人体正后方

    实现：
        1. 计算从人体指向视角的方向向量
        2. 计算该方向与人体前向方向的夹角
    """
    # 人体前向向量
    human_forward = human_yaw_to_forward(human_yaw)

    # 从人体指向视角的向量（X-Z 平面）
    view_dir = compute_view_direction_from_human(human_pos, view_pos)

    # 计算夹角
    # arctan2(叉积的 Y 分量, 点积) 可以得到带符号的角度
    cross_y = (human_forward[0] * view_dir[2]
               - human_forward[2] * view_dir[0])
    dot = human_forward[0] * view_dir[0] + human_forward[2] * view_dir[2]

    relative_angle = float(np.arctan2(cross_y, dot))
    return relative_angle


def compute_orientation_score(
    relative_angle: float,
    config: dict,
) -> float:
    """根据相对角度计算朝向适配得分。

    评分原则：
        侧前方 45° 得分最高（便于观察面部 + 身体动作），
        正前方次之，侧面可接受，背面较差。

    实现方法：
        使用高斯函数，峰值在 preferred_angle 处：
        S_orient = exp(-((|relative_angle| - preferred_angle)²) / (2 * σ²))

    参数：
        relative_angle: 相对偏角（弧度制），范围 [-π, π]。
        config: 配置字典，需要 orientation_score 配置段：
            - preferred_angle_deg: 偏好角度（度），默认 45°
            - sigma_deg: 高斯标准差（度），默认 35°

    返回：
        [0, 1] 范围的评分。
    """
    orient_cfg = config["orientation_score"]
    preferred_deg = orient_cfg["preferred_angle_deg"]
    sigma_deg = orient_cfg["sigma_deg"]

    preferred_rad = np.deg2rad(preferred_deg)
    sigma_rad = np.deg2rad(sigma_deg)

    abs_angle = abs(relative_angle)

    # 高斯评分：峰值在 preferred_angle 处
    score = float(np.exp(
        -((abs_angle - preferred_rad) ** 2) / (2 * sigma_rad ** 2)
    ))

    return score
