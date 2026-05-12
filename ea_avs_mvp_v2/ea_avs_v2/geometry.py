"""
几何计算工具模块 —— geometry.py
=================================

功能：
    提供 MVP v2.0 所需的所有几何计算函数，包括角度归一化、朝向计算、
    视场角判断、高斯评分等。

坐标系约定：
    - Y 轴向上（符合 Habitat-Sim 坐标系）
    - yaw = 0 表示朝向 +Z 方向
    - yaw 为正表示绕 Y 轴逆时针旋转（朝向 +X 方向）
    - 相机位置 = robot_base_pos + [0, camera_height, 0]

本模块不依赖 Habitat API，可独立测试。
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """将任意角度归一化到 [-π, π] 范围内。

    参数：
        angle: 输入角度（弧度制）。

    返回：
        归一化后的角度，范围 [-π, π]。
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_to_forward(yaw: float) -> np.ndarray:
    """将 yaw 朝向角转换为三维前向向量。

    参数：
        yaw: 朝向角（弧度制）。

    返回：
        shape=(3,) 的前向单位向量，Y 分量为 0。
        yaw=0 → (0, 0, 1)，yaw=π/2 → (1, 0, 0)。
    """
    return np.array([np.sin(yaw), 0.0, np.cos(yaw)])


def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """计算从源位置看向目标位置所需的 yaw 朝向角。

    参数：
        source_pos: 源位置，shape=(3,)。
        target_pos: 目标位置，shape=(3,)。

    返回：
        yaw 角（弧度制），使得 source 朝向 target。
    """
    dx = target_pos[0] - source_pos[0]
    dz = target_pos[2] - source_pos[2]
    return float(np.arctan2(dx, dz))


def angle_in_camera_fov(
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    point: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    camera_height: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    """判断三维点是否在相机视场角（FOV）内。

    同时检查水平角度、垂直角度和深度距离三个维度。

    参数：
        camera_base_pos: 机器人基座位置，shape=(3,)。
        camera_yaw: 相机朝向角（弧度）。
        point: 待判断的三维点，shape=(3,)。
        hfov_deg: 水平视场角（度）。
        vfov_deg: 垂直视场角（度）。
        camera_height: 相机安装高度（米）。
        min_depth: 最小有效深度（米）。
        max_depth: 最大有效深度（米）。

    返回：
        字典：{"in_fov", "horizontal_angle", "vertical_angle", "distance", "depth"}
    """
    camera_pos = camera_base_pos + np.array([0.0, camera_height, 0.0])
    vec = point - camera_pos

    horizontal_dist = np.sqrt(vec[0]**2 + vec[2]**2)
    distance = float(np.linalg.norm(vec))

    # 水平夹角
    point_yaw = float(np.arctan2(vec[0], vec[2]))
    horizontal_angle = normalize_angle(point_yaw - camera_yaw)

    # 垂直夹角
    vertical_angle = float(np.arctan2(vec[1], horizontal_dist))

    hfov_rad = np.deg2rad(hfov_deg / 2.0)
    vfov_rad = np.deg2rad(vfov_deg / 2.0)

    in_h = abs(horizontal_angle) <= hfov_rad
    in_v = abs(vertical_angle) <= vfov_rad
    in_d = min_depth <= distance <= max_depth
    in_fov = bool(in_h and in_v and in_d)

    return {
        "in_fov": in_fov,
        "horizontal_angle": horizontal_angle,
        "vertical_angle": vertical_angle,
        "distance": distance,
        "depth": distance,
    }


def gaussian_score(value: float, optimal: float, sigma: float) -> float:
    """计算高斯形状的评分，范围 (0, 1]。

    公式：score = exp(-(value - optimal)² / (2σ²))

    参数：
        value: 输入值。
        optimal: 最优值（高斯中心）。
        sigma: 标准差。

    返回：
        [0, 1] 范围的评分，在 value=optimal 时取最大值 1。
    """
    return float(np.exp(-((value - optimal) ** 2) / (2 * sigma ** 2)))
