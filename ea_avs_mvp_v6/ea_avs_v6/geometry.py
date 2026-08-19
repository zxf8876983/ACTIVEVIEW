"""
几何计算工具模块 —— geometry.py
=================================

功能：
    提供 ACTIVEVIEW 所需的基础几何计算、角度归一化、视场角推导、相机内参及逆投影函数。
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """将任意角度归一化到 [-π, π] 范围。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_to_forward(yaw: float) -> np.ndarray:
    """将 yaw 朝向角转换为三维前向向量 (yaw=0 -> +Z)。"""
    return np.array([np.sin(yaw), 0.0, np.cos(yaw)], dtype=np.float64)


def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """计算从源位置看向目标位置所需的 yaw 朝向角。"""
    dx = target_pos[0] - source_pos[0]
    dz = target_pos[2] - source_pos[2]
    return float(np.arctan2(dx, dz))


def unit_direction(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    """计算从 from_pos 指向 to_pos 的单位方向向量。"""
    vec = np.array(to_pos, dtype=np.float64) - np.array(from_pos, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def compute_vfov_deg(width: int, height: int, hfov_deg: float) -> float:
    """根据 pinhole 相机模型从 HFOV 与图像分辨率推导垂直视场角 VFOV。"""
    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) * height / width)
    return float(np.rad2deg(vfov_rad))


def compute_camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    """统一相机内参计算（方形像素假设 fx == fy）。"""
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": int(width),
        "height": int(height),
        "hfov_deg": float(hfov_deg),
        "vfov_deg": compute_vfov_deg(width, height, hfov_deg),
    }


def angle_in_camera_fov(
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    point: np.ndarray,
    hfov_deg: float,
    width: int,
    height: int,
    camera_height: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    """判断三维点是否在相机视场角（FOV）内。"""
    intrinsics = compute_camera_intrinsics(width, height, hfov_deg)
    vfov_deg = intrinsics["vfov_deg"]

    camera_pos = camera_base_pos + np.array([0.0, camera_height, 0.0])
    vec = point - camera_pos

    horizontal_dist = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
    distance = float(np.linalg.norm(vec))

    point_yaw = float(np.arctan2(vec[0], vec[2]))
    horizontal_angle = normalize_angle(point_yaw - camera_yaw)
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
    """计算高斯形状评分，范围 (0, 1]。"""
    return float(np.exp(-((value - optimal) ** 2) / (2 * sigma ** 2)))
