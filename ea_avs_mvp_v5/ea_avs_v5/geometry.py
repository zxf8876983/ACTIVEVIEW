"""
几何计算工具模块 —— geometry.py
=================================

功能：
    提供 MVP v4.0 所需的基础几何计算函数。
    从 v3.0 迁移，并新增 v4.0 ray casting 需要的单位方向向量函数。

坐标系约定（与 Habitat-Sim 一致）：
    - Y 轴向上
    - yaw = 0 表示朝向 +Z 方向
    - yaw 正方向朝 +X 旋转（绕 Y 轴逆时针）
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """将任意角度归一化到 [-π, π] 范围。

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
        shape=(3,) 的前向单位向量。yaw=0 → (0, 0, 1)。
    """
    return np.array([np.sin(yaw), 0.0, np.cos(yaw)])


def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """计算从源位置看向目标位置所需的 yaw 朝向角。

    参数：
        source_pos: 源位置，shape=(3,)。
        target_pos: 目标位置，shape=(3,)。

    返回：
        yaw 角（弧度制）。
    """
    dx = target_pos[0] - source_pos[0]
    dz = target_pos[2] - source_pos[2]
    return float(np.arctan2(dx, dz))


def unit_direction(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    """计算从 from_pos 指向 to_pos 的单位方向向量。

    v4.0 新增：用于 ray casting 的射线方向。

    参数：
        from_pos: 起点位置，shape=(3,)。
        to_pos: 终点位置，shape=(3,)。

    返回：
        shape=(3,) 的单位方向向量；两点重合时返回默认 +Z 方向。
    """
    vec = np.array(to_pos, dtype=np.float64) - np.array(from_pos, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def compute_vfov_deg(width: int, height: int, hfov_deg: float) -> float:
    """根据 pinhole 相机模型从 HFOV 与图像分辨率推导垂直视场角 VFOV。

    v4.0 相机模型统一：
        - width / height / hfov_deg 是唯一基础相机参数
        - vfov 由 pinhole 模型自动推导，不再手动固定

    公式：
        vfov_rad = 2 * arctan( tan(hfov_rad / 2) * height / width )

    例如 640×480、HFOV=90° → VFOV ≈ 73.74°。

    参数：
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        hfov_deg: 水平视场角（度）。

    返回：
        垂直视场角（度）。
    """
    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) * height / width)
    return float(np.rad2deg(vfov_rad))


def compute_camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    """统一相机内参计算（方形像素假设 fx == fy）。

    v4.0：预测模型、Habitat 真实渲染、depth 投影必须使用同一套内参，
    避免各文件手写不同版本的 fx 导致模型不一致。

    参数：
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        hfov_deg: 水平视场角（度）。

    返回：
        {
            "fx": float, "fy": float, "cx": float, "cy": float,
            "width": int, "height": int,
            "hfov_deg": float, "vfov_deg": float,
        }
    """
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # 方形像素
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
    """判断三维点是否在相机视场角（FOV）内。

    综合考虑水平角度、垂直角度和深度距离三个维度。
    VFOV 通过 compute_camera_intrinsics 从 width/height/hfov 统一推导，
    保证与 Habitat 渲染及 depth 投影使用同一相机模型。

    参数：
        camera_base_pos: 机器人基座位置，shape=(3,)。
        camera_yaw: 相机朝向角（弧度）。
        point: 待判断的三维点，shape=(3,)。
        hfov_deg: 水平视场角（度）。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        camera_height: 相机安装高度（米）。
        min_depth: 最小有效深度（米）。
        max_depth: 最大有效深度（米）。

    返回：
        字典：{"in_fov", "horizontal_angle", "vertical_angle", "distance", "depth"}
    """
    # 统一相机内参：VFOV 由 pinhole 模型推导
    intrinsics = compute_camera_intrinsics(width, height, hfov_deg)
    vfov_deg = intrinsics["vfov_deg"]

    camera_pos = camera_base_pos + np.array([0.0, camera_height, 0.0])
    vec = point - camera_pos

    horizontal_dist = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
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
    """计算高斯形状评分，范围 (0, 1]。

    公式：score = exp(-(value - optimal)² / (2σ²))

    参数：
        value: 输入值。
        optimal: 最优值（高斯中心）。
        sigma: 标准差。

    返回：
        [0, 1] 范围的评分。
    """
    return float(np.exp(-((value - optimal) ** 2) / (2 * sigma ** 2)))