"""
深度采样与 2D→3D 逆投影模块 —— depth_lifter.py
=============================================

功能：
    对 2D 关键点进行稳健邻域深度采样，并基于统一 pinhole 相机内参及当前相机外参
    完成 3D 相机坐标与世界坐标提升（Lifting）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .keypoint_schema import Keypoint2D


@dataclass
class DepthSample:
    """关键点邻域深度采样结果。"""
    valid: bool
    depth_m: Optional[float]
    valid_pixel_count: int
    patch_size: int
    spread: Optional[float]
    reason: Optional[str] = None


@dataclass
class EstimatedJoint3D:
    """3D 关键点估计结果。"""
    name: str
    position_world: Optional[np.ndarray]
    position_camera: Optional[np.ndarray]
    confidence_2d: float
    depth_valid: bool
    observable_3d: bool
    source: str  # "observed_2d" | "derived_2d" | "template_completion" | "missing"
    uncertainty: Optional[float] = None


def sample_depth_at_pixel(
    depth_img: np.ndarray,
    u: float,
    v: float,
    patch_size: int = 5,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    max_spread: float = 0.5,
) -> DepthSample:
    """在指定像素 (u, v) 周围小邻域内稳健采样公制深度。

    参数：
        depth_img: shape=(H, W) 的公制深度图（单位：米）。
        u: 像素横坐标。
        v: 像素纵坐标。
        patch_size: 邻域窗口边长（奇数，如 3, 5）。
        min_depth: 有效深度下限。
        max_depth: 有效深度上限。
        max_spread: 允许的最大局部深度波动（MAD）。

    返回：
        DepthSample 对象。
    """
    if depth_img is None or depth_img.size == 0:
        return DepthSample(False, None, 0, patch_size, None, "no_depth_image")

    if depth_img.ndim == 3:
        depth_img = depth_img[..., 0]

    height, width = depth_img.shape
    ui, vi = int(round(u)), int(round(v))

    if not (0 <= ui < width and 0 <= vi < height):
        return DepthSample(False, None, 0, patch_size, None, "pixel_out_of_bounds")

    half = patch_size // 2
    u_min = max(0, ui - half)
    u_max = min(width, ui + half + 1)
    v_min = max(0, vi - half)
    v_max = min(height, vi + half + 1)

    patch = depth_img[v_min:v_max, u_min:u_max]
    valid_mask = (patch >= min_depth) & (patch <= max_depth)
    valid_depths = patch[valid_mask]

    if valid_depths.size == 0:
        return DepthSample(False, None, 0, patch_size, None, "no_valid_depth_in_patch")

    # 稳健中位数深度
    median_d = float(np.median(valid_depths))
    # 中位数绝对偏差 (MAD) 作为局部波动 spread 指标
    mad = float(np.median(np.abs(valid_depths - median_d)))

    if mad > max_spread:
        return DepthSample(
            False, median_d, int(valid_depths.size), patch_size, mad, "depth_spread_too_large"
        )

    return DepthSample(
        True, median_d, int(valid_depths.size), patch_size, mad, None
    )


def backproject_keypoint_to_world(
    u: float,
    v: float,
    depth: float,
    intrinsics: dict,
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    camera_height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """将单像素 (u, v) 及深度逆投影到相机坐标系与世界坐标系。

    坐标系约定（与 Habitat 及 v5.0 geometry 完全一致）：
        - 相机前向 (yaw) = (sin yaw, 0, cos yaw)
        - 相机右向 = (-cos yaw, 0, sin yaw)
        - 相机上向 = (0, 1, 0)
        - 相机真实位置 = camera_base_pos + (0, camera_height, 0)

    参数：
        u, v: 像素坐标。
        depth: 前向深度 z_cam（米）。
        intrinsics: 相机内参字典 {"fx", "fy", "cx", "cy"}。
        camera_base_pos: 机器人底盘世界坐标。
        camera_yaw: 相机朝向角（弧度）。
        camera_height: 相机安装高度（米）。

    返回：
        (pos_camera, pos_world) 两个 shape=(3,) 的 float64 数组。
    """
    fx = intrinsics["fx"]
    fy = intrinsics.get("fy", fx)
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    z_cam = float(depth)
    x_cam = float((u - cx) * z_cam / fx)
    y_cam = float((cy - v) * z_cam / fy)
    pos_camera = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

    th = camera_yaw
    forward = np.array([np.sin(th), 0.0, np.cos(th)], dtype=np.float64)
    right = np.array([-np.cos(th), 0.0, np.sin(th)], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    camera_pos = np.array(camera_base_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0], dtype=np.float64
    )
    pos_world = camera_pos + x_cam * right + y_cam * up + z_cam * forward

    return pos_camera, pos_world


def lift_2d_keypoints_to_3d(
    keypoints_2d: Dict[str, Keypoint2D],
    depth_img: np.ndarray,
    intrinsics: dict,
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    camera_height: float,
    depth_config: dict,
) -> Dict[str, EstimatedJoint3D]:
    """将全体 2D 关键点字典提升为 3D 关节字典。"""
    patch_size = depth_config.get("depth_patch_size", 5)
    min_depth = depth_config.get("depth_min_m", 0.3)
    max_depth = depth_config.get("depth_max_m", 8.0)
    max_spread = depth_config.get("max_depth_spread_m", 0.5)

    joints_3d: Dict[str, EstimatedJoint3D] = {}

    for name, kpt in keypoints_2d.items():
        if not kpt.detected:
            joints_3d[name] = EstimatedJoint3D(
                name=name,
                position_world=None,
                position_camera=None,
                confidence_2d=kpt.confidence,
                depth_valid=False,
                observable_3d=False,
                source="missing",
                uncertainty=None,
            )
            continue

        sample = sample_depth_at_pixel(
            depth_img,
            u=kpt.u,
            v=kpt.v,
            patch_size=patch_size,
            min_depth=min_depth,
            max_depth=max_depth,
            max_spread=max_spread,
        )

        if sample.valid and sample.depth_m is not None:
            pos_cam, pos_world = backproject_keypoint_to_world(
                u=kpt.u,
                v=kpt.v,
                depth=sample.depth_m,
                intrinsics=intrinsics,
                camera_base_pos=camera_base_pos,
                camera_yaw=camera_yaw,
                camera_height=camera_height,
            )
            joints_3d[name] = EstimatedJoint3D(
                name=name,
                position_world=pos_world,
                position_camera=pos_cam,
                confidence_2d=kpt.confidence,
                depth_valid=True,
                observable_3d=True,
                source=kpt.source,
                uncertainty=sample.spread,
            )
        else:
            joints_3d[name] = EstimatedJoint3D(
                name=name,
                position_world=None,
                position_camera=None,
                confidence_2d=kpt.confidence,
                depth_valid=False,
                observable_3d=False,
                source="missing",
                uncertainty=sample.spread,
            )

    return joints_3d
