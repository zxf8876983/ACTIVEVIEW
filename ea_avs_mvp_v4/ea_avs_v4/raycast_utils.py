"""
射线工具模块 —— raycast_utils.py
==================================

功能：
    封装 Habitat 场景几何射线检测，用于判断目标关键点是否被环境物体遮挡。

v4.0 核心原理：
    in_fov ≠ visible
    visible = in_fov AND NOT occluded

判断逻辑：
    target_distance = ||target_point - ray_origin||
    如果射线在目标点之前击中场景 mesh：
        hit_distance < target_distance - target_tolerance
    则： occluded = True
    否则：occluded = False

约束：
    - 射线起点必须从真实相机位置发出（view_pos + camera_height）
    - Habitat 版本 API 差异已经封装在 habitat_runner.cast_ray() 中
"""

from typing import Dict

import numpy as np

from .geometry import unit_direction


def cast_ray_to_point(
    runner,
    ray_origin: np.ndarray,
    target_point: np.ndarray,
    ray_epsilon: float = 0.05,
    target_tolerance: float = 0.08,
    min_hit_distance: float = 0.05,
) -> dict:
    """从 ray_origin 向 target_point 发射射线，判断目标是否被环境遮挡。

    参数：
        runner: HabitatRunner 实例（提供 cast_ray 统一接口）。
        ray_origin: 射线起点（真实相机位置），shape=(3,)。
        target_point: 目标点（关键点世界坐标），shape=(3,)。
        ray_epsilon: 起点/目标的最小有效距离，低于该值视为退化，不判遮挡。
        target_tolerance: 判定"目标前命中"的距离容差（米）。
            命中距离小于 (target_distance - tolerance) 才判定为遮挡，
            避免目标点本身贴近墙面时被误判。
        min_hit_distance: 忽略起点附近（小于该距离）的初始命中，防止起点
            与场景穿插造成的误判。

    返回：
        {
            "hit": bool,           # 射线在射程内是否命中场景
            "occluded": bool,      # 命中是否发生在目标点之前（被遮挡）
            "hit_distance": float, # 最近命中距离（世界单位，米）；未命中为 inf
            "target_distance": float, # 起点到目标点的距离（米）
            "clearance": float,    # 目标距离 - 命中距离（被遮挡时为正，否则 0）
        }
    """
    origin = np.array(ray_origin, dtype=np.float64)
    target = np.array(target_point, dtype=np.float64)

    target_distance = float(np.linalg.norm(target - origin))

    # 退化情形：起点与目标几乎重合，无法做有意义的遮挡判断
    if target_distance <= ray_epsilon:
        return {
            "hit": False,
            "occluded": False,
            "hit_distance": 0.0,
            "target_distance": target_distance,
            "clearance": 0.0,
        }

    # 射程比目标距离略长，确保能命中目标点附近的场景
    max_distance = target_distance + max(min_hit_distance, 0.1)

    direction = unit_direction(origin, target)
    result = runner.cast_ray(origin, direction, max_distance)

    if not result["has_hits"]:
        return {
            "hit": False,
            "occluded": False,
            "hit_distance": float("inf"),
            "target_distance": target_distance,
            "clearance": 0.0,
        }

    hit_distance = float(result["hit_distance"])

    # 忽略起点附近的初始命中（起点与场景穿插等数值问题）
    if hit_distance < min_hit_distance:
        return {
            "hit": True,
            "occluded": False,
            "hit_distance": hit_distance,
            "target_distance": target_distance,
            "clearance": 0.0,
        }

    # 核心判定：命中发生在目标点之前（含容差）则被遮挡
    occluded = bool(hit_distance < target_distance - target_tolerance)
    clearance = max(target_distance - hit_distance, 0.0) if occluded else 0.0

    return {
        "hit": True,
        "occluded": occluded,
        "hit_distance": hit_distance,
        "target_distance": target_distance,
        "clearance": clearance,
    }


def is_target_occluded(ray_result: dict) -> bool:
    """从 cast_ray_to_point 的返回结果中提取遮挡判定。

    参数：
        ray_result: cast_ray_to_point 返回的字典。

    返回：
        布尔值，True 表示目标被环境遮挡。
    """
    return bool(ray_result.get("occluded", False))