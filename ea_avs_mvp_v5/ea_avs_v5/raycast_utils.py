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
    humanoid_object_ids: set = None,
) -> dict:
    """从 ray_origin 向 target_point 发射射线，判断目标是否被遮挡。

    参数：
        runner: HabitatRunner 实例（提供 cast_ray 统一接口）。
        ray_origin: 射线起点（真实相机位置），shape=(3,)。
        target_point: 目标点（关键点世界坐标），shape=(3,)。
        ray_epsilon: 起点/目标的最小有效距离，低于该值视为退化，不判遮挡。
        target_tolerance: 判定"目标前命中"的距离容差（米）。
        min_hit_distance: 忽略起点附近（小于该距离）的初始命中。
        humanoid_object_ids: Humanoid 相关 object id 集合（用于区分
            environment / humanoid_self 遮挡来源）；None 时统一记为
            environment。

    返回：
        {
            "hit": bool,
            "occluded": bool,
            "hit_distance": float,
            "target_distance": float,
            "clearance": float,
            "hit_object_id": int|None,
            "occlusion_source": "environment"|"humanoid_self"|"none"|"unknown",
            "hit_is_humanoid": bool,
        }
        occlusion_source：
            - "none"：未命中或被忽略的起点附近命中
            - "environment"：命中发生在目标前，且命中物体非 Humanoid
            - "humanoid_self"：命中发生在目标前，且命中物体为 Humanoid
              （人体其他部位提前遮挡，即 self-occlusion）
            - "unknown"：目标距离退化或数据异常
    """
    origin = np.array(ray_origin, dtype=np.float64)
    target = np.array(target_point, dtype=np.float64)

    target_distance = float(np.linalg.norm(target - origin))
    humanoid_ids = set(humanoid_object_ids) if humanoid_object_ids else set()

    # 退化情形：起点与目标几乎重合，无法做有意义的遮挡判断
    if target_distance <= ray_epsilon:
        return {
            "hit": False, "occluded": False, "hit_distance": 0.0,
            "target_distance": target_distance, "clearance": 0.0,
            "hit_object_id": None,
            "occlusion_source": "unknown", "hit_is_humanoid": False,
        }

    # 射程比目标距离略长，确保能命中目标点附近的场景
    max_distance = target_distance + max(min_hit_distance, 0.1)

    direction = unit_direction(origin, target)
    result = runner.cast_ray(origin, direction, max_distance)

    if not result["has_hits"]:
        return {
            "hit": False, "occluded": False, "hit_distance": float("inf"),
            "target_distance": target_distance, "clearance": 0.0,
            "hit_object_id": None,
            "occlusion_source": "none", "hit_is_humanoid": False,
        }

    hit_distance = float(result["hit_distance"])
    hit_object_id = result.get("hit_object_id")
    hit_is_humanoid = bool(hit_object_id in humanoid_ids)

    # 忽略起点附近的初始命中
    if hit_distance < min_hit_distance:
        return {
            "hit": True, "occluded": False,
            "hit_distance": hit_distance, "target_distance": target_distance,
            "clearance": 0.0, "hit_object_id": hit_object_id,
            "occlusion_source": "none", "hit_is_humanoid": hit_is_humanoid,
        }

    # 核心判定：命中发生在目标点之前（含容差）则被遮挡
    occluded = bool(hit_distance < target_distance - target_tolerance)
    clearance = max(target_distance - hit_distance, 0.0) if occluded else 0.0

    if not occluded:
        occlusion_source = "none"
    else:
        occlusion_source = ("humanoid_self" if hit_is_humanoid
                            else "environment")

    return {
        "hit": True, "occluded": occluded,
        "hit_distance": hit_distance, "target_distance": target_distance,
        "clearance": clearance, "hit_object_id": hit_object_id,
        "occlusion_source": occlusion_source,
        "hit_is_humanoid": hit_is_humanoid,
    }


def is_target_occluded(ray_result: dict) -> bool:
    """从 cast_ray_to_point 的返回结果中提取遮挡判定。

    参数：
        ray_result: cast_ray_to_point 返回的字典。

    返回：
        布尔值，True 表示目标被环境遮挡。
    """
    return bool(ray_result.get("occluded", False))