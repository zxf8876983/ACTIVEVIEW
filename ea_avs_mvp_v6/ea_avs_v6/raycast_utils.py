"""
射线工具模块 —— raycast_utils.py
==================================

功能：
    封装 Habitat 场景几何射线检测，用于判断目标关键点是否被环境物体或人体自身遮挡。
"""

from typing import Dict, Optional, Set
import numpy as np

from .geometry import unit_direction


def cast_ray_to_point(
    runner,
    ray_origin: np.ndarray,
    target_point: np.ndarray,
    ray_epsilon: float = 0.05,
    target_tolerance: float = 0.08,
    min_hit_distance: float = 0.05,
    humanoid_object_ids: Optional[Set[int]] = None,
    target_link_object_ids: Optional[Set[int]] = None,
) -> dict:
    """从 ray_origin 向 target_point 发射射线，执行 5 态遮挡分类。"""
    origin = np.array(ray_origin, dtype=np.float64)
    target = np.array(target_point, dtype=np.float64)

    target_distance = float(np.linalg.norm(target - origin))
    humanoid_ids = set(humanoid_object_ids) if humanoid_object_ids else set()
    target_ids = set(target_link_object_ids) if target_link_object_ids else set()

    if target_distance <= ray_epsilon:
        return {
            "hit": False, "occluded": False, "valid": False,
            "hit_distance": 0.0,
            "target_distance": target_distance, "clearance": 0.0,
            "hit_object_id": None, "hit_is_humanoid": False,
            "hit_is_target_surface": False,
            "occlusion_source": "unknown",
        }

    max_distance = target_distance + max(min_hit_distance, 0.1)
    direction = unit_direction(origin, target)
    result = runner.cast_ray(origin, direction, max_distance)

    if not result["has_hits"]:
        return {
            "hit": False, "occluded": False, "valid": True,
            "hit_distance": float("inf"),
            "target_distance": target_distance, "clearance": 0.0,
            "hit_object_id": None, "hit_is_humanoid": False,
            "hit_is_target_surface": False,
            "occlusion_source": "none",
        }

    hit_distance = float(result["hit_distance"])
    hit_object_id = result.get("hit_object_id")
    hit_is_humanoid = bool(hit_object_id in humanoid_ids)
    hit_is_target_surface = bool(hit_object_id in target_ids)

    if hit_distance < min_hit_distance:
        return {
            "hit": True, "occluded": False, "valid": True,
            "hit_distance": hit_distance, "target_distance": target_distance,
            "clearance": 0.0, "hit_object_id": hit_object_id,
            "hit_is_humanoid": hit_is_humanoid,
            "hit_is_target_surface": hit_is_target_surface,
            "occlusion_source": "none",
        }

    genuinely_occluding = bool(hit_distance < target_distance - target_tolerance)
    clearance = max(target_distance - hit_distance, 0.0) if genuinely_occluding else 0.0

    if not genuinely_occluding:
        occluded = False
        occlusion_source = "none"
        valid = True
    elif hit_is_target_surface:
        occluded = False
        occlusion_source = "target_surface"
        valid = True
    elif hit_is_humanoid:
        occluded = True
        occlusion_source = "humanoid_self"
        valid = True
    elif humanoid_ids:
        occluded = True
        occlusion_source = "environment"
        valid = True
    else:
        occluded = False
        occlusion_source = "unknown"
        valid = False

    return {
        "hit": True, "occluded": occluded, "valid": valid,
        "hit_distance": hit_distance, "target_distance": target_distance,
        "clearance": clearance, "hit_object_id": hit_object_id,
        "hit_is_humanoid": hit_is_humanoid,
        "hit_is_target_surface": hit_is_target_surface,
        "occlusion_source": occlusion_source,
    }


def is_target_occluded(ray_result: dict) -> bool:
    return bool(ray_result.get("occluded", False))
