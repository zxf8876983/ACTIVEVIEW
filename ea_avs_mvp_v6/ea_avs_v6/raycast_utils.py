"""
射线工具模块 —— raycast_utils.py
==================================

功能：
    封装 Habitat 场景几何射线检测。
    明确分离两条路径：
        1. GT-State: object-ID-aware 5 态遮挡分类（target_surface / humanoid_self / environment / none / unknown）
        2. Estimated-State: Static-scene-only 3 态遮挡分类（clear / static_scene_blocked / unknown）
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
    """GT-State 路径：从 ray_origin 向 target_point 发射射线，执行 object-ID-aware 5 态遮挡分类。"""
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


def cast_ray_to_estimated_point(
    runner,
    ray_origin: np.ndarray,
    target_point: np.ndarray,
    ray_epsilon: float = 0.05,
    target_tolerance: float = 0.12,
    min_hit_distance: float = 0.05,
) -> dict:
    """Estimated-State 路径：Static-scene-only 几何光路遮挡检测。

    科研设计：
        - 严禁输入任何 GT 变量（humanoid_object_ids、target_link_object_ids、gt_skeleton、keypoint_meta）。
        - 仅使用静态场景几何（static stage mesh）做 raycast，自动忽略真实 Humanoid 及动态物体。
        - 3 态分类：
            - clear: valid=True, occluded=False, occlusion_source="clear"
            - static_scene_blocked: valid=True, occluded=True, occlusion_source="static_scene_blocked"
            - unknown: valid=False, occluded=False, occlusion_source="unknown"
    """
    origin = np.array(ray_origin, dtype=np.float64)
    target = np.array(target_point, dtype=np.float64)

    target_distance = float(np.linalg.norm(target - origin))

    if target_distance <= ray_epsilon:
        return {
            "hit": False,
            "occluded": False,
            "valid": False,
            "hit_distance": 0.0,
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "unknown",
        }

    max_distance = target_distance + max(min_hit_distance, 0.1)
    direction = unit_direction(origin, target)

    if not hasattr(runner, "cast_ray_static_scene"):
        return {
            "hit": False,
            "occluded": False,
            "valid": False,
            "hit_distance": float("inf"),
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "unknown",
            "failure_reason": "static_scene_raycast_unavailable",
        }

    try:
        result = runner.cast_ray_static_scene(origin, direction, max_distance)
    except Exception as exc:
        return {
            "hit": False,
            "occluded": False,
            "valid": False,
            "hit_distance": float("inf"),
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "unknown",
            "failure_reason": f"static_scene_raycast_error:{type(exc).__name__}",
        }

    if not result.get("has_hits", False):
        return {
            "hit": False,
            "occluded": False,
            "valid": True,
            "hit_distance": float("inf"),
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "clear",
        }

    hit_distance = float(result["hit_distance"])

    if hit_distance < min_hit_distance:
        return {
            "hit": True,
            "occluded": False,
            "valid": True,
            "hit_distance": hit_distance,
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "clear",
        }

    # 目标点自身表面 tolerance band 判定：
    # 若 hit_distance >= target_distance - target_tolerance，视为在目标点邻域表面，不判定为遮挡
    genuinely_blocked = bool(hit_distance < target_distance - target_tolerance)
    clearance = max(target_distance - hit_distance, 0.0) if genuinely_blocked else 0.0

    if not genuinely_blocked:
        return {
            "hit": True,
            "occluded": False,
            "valid": True,
            "hit_distance": hit_distance,
            "target_distance": target_distance,
            "clearance": 0.0,
            "occlusion_source": "clear",
        }
    else:
        return {
            "hit": True,
            "occluded": True,
            "valid": True,
            "hit_distance": hit_distance,
            "target_distance": target_distance,
            "clearance": clearance,
            "occlusion_source": "static_scene_blocked",
        }


def is_target_occluded(ray_result: dict) -> bool:
    return bool(ray_result.get("occluded", False))
