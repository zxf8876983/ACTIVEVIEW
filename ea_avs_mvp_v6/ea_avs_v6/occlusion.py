"""
遮挡分析模块 —— occlusion.py
==============================

功能：
    对整个人体骨架进行环境与自身遮挡分析，统计遮挡率与有效性。
    支持两条独立路径：
        1. GT-State: compute_keypoint_occlusion (5 态分类，含 object-ID 辨识)
        2. Estimated-State: compute_estimated_keypoint_occlusion (Static-scene-only 3 态几何遮挡)
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np

from .raycast_utils import cast_ray_to_point, cast_ray_to_estimated_point

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_RAYCAST_ERROR = "raycast_error"


def compute_keypoint_occlusion(
    runner,
    view_pos: np.ndarray,
    human_skeleton: Dict[str, np.ndarray],
    camera_height: float,
    config: dict,
    humanoid_object_ids: Optional[Set[int]] = None,
    keypoint_meta: Optional[dict] = None,
) -> dict:
    """GT-State 路径：对整个人体骨架进行 object-ID-aware 遮挡分析（5 态分类）。"""
    occ_cfg = config.get("occlusion", {})
    enabled = occ_cfg.get("enabled", True)
    ray_epsilon = occ_cfg.get("ray_epsilon", 0.05)
    target_tolerance = occ_cfg.get("target_tolerance", 0.08)
    min_hit_distance = occ_cfg.get("min_hit_distance", 0.05)
    humanoid_ids = set(humanoid_object_ids) if humanoid_object_ids else None
    meta = keypoint_meta or {}

    camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0], dtype=np.float64
    )

    occlusion_result: Dict[str, dict] = {}

    for name, keypoint_pos in human_skeleton.items():
        target_distance = float(np.linalg.norm(np.array(keypoint_pos) - camera_pos))
        tm = meta.get(name, {})
        target_ids = tm.get("target_link_object_ids") or None

        if not enabled:
            occlusion_result[name] = {
                "occluded": False,
                "valid": True,
                "status": STATUS_OK,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "hit_object_id": None,
                "hit_is_humanoid": False,
                "hit_is_target_surface": False,
                "occlusion_cause": "none",
            }
            continue

        try:
            ray_result = cast_ray_to_point(
                runner=runner,
                ray_origin=camera_pos,
                target_point=keypoint_pos,
                ray_epsilon=ray_epsilon,
                target_tolerance=target_tolerance,
                min_hit_distance=min_hit_distance,
                humanoid_object_ids=humanoid_ids,
                target_link_object_ids=target_ids,
            )
            occlusion_result[name] = {
                "occluded": ray_result["occluded"],
                "valid": ray_result["valid"],
                "status": STATUS_OK,
                "hit_distance": ray_result["hit_distance"],
                "target_distance": ray_result["target_distance"],
                "hit_object_id": ray_result["hit_object_id"],
                "hit_is_humanoid": ray_result["hit_is_humanoid"],
                "hit_is_target_surface": ray_result["hit_is_target_surface"],
                "occlusion_cause": ray_result["occlusion_source"],
            }
        except Exception as e:
            logger.warning(
                "GT ray cast 失败（关键点 %s）: %s —— 状态标记为 raycast_error",
                name, e,
            )
            occlusion_result[name] = {
                "occluded": False,
                "valid": False,
                "status": STATUS_RAYCAST_ERROR,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "hit_object_id": None,
                "hit_is_humanoid": False,
                "hit_is_target_surface": False,
                "occlusion_cause": "unknown",
                "error": str(e),
            }

    return occlusion_result


def compute_estimated_keypoint_occlusion(
    runner,
    view_pos: np.ndarray,
    proxy_skeleton: Dict[str, np.ndarray],
    camera_height: float,
    config: dict,
) -> dict:
    """Estimated-State 路径：Static-scene-only 几何光路遮挡检测。

    严禁传入任何 humanoid_object_ids 或 keypoint_meta。
    仅使用静态场景几何（static stage mesh）进行光路遮挡判定。
    """
    occ_cfg = config.get("occlusion", {})
    enabled = occ_cfg.get("enabled", True)
    ray_epsilon = occ_cfg.get("ray_epsilon", 0.05)
    target_tolerance = occ_cfg.get("estimated_target_tolerance", occ_cfg.get("target_tolerance", 0.12))
    min_hit_distance = occ_cfg.get("min_hit_distance", 0.05)

    camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0], dtype=np.float64
    )

    occlusion_result: Dict[str, dict] = {}

    for name, keypoint_pos in proxy_skeleton.items():
        target_distance = float(np.linalg.norm(np.array(keypoint_pos) - camera_pos))

        if not enabled:
            occlusion_result[name] = {
                "occluded": False,
                "valid": True,
                "status": STATUS_OK,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "occlusion_cause": "clear",
            }
            continue

        try:
            ray_result = cast_ray_to_estimated_point(
                runner=runner,
                ray_origin=camera_pos,
                target_point=keypoint_pos,
                ray_epsilon=ray_epsilon,
                target_tolerance=target_tolerance,
                min_hit_distance=min_hit_distance,
            )
            occlusion_result[name] = {
                "occluded": ray_result["occluded"],
                "valid": ray_result["valid"],
                "status": STATUS_OK if ray_result["valid"] else STATUS_RAYCAST_ERROR,
                "hit_distance": ray_result["hit_distance"],
                "target_distance": ray_result["target_distance"],
                "occlusion_cause": ray_result["occlusion_source"],
            }
        except Exception as e:
            logger.warning(
                "Estimated ray cast 失败（关键点 %s）: %s",
                name, e,
            )
            occlusion_result[name] = {
                "occluded": False,
                "valid": False,
                "status": STATUS_RAYCAST_ERROR,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "occlusion_cause": "unknown",
                "error": str(e),
            }

    return occlusion_result


def compute_occlusion_stats(
    occlusion_result: dict,
    keypoint_names: List[str],
    cause_key: str = "occlusion_cause",
) -> dict:
    """GT-State 遮挡统计指标（含 5 态细分）。"""
    total = len(keypoint_names)
    empty = {
        "occlusion_rate": 0.0,
        "occlusion_valid_keypoint_count": 0,
        "occluded_valid_keypoint_count": 0,
        "raycast_error_count": 0,
        "raycast_error_rate": 0.0,
        "target_surface_keypoint_count": 0,
        "environment_occluded_keypoint_count": 0,
        "self_occluded_keypoint_count": 0,
        "unknown_occlusion_keypoint_count": 0,
    }
    if total == 0:
        return empty

    valid_count = 0
    occluded_valid_count = 0
    raycast_error_count = 0
    target_surface_count = 0
    env_count = 0
    self_count = 0
    unknown_occ_count = 0

    for name in keypoint_names:
        info = occlusion_result.get(name, {})
        cause = info.get(cause_key) or info.get("occlusion_source") or "unknown"
        valid = bool(info.get("valid", True))

        if cause == "unknown":
            unknown_occ_count += 1
            if valid:
                continue
        if not valid:
            if info.get("status") == STATUS_RAYCAST_ERROR:
                raycast_error_count += 1
            continue
        valid_count += 1
        if cause == "target_surface":
            target_surface_count += 1
            continue
        if info.get("occluded", False):
            occluded_valid_count += 1
            if cause == "humanoid_self":
                self_count += 1
            elif cause == "environment":
                env_count += 1

    occlusion_rate = (occluded_valid_count / valid_count) if valid_count > 0 else 0.0
    raycast_error_rate = raycast_error_count / total

    return {
        "occlusion_rate": float(occlusion_rate),
        "occlusion_valid_keypoint_count": valid_count,
        "occluded_valid_keypoint_count": occluded_valid_count,
        "raycast_error_count": raycast_error_count,
        "raycast_error_rate": float(raycast_error_rate),
        "target_surface_keypoint_count": target_surface_count,
        "environment_occluded_keypoint_count": env_count,
        "self_occluded_keypoint_count": self_count,
        "unknown_occlusion_keypoint_count": unknown_occ_count,
    }


def compute_estimated_occlusion_stats(
    occlusion_result: dict,
    keypoint_names: List[str],
) -> dict:
    """Estimated-State 遮挡统计指标（Static-scene-only，统计 clear/static_scene_blocked/unknown）。"""
    total = len(keypoint_names)
    empty = {
        "occlusion_rate": 0.0,
        "occlusion_valid_keypoint_count": 0,
        "occluded_valid_keypoint_count": 0,
        "raycast_error_count": 0,
        "raycast_error_rate": 0.0,
        "estimated_clear_keypoint_count": 0,
        "estimated_static_blocked_keypoint_count": 0,
        "estimated_blocked_keypoint_count": 0,
        "estimated_unknown_keypoint_count": 0,
        "self_occluded_keypoint_count": 0,
    }
    if total == 0:
        return empty

    valid_count = 0
    occluded_valid_count = 0
    raycast_error_count = 0
    clear_count = 0
    blocked_count = 0
    unknown_count = 0

    for name in keypoint_names:
        info = occlusion_result.get(name, {})
        cause = info.get("occlusion_cause") or info.get("occlusion_source") or "unknown"
        valid = bool(info.get("valid", True))

        if not valid or cause == "unknown":
            unknown_count += 1
            if info.get("status") == STATUS_RAYCAST_ERROR:
                raycast_error_count += 1
            continue

        valid_count += 1
        if info.get("occluded", False) or cause in ("static_scene_blocked", "estimated_geometric_blocked"):
            occluded_valid_count += 1
            blocked_count += 1
        else:
            clear_count += 1

    occlusion_rate = (occluded_valid_count / valid_count) if valid_count > 0 else 0.0
    raycast_error_rate = raycast_error_count / total

    return {
        "occlusion_rate": float(occlusion_rate),
        "occlusion_valid_keypoint_count": valid_count,
        "occluded_valid_keypoint_count": occluded_valid_count,
        "raycast_error_count": raycast_error_count,
        "raycast_error_rate": float(raycast_error_rate),
        "estimated_clear_keypoint_count": clear_count,
        "estimated_static_blocked_keypoint_count": blocked_count,
        "estimated_blocked_keypoint_count": blocked_count,
        "estimated_unknown_keypoint_count": unknown_count,
        "self_occluded_keypoint_count": 0,
    }


def compute_occlusion_rate(
    occlusion_result: dict,
    keypoint_names: List[str],
) -> float:
    stats = compute_occlusion_stats(occlusion_result, keypoint_names)
    return stats["occlusion_rate"]
