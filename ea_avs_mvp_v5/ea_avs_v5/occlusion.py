"""
遮挡分析模块 —— occlusion.py
==============================

功能：
    对整个人体骨架进行环境遮挡分析。
    遍历所有关键点，从相机位置向每个关键点发射射线，判断是否被场景遮挡。

v4.0 核心概念：
    visible_after_occlusion = in_fov AND NOT occluded

ray cast 状态约定：
    成功 → {"occluded": bool, "valid": True, "status": "ok"}
    失败 → {"occluded": False, "valid": False, "status": "raycast_error", "error": ...}
    ⚠ 失败（无法判断）不能等同于"未遮挡"：统计遮挡率时不把失败关键点计入
      分子或分母，避免实验结果被乐观偏置。
"""

import logging
from typing import Dict, List

import numpy as np

from .raycast_utils import cast_ray_to_point

logger = logging.getLogger(__name__)

# 遮挡状态常量
STATUS_OK = "ok"
STATUS_RAYCAST_ERROR = "raycast_error"


def compute_keypoint_occlusion(
    runner,
    view_pos: np.ndarray,
    human_skeleton: Dict[str, np.ndarray],
    camera_height: float,
    config: dict,
    humanoid_object_ids: set = None,
) -> dict:
    """对整个人体骨架进行遮挡分析。

    参数：
        runner: HabitatRunner 实例。
        view_pos: 视角位置（机器人基座位置），shape=(3,)。
        human_skeleton: 世界坐标人体骨架 {关键点名称: (3,) 坐标}。
        camera_height: 相机安装高度（米）。
        config: 配置字典，需要 occlusion 配置段。
        humanoid_object_ids: Humanoid 相关 object id 集合；传入时逐关键点
            记录 occlusion_source（environment / humanoid_self）。

    返回：
        字典 {关键点名称:
            {"occluded": bool, "valid": bool, "status": str,
             "hit_distance": float, "target_distance": float,
             "occlusion_source": str, "hit_object_id": int|None, ...}}
        - status="ok"：ray cast 成功，valid=True
        - status="raycast_error"：ray cast 失败，valid=False，occluded=False
        （无法判断 ≠ 未遮挡，统计时排除）
        - occlusion_source：v5.0 区分 environment / humanoid_self（仅当传入
          humanoid_object_ids 且命中发生在目标前才有效；否则为环境）
        射线起点为真实相机位置：view_pos + [0, camera_height, 0]
    """
    occ_cfg = config.get("occlusion", {})
    enabled = occ_cfg.get("enabled", True)
    ray_epsilon = occ_cfg.get("ray_epsilon", 0.05)
    target_tolerance = occ_cfg.get("target_tolerance", 0.08)
    min_hit_distance = occ_cfg.get("min_hit_distance", 0.05)
    humanoid_ids = set(humanoid_object_ids) if humanoid_object_ids else None

    camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0]
    )

    occlusion_result: Dict[str, dict] = {}

    for name, keypoint_pos in human_skeleton.items():
        target_distance = float(np.linalg.norm(
            np.array(keypoint_pos) - camera_pos))

        # 遮挡关闭时：所有关键点视为未遮挡（valid=True，status=ok）
        if not enabled:
            occlusion_result[name] = {
                "occluded": False,
                "valid": True,
                "status": STATUS_OK,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "occlusion_source": "none",
                "hit_object_id": None,
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
            )
            occlusion_result[name] = {
                "occluded": ray_result["occluded"],
                "valid": True,
                "status": STATUS_OK,
                "hit_distance": ray_result["hit_distance"],
                "target_distance": ray_result["target_distance"],
                "occlusion_source": ray_result["occlusion_source"],
                "hit_object_id": ray_result["hit_object_id"],
                "hit_is_humanoid": ray_result["hit_is_humanoid"],
            }
        except Exception as e:  # 单个关键点失败不中断整个骨架分析
            logger.warning(
                "ray cast 失败（关键点 %s）: %s —— 状态标记为 raycast_error",
                name, e,
            )
            occlusion_result[name] = {
                "occluded": False,
                "valid": False,
                "status": STATUS_RAYCAST_ERROR,
                "hit_distance": float("inf"),
                "target_distance": target_distance,
                "occlusion_source": "unknown",
                "hit_object_id": None,
                "error": str(e),
            }

    return occlusion_result


def compute_occlusion_stats(
    occlusion_result: dict,
    keypoint_names: List[str],
) -> dict:
    """计算遮挡统计（只统计 ray cast 成功的关键点，失败不当作未遮挡）。

    v5.0 新增：按遮挡来源拆分 environment / humanoid_self 计数。

    返回：
        {
            "occlusion_rate": float,
            "occlusion_valid_keypoint_count": int,
            "occluded_valid_keypoint_count": int,
            "raycast_error_count": int,
            "raycast_error_rate": float,
            "environment_occluded_keypoint_count": int,
            "self_occluded_keypoint_count": int,
            "unknown_occlusion_keypoint_count": int,
        }
    """
    total = len(keypoint_names)
    if total == 0:
        return {
            "occlusion_rate": 0.0,
            "occlusion_valid_keypoint_count": 0,
            "occluded_valid_keypoint_count": 0,
            "raycast_error_count": 0,
            "raycast_error_rate": 0.0,
            "environment_occluded_keypoint_count": 0,
            "self_occluded_keypoint_count": 0,
            "unknown_occlusion_keypoint_count": 0,
        }

    valid_count = 0
    occluded_valid_count = 0
    raycast_error_count = 0
    env_count = 0
    self_count = 0
    unknown_occ_count = 0

    for name in keypoint_names:
        info = occlusion_result.get(name, {})
        if not info.get("valid", True):
            raycast_error_count += 1
            continue
        valid_count += 1
        if info.get("occluded", False):
            occluded_valid_count += 1
            occ_source = info.get("occlusion_source", "environment")
            if occ_source == "humanoid_self":
                self_count += 1
            elif occ_source == "unknown":
                unknown_occ_count += 1
            else:
                env_count += 1

    occlusion_rate = (occluded_valid_count / valid_count) if valid_count > 0 else 0.0
    raycast_error_rate = raycast_error_count / total

    return {
        "occlusion_rate": float(occlusion_rate),
        "occlusion_valid_keypoint_count": valid_count,
        "occluded_valid_keypoint_count": occluded_valid_count,
        "raycast_error_count": raycast_error_count,
        "raycast_error_rate": float(raycast_error_rate),
        "environment_occluded_keypoint_count": env_count,
        "self_occluded_keypoint_count": self_count,
        "unknown_occlusion_keypoint_count": unknown_occ_count,
    }


def compute_occlusion_rate(
    occlusion_result: dict,
    keypoint_names: List[str],
) -> float:
    """计算关键点遮挡率（只统计 ray cast 成功的关键点）。

    定义：
        occlusion_rate = 被环境遮挡且 ray cast 成功的关键点数 /
                          ray cast 成功的关键点总数

    参数：
        occlusion_result: compute_keypoint_occlusion 的返回结果。
        keypoint_names: 需要统计的关键点名称列表。

    返回：
        [0, 1] 范围的遮挡率；无有效关键点时返回 0.0。
    """
    stats = compute_occlusion_stats(occlusion_result, keypoint_names)
    return stats["occlusion_rate"]