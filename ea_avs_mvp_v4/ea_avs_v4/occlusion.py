"""
遮挡分析模块 —— occlusion.py
==============================

功能：
    对整个人体骨架进行环境遮挡分析。
    遍历所有关键点，从相机位置向每个关键点发射射线，判断是否被场景遮挡。

v4.0 核心概念：
    visible_after_occlusion = in_fov AND NOT occluded
"""

import logging
from typing import Dict, List

import numpy as np

from .raycast_utils import cast_ray_to_point

logger = logging.getLogger(__name__)


def compute_keypoint_occlusion(
    runner,
    view_pos: np.ndarray,
    human_skeleton: Dict[str, np.ndarray],
    camera_height: float,
    config: dict,
) -> dict:
    """对整个人体骨架进行环境遮挡分析。

    参数：
        runner: HabitatRunner 实例。
        view_pos: 视角位置（机器人基座位置），shape=(3,)。
        human_skeleton: 世界坐标人体骨架 {关键点名称: (3,) 坐标}。
        camera_height: 相机安装高度（米）。
        config: 配置字典，需要 occlusion 配置段：
            - enabled: 是否启用遮挡判断
            - ray_epsilon / target_tolerance / min_hit_distance

    返回：
        字典 {关键点名称:
            {"occluded": bool, "hit_distance": float, "target_distance": float}}
        射线起点为真实相机位置：view_pos + [0, camera_height, 0]
    """
    occ_cfg = config.get("occlusion", {})
    enabled = occ_cfg.get("enabled", True)
    ray_epsilon = occ_cfg.get("ray_epsilon", 0.05)
    target_tolerance = occ_cfg.get("target_tolerance", 0.08)
    min_hit_distance = occ_cfg.get("min_hit_distance", 0.05)

    camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0]
    )

    occlusion_result: Dict[str, dict] = {}

    for name, keypoint_pos in human_skeleton.items():
        # 遮挡关闭时：所有关键点视为未遮挡
        if not enabled:
            occlusion_result[name] = {
                "occluded": False,
                "hit_distance": float("inf"),
                "target_distance": float(np.linalg.norm(
                    np.array(keypoint_pos) - camera_pos)),
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
            )
            occlusion_result[name] = {
                "occluded": ray_result["occluded"],
                "hit_distance": ray_result["hit_distance"],
                "target_distance": ray_result["target_distance"],
            }
        except Exception as e:  # 单个关键点失败不中断整个骨架分析
            # ⚠ 记录 WARNING，避免 ray casting 全部失败时静默退化为"无遮挡"
            logger.warning(
                "ray cast 失败（关键点 %s）: %s —— 该关键点按未遮挡处理",
                name, e,
            )
            occlusion_result[name] = {
                "occluded": False,
                "hit_distance": float("inf"),
                "target_distance": float(np.linalg.norm(
                    np.array(keypoint_pos) - camera_pos)),
                "error": str(e),
            }

    return occlusion_result


def compute_occlusion_rate(
    occlusion_result: dict,
    keypoint_names: List[str],
) -> float:
    """计算关键点遮挡率。

    定义：
        occlusion_rate = 被环境遮挡关键点数 / 目标关键点总数

    参数：
        occlusion_result: compute_keypoint_occlusion 的返回结果。
        keypoint_names: 需要统计的关键点名称列表。

    返回：
        [0, 1] 范围的遮挡率；空列表时返回 0.0。
    """
    if len(keypoint_names) == 0:
        return 0.0

    occluded_count = 0
    for name in keypoint_names:
        info = occlusion_result.get(name)
        if info is not None and info.get("occluded", False):
            occluded_count += 1

    return occluded_count / len(keypoint_names)