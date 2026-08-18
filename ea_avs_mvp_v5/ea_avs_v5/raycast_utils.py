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
    target_link_object_ids: set = None,
) -> dict:
    """从 ray_origin 向 target_point 发射射线，执行 5 态遮挡分类。

    参数：
        runner: HabitatRunner 实例（提供 cast_ray 统一接口）。
        ray_origin: 射线起点（真实相机位置），shape=(3,)。
        target_point: 目标点（关键点世界坐标），shape=(3,)。
        ray_epsilon: 起点/目标的最小有效距离，低于该值视为退化。
        target_tolerance: 判定"目标前命中"的距离容差（米）。
            仍保留 hit_distance < target_distance - target_tolerance 以排除
            目标点附近命中/数值误差/目标碰撞表面与 joint center 的小距离。
        min_hit_distance: 忽略起点附近（小于该距离）的初始命中。
        humanoid_object_ids: 所有 Humanoid root/link 对应的 Habitat object id
            （用于 humanoid_self 判定）。
        target_link_object_ids: 当前 target keypoint 对应 body part 的 Habitat
            object id（用于 target_surface 判定）。若为 None，则无法区分
            target_surface 与 humanoid_self，此时命中该 Humanoid 一律按
            humanoid_self。

    返回：
        {
            "hit": bool,
            "occluded": bool,
            "valid": bool,        # 遮挡判定是否可信（unknown=False）
            "hit_distance": float,
            "target_distance": float,
            "clearance": float,
            "hit_object_id": int|None,
            "hit_is_humanoid": bool,
            "hit_is_target_surface": bool,
            "occlusion_source": "none"|"target_surface"|"environment"
                                |"humanoid_self"|"unknown",
        }

    分类顺序（5 态）：
        1. 无真正提前 hit              → "none"      , occluded=False, valid=True
        2. 提前 hit 命中 target link   → "target_surface", occluded=False, valid=True
           （目标 body part 本身被看见）
        3. 提前 hit 命中其他 Humanoid link → "humanoid_self", occluded=True, valid=True
        4. 提前 hit 命中环境物体       → "environment", occluded=True, valid=True
        5. object-id 映射无法可靠判定  → "unknown"   , valid=False（occluded=False）
    ⚠ unknown 必须 valid=False：绝不进入 visible，也不进入有效遮挡率分母。
    """
    origin = np.array(ray_origin, dtype=np.float64)
    target = np.array(target_point, dtype=np.float64)

    target_distance = float(np.linalg.norm(target - origin))
    humanoid_ids = set(humanoid_object_ids) if humanoid_object_ids else set()
    target_ids = set(target_link_object_ids) if target_link_object_ids else set()

    # 退化情形：起点与目标几乎重合，无法做有意义的遮挡判断
    if target_distance <= ray_epsilon:
        return {
            "hit": False, "occluded": False, "valid": False,
            "hit_distance": 0.0,
            "target_distance": target_distance, "clearance": 0.0,
            "hit_object_id": None, "hit_is_humanoid": False,
            "hit_is_target_surface": False,
            "occlusion_source": "unknown",
        }

    # 射程比目标距离略长，确保能命中目标点附近的场景
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

    # 忽略起点附近的初始命中
    if hit_distance < min_hit_distance:
        return {
            "hit": True, "occluded": False, "valid": True,
            "hit_distance": hit_distance, "target_distance": target_distance,
            "clearance": 0.0, "hit_object_id": hit_object_id,
            "hit_is_humanoid": hit_is_humanoid,
            "hit_is_target_surface": hit_is_target_surface,
            "occlusion_source": "none",
        }

    # 核心判定：是否发生真正提前 hit（排除目标点附近/精度容差）
    genuinely_occluding = bool(hit_distance < target_distance - target_tolerance)
    clearance = max(target_distance - hit_distance, 0.0) if genuinely_occluding else 0.0

    if not genuinely_occluding:
        occluded = False
        occlusion_source = "none"
        valid = True
    elif hit_is_target_surface:
        # 命中目标 body part 自身表面 → 目标可见，不算 self-occlusion
        occluded = False
        occlusion_source = "target_surface"
        valid = True
    elif hit_is_humanoid:
        # 命中其他 Humanoid link → self-occlusion
        occluded = True
        occlusion_source = "humanoid_self"
        valid = True
    elif humanoid_ids:
        # 明确知道命中物不是 Humanoid → environment
        occluded = True
        occlusion_source = "environment"
        valid = True
    else:
        # 无 humanoid id 信息可判定 → unknown，valid=False
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
    """从 cast_ray_to_point 的返回结果中提取遮挡判定。

    参数：
        ray_result: cast_ray_to_point 返回的字典。

    返回：
        布尔值，True 表示目标被环境遮挡。
    """
    return bool(ray_result.get("occluded", False))