"""
Humanoid 骨架适配器 —— humanoid_skeleton_adapter.py
=====================================================

功能：
    将 Humanoid 实际关节/链接状态转换为 v4.0 evaluator 能使用的关键点字典。

输出形式（与 v4.0 完全一致，15 个关键点）：
    {
        "head": [x,y,z], "neck": [...], "pelvis": [...],
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    }

关键要求（v5.0 严谨性）：
    - 正式模式（strict_gt_skeleton=true）：15/15 关键点必须全部来自 Humanoid
      links（世界坐标），source=="habitat_humanoid_gt"，否则报错。
    - legacy fallback 仅保留在 debug 模式，且 legacy 局部坐标必须先经
      transform_legacy_local_skeleton_to_world 转换为世界坐标，禁止直接混用。
"""

from typing import Dict

import numpy as np

# 从 Humanoid 的 URDF link 名到 v4.0 关键点的映射
LINK_TO_KEYPOINT = {
    "head": "head",
    "neck": "neck",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    # pelvis：URDF 中通常无几何，由左右髋中点近似（link_midpoint_hips）
}

ALL_KEYPOINTS = (
    "head", "neck", "pelvis",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _link_world_position(humanoid_manager, link_name: str):
    """从 Humanoid scene 节点获取指定 link 的世界坐标。"""
    ao = humanoid_manager.sim_obj
    try:
        link_id = ao.get_link_id_from_name(link_name)
    except Exception:
        return None
    if link_id is None or int(link_id) < 0:
        return None
    try:
        node = ao.get_link_scene_node(link_id)
    except Exception:
        return None
    pos = np.array(node.transformation.translation, dtype=np.float64).copy()
    return pos


def transform_legacy_local_skeleton_to_world(
    local_skeleton: Dict[str, np.ndarray],
    human_base_pos: np.ndarray,
    human_yaw: float,
) -> Dict[str, np.ndarray]:
    """将 legacy 局部坐标骨架转换为世界坐标（debug fallback 使用）。

    不能直接把 legacy 局部坐标当作世界坐标：先按实际 human_yaw 绕 Y 轴旋转，
    再平移到 Humanoid base 位置。

    参数：
        local_skeleton: {关键点: 局部坐标}（原点在人体脚底中心，正面 +Z）。
        human_base_pos: Humanoid 脚底中心世界坐标，shape=(3,)。
        human_yaw: 人体朝向（弧度）。

    返回：
        世界坐标骨架字典。
    """
    cos_y, sin_y = np.cos(human_yaw), np.sin(human_yaw)
    base = np.asarray(human_base_pos, dtype=np.float64)
    world = {}
    for kp, local in local_skeleton.items():
        l = np.asarray(local, dtype=np.float64)
        # RotY(human_yaw)
        x = l[0] * cos_y + l[2] * sin_y
        y = l[1]
        z = -l[0] * sin_y + l[2] * cos_y
        world[kp] = base + np.array([x, y, z])
    return world


def get_humanoid_gt_skeleton(
    humanoid_manager,
    fallback_skeleton: Dict[str, np.ndarray] = None,
    human_base_pos: np.ndarray = None,
    human_yaw: float = 0.0,
    strict: bool = True,
) -> dict:
    """从 Humanoid 实际关节/链接状态构建 GT skeleton（含 keypoint_meta）。

    参数：
        humanoid_manager: HumanoidManager 实例（须已 load）。
        fallback_skeleton: 可选 legacy 局部坐标骨架（仅 debug 模式，且会
            被转换为世界坐标后使用）。
        human_base_pos: Humanoid 脚底中心世界坐标（用于 legacy 转换）。
        human_yaw: Humanoid 实际朝向（用于 legacy 旋转）。
        strict: 严格模式。True 时要求 15 个 Humanoid-derived keypoints 且
            fallback=0。

    返回：
        {
            "skeleton": {关键点: 世界坐标},
            "keypoint_meta": {
                关键点: {
                    "source": "link" | "link_derived" | "legacy",
                    "link_name": str|None,
                    "link_id": int|None,        # 本地 link_id
                    "link_object_id": int|None, # Habitat object_id（可匹配 hit）
                    "target_link_object_ids": list[int],  # 用于 self-occlusion
                },
            },
            "source": "habitat_humanoid_gt" | "mixed" | "legacy_fallback",
            "per_keypoint_source": {关键点: "link"|"link_derived"|"legacy"},
            "keypoint_count": int,
            "direct_link_count": int,
            "link_derived_count": int,
            "fallback_count": int,
            "missing_keypoints": list,
        }

    抛出异常：
        RuntimeError: strict=True 且不满足 15 个 Humanoid-derived keypoints。
    """
    skeleton: Dict[str, np.ndarray] = {}
    per_source: Dict[str, str] = {}
    keypoint_meta: Dict[str, dict] = {}

    # link 元数据（link_name -> {link_id, object_id}），用于 self-occlusion 判定
    link_meta = humanoid_manager.get_humanoid_link_metadata()

    def _object_id_for(link_name):
        m = link_meta.get(link_name)
        return m["object_id"] if m else None

    # 1) 优先从 link transforms 取直接 link 关键点
    for link_name, kp in LINK_TO_KEYPOINT.items():
        if kp == "pelvis":
            continue
        pos = _link_world_position(humanoid_manager, link_name)
        if pos is not None:
            skeleton[kp] = pos
            per_source[kp] = "link"
            m = link_meta.get(link_name, {})
            keypoint_meta[kp] = {
                "source": "link",
                "link_name": link_name,
                "link_id": m.get("link_id"),
                "link_object_id": m.get("object_id"),
                "target_link_object_ids": [m["object_id"]] if m.get("object_id") is not None else [],
            }

    # 2) pelvis：由左右髋中点推导（Humanoid-derived point，非直接 link）
    if "left_hip" in skeleton and "right_hip" in skeleton:
        skeleton["pelvis"] = 0.5 * (skeleton["left_hip"] + skeleton["right_hip"])
        per_source["pelvis"] = "link_derived"
        hip_oids = [oid for oid in (
            _object_id_for("left_hip"), _object_id_for("right_hip")) if oid is not None]
        keypoint_meta["pelvis"] = {
            "source": "link_derived",
            "link_name": None,
            "link_id": None,
            "link_object_id": None,
            "target_link_object_ids": hip_oids,
        }

    direct_link_count = sum(1 for s in per_source.values() if s == "link")
    link_derived_count = sum(1 for s in per_source.values() if s == "link_derived")

    # 3) legacy fallback（debug 模式）：转换为世界坐标后补缺失关键点
    if fallback_skeleton is not None and (not strict or direct_link_count + link_derived_count < 15):
        legacy_world = fallback_skeleton
        if human_base_pos is not None:
            legacy_world = transform_legacy_local_skeleton_to_world(
                fallback_skeleton, human_base_pos, human_yaw)
        for kp in ALL_KEYPOINTS:
            if kp not in skeleton and kp in legacy_world:
                skeleton[kp] = np.asarray(legacy_world[kp], dtype=np.float64)
                per_source[kp] = "legacy"
                keypoint_meta[kp] = {
                    "source": "legacy",
                    "link_name": None,
                    "link_id": None,
                    "link_object_id": None,
                    "target_link_object_ids": [],
                }

    # 4) 来源判定
    total = len(skeleton)
    fallback_count = sum(1 for s in per_source.values() if s == "legacy")
    if total == 0:
        source = "legacy_fallback"
    elif fallback_count == 0 and (direct_link_count + link_derived_count) == total:
        source = "habitat_humanoid_gt"
    else:
        source = "mixed"

    result = {
        "skeleton": skeleton,
        "keypoint_meta": keypoint_meta,
        "source": source,
        "per_keypoint_source": per_source,
        "keypoint_count": total,
        "direct_link_count": direct_link_count,
        "link_derived_count": link_derived_count,
        "fallback_count": fallback_count,
        "missing_keypoints": [k for k in ALL_KEYPOINTS if k not in skeleton],
    }

    if strict:
        from .humanoid_validation import validate_gt_skeleton
        validate_gt_skeleton(result, strict=True)

    return result