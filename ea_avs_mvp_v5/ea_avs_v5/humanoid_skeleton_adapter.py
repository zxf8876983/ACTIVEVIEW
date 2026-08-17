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

关键要求：
    - 优先从 Humanoid articulated structure / joint transforms 获取（v5.0 主 GT）
    - 旧 action_pose_library 的手工坐标只能作为 legacy/debug fallback，且必须
      显式标记 source
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
    # pelvis 在 URDF 中通常无几何，用 spinae/髋中点近似，见 _pelvis_from_hips
}

# 髋部 link（用于推导 pelvis）
PELVIS_HIP_LINKS = ("left_hip", "right_hip")


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


def get_humanoid_gt_skeleton(
    humanoid_manager,
    fallback_skeleton: Dict[str, np.ndarray] = None,
) -> dict:
    """从 Humanoid 实际关节/链接状态构建 GT skeleton。

    参数：
        humanoid_manager: HumanoidManager 实例（须已 load）。
        fallback_skeleton: 可选的手工骨架字典（仅部分 link 取不到时用于该
            关键点填充，并标记 source="legacy_fallback"）。

    返回：
        {
            "skeleton": {关键点: 世界坐标},
            "source": "habitat_humanoid_gt" | "mixed" | "legacy_fallback",
            "per_keypoint_source": {关键点: "link" | "legacy"},
        }
    """
    skeleton: Dict[str, np.ndarray] = {}
    per_source: Dict[str, str] = {}

    # 1) 优先从 link transforms 取可用的关键点
    for link_name, kp in LINK_TO_KEYPOINT.items():
        if kp == "pelvis":
            continue  # 单独处理
        pos = _link_world_position(humanoid_manager, link_name)
        if pos is not None:
            skeleton[kp] = pos
            per_source[kp] = "link"

    # 2) pelvis 由左右髋中点近似
    hip_l, hip_r = None, None
    if "left_hip" in skeleton and skeleton["left_hip"] is not None:
        hip_l = skeleton["left_hip"]
    if "right_hip" in skeleton and skeleton["right_hip"] is not None:
        hip_r = skeleton["right_hip"]
    if hip_l is not None and hip_r is not None:
        skeleton["pelvis"] = 0.5 * (hip_l + hip_r)
        per_source["pelvis"] = "link_midpoint_hips"

    # 3) 缺失的关键点用 fallback 填充（显式标记 legacy）
    if fallback_skeleton is not None:
        for kp in (
            "head", "neck", "pelvis",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ):
            if kp not in skeleton and kp in fallback_skeleton:
                skeleton[kp] = np.array(fallback_skeleton[kp], dtype=np.float64)
                per_source[kp] = "legacy"

    # 4) 来源判定
    link_count = sum(1 for s in per_source.values() if s != "legacy")
    total = len(skeleton)
    if total == 0:
        source = "legacy_fallback"
    elif link_count == total:
        source = "habitat_humanoid_gt"
    else:
        source = "mixed"

    return {
        "skeleton": skeleton,
        "source": source,
        "per_keypoint_source": per_source,
    }