"""
Habitat Humanoid 关节到人体 16 关键点映射与提取模块 —— keypoint_mapping.py
========================================================================

职责：
    1. 显式管理 Habitat Humanoid URDF link 到人体 16 个核心关键点的映射关系；
    2. 从仿真世界 ArticulatedAgent sim_obj 中提取各关节的 3D 世界坐标；
    3. Pelvis 关键点通过左右髋关节中点几何计算或基座变换确定；
    4. 提供关键点完备性校验函数，防止关键关节缺失。
"""

from typing import Any, Dict, List, Optional
import numpy as np

# 标准 16 关键点名称映射至 KinematicHumanoid URDF 关节 link name
KEYPOINT_LINK_MAP: Dict[str, str] = {
    "pelvis": "pelvis",
    "spine3": "spine3",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
}

HUMAN_16_KEYPOINTS: List[str] = list(KEYPOINT_LINK_MAP.keys())


def extract_human_keypoints_3d(
    ao_sim_obj: Any,
    fallback_base_pos: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """从 ArticulatedObject/KinematicHumanoid 仿真对象中提取 16 关节 3D 世界坐标。"""
    if ao_sim_obj is None:
        return {}

    positions: Dict[str, np.ndarray] = {}

    for kpt_name, link_name in KEYPOINT_LINK_MAP.items():
        if kpt_name == "pelvis":
            continue
        try:
            link_id = ao_sim_obj.get_link_id_from_name(link_name)
            if link_id is not None and int(link_id) >= 0:
                node = ao_sim_obj.get_link_scene_node(link_id)
                pos = np.array(node.transformation.translation, dtype=np.float32)
                positions[kpt_name] = pos
        except Exception:
            pass

    # pelvis 由左右髋关节中点几何推导
    if "left_hip" in positions and "right_hip" in positions:
        positions["pelvis"] = 0.5 * (positions["left_hip"] + positions["right_hip"])
    else:
        try:
            positions["pelvis"] = np.array(ao_sim_obj.transformation.translation, dtype=np.float32)
        except Exception:
            if fallback_base_pos is not None:
                positions["pelvis"] = np.asarray(fallback_base_pos, dtype=np.float32)
            else:
                positions["pelvis"] = np.zeros(3, dtype=np.float32)

    return {k: [float(x) for x in v] for k, v in positions.items()}


def validate_keypoints(
    keypoints_dict: Dict[str, List[float]],
    min_joints: int = 15,
) -> Dict[str, Any]:
    """校验关键点字典提取完整性。"""
    num_extracted = len(keypoints_dict)
    if num_extracted < min_joints:
        raise ValueError(
            f"Insufficient extracted keypoints: got {num_extracted}, required >= {min_joints}. "
            f"Extracted: {list(keypoints_dict.keys())}"
        )

    for kpt_name, coords in keypoints_dict.items():
        if len(coords) != 3:
            raise ValueError(f"Keypoint '{kpt_name}' does not have 3D coordinates: {coords}")

    return {
        "num_habitat_links": len(KEYPOINT_LINK_MAP),
        "num_human_keypoints": len(HUMAN_16_KEYPOINTS),
        "num_extracted": num_extracted,
        "is_complete": num_extracted >= len(HUMAN_16_KEYPOINTS),
    }
