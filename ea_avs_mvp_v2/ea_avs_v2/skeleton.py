"""
抽象人体骨架模块 —— skeleton.py
=================================

功能：
    生成抽象人体 3D 骨架关键点。MVP v2.0 只支持 standing 姿态。

关键点分组：
    - torso（躯干）：权重 0.4
    - lower_body（下肢）：权重 0.4
    - head（头部）：权重 0.2

注意：left_hip 和 right_hip 同时属于 torso 和 lower_body 两组。
"""

from typing import Dict
import numpy as np

# 固定站姿骨架 —— 相对坐标（相对于脚底中心）
# 身高约 1.6 米
SKELETON_STANDING = {
    "head":           [0.00, 1.60, 0.00],
    "neck":           [0.00, 1.40, 0.00],
    "pelvis":         [0.00, 0.95, 0.00],
    "left_shoulder":  [-0.22, 1.35, 0.00],
    "right_shoulder": [ 0.22, 1.35, 0.00],
    "left_hip":       [-0.16, 0.90, 0.00],
    "right_hip":      [ 0.16, 0.90, 0.00],
    "left_knee":      [-0.16, 0.50, 0.00],
    "right_knee":     [ 0.16, 0.50, 0.00],
    "left_ankle":     [-0.16, 0.10, 0.00],
    "right_ankle":    [ 0.16, 0.10, 0.00],
}

KEYPOINT_GROUPS = {
    "torso": [
        "neck", "pelvis",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
    ],
    "lower_body": [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ],
    "head": ["head"],
}


def get_standing_skeleton(human_base_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """在指定位置生成站立姿态的 3D 骨架。

    参数：
        human_base_pos: 人体脚底中心位置，shape=(3,)。

    返回：
        字典 {关键点名称: 三维世界坐标}，共 11 个关键点。
    """
    skeleton = {}
    for name, rel_pos in SKELETON_STANDING.items():
        skeleton[name] = human_base_pos + np.array(rel_pos, dtype=np.float32)
    return skeleton


def get_skeleton(
    human_base_pos: np.ndarray,
    pose_type: str = "standing",
) -> Dict[str, np.ndarray]:
    """根据姿态类型返回骨架。

    参数：
        human_base_pos: 人体脚底中心位置，shape=(3,)。
        pose_type: 姿态类型，目前仅支持 "standing"。

    返回：
        字典 {关键点名称: 三维世界坐标}。
    """
    if pose_type == "standing":
        return get_standing_skeleton(human_base_pos)
    else:
        raise ValueError(f"不支持的姿态类型: {pose_type}")
