"""
动作姿态库模块 —— action_pose_library.py
==========================================

功能：
    定义人体姿态的骨架关键点相对坐标。
"""

from typing import Dict

POSE_SKELETONS: Dict[str, Dict[str, list]] = {
    "standing": {
        "head":           [0.00, 1.60, 0.00],
        "neck":           [0.00, 1.40, 0.00],
        "pelvis":         [0.00, 0.95, 0.00],
        "left_shoulder":  [-0.22, 1.35, 0.00],
        "right_shoulder": [ 0.22, 1.35, 0.00],
        "left_elbow":     [-0.28, 0.95, 0.00],
        "right_elbow":    [ 0.28, 0.95, 0.00],
        "left_wrist":     [-0.28, 0.60, 0.00],
        "right_wrist":    [ 0.28, 0.60, 0.00],
        "left_hip":       [-0.16, 0.90, 0.00],
        "right_hip":      [ 0.16, 0.90, 0.00],
        "left_knee":      [-0.16, 0.50, 0.00],
        "right_knee":     [ 0.16, 0.50, 0.00],
        "left_ankle":     [-0.16, 0.10, 0.00],
        "right_ankle":    [ 0.16, 0.10, 0.00],
    },
    "sitting": {
        "head":           [0.00, 1.25, 0.00],
        "neck":           [0.00, 1.05, 0.00],
        "pelvis":         [0.00, 0.50, 0.00],
        "left_shoulder":  [-0.22, 1.00, 0.00],
        "right_shoulder": [ 0.22, 1.00, 0.00],
        "left_elbow":     [-0.28, 0.70, 0.15],
        "right_elbow":    [ 0.28, 0.70, 0.15],
        "left_wrist":     [-0.20, 0.55, 0.35],
        "right_wrist":    [ 0.20, 0.55, 0.35],
        "left_hip":       [-0.16, 0.50, 0.00],
        "right_hip":      [ 0.16, 0.50, 0.00],
        "left_knee":      [-0.16, 0.50, 0.40],
        "right_knee":     [ 0.16, 0.50, 0.40],
        "left_ankle":     [-0.16, 0.10, 0.40],
        "right_ankle":    [ 0.16, 0.10, 0.40],
    },
    "lying_fallen": {
        "head":           [0.00, 0.15, -0.70],
        "neck":           [0.00, 0.15, -0.50],
        "pelvis":         [0.00, 0.15,  0.00],
        "left_shoulder":  [-0.22, 0.15, -0.45],
        "right_shoulder": [ 0.22, 0.15, -0.45],
        "left_elbow":     [-0.35, 0.15, -0.20],
        "right_elbow":    [ 0.35, 0.15, -0.20],
        "left_wrist":     [-0.40, 0.15,  0.05],
        "right_wrist":    [ 0.40, 0.15,  0.05],
        "left_hip":       [-0.16, 0.15,  0.00],
        "right_hip":      [ 0.16, 0.15,  0.00],
        "left_knee":      [-0.18, 0.15,  0.40],
        "right_knee":     [ 0.18, 0.15,  0.40],
        "left_ankle":     [-0.18, 0.15,  0.80],
        "right_ankle":    [ 0.18, 0.15,  0.80],
    },
    "bending": {
        "head":           [0.00, 0.85, 0.45],
        "neck":           [0.00, 0.90, 0.30],
        "pelvis":         [0.00, 0.90, 0.00],
        "left_shoulder":  [-0.22, 0.90, 0.25],
        "right_shoulder": [ 0.22, 0.90, 0.25],
        "left_elbow":     [-0.25, 0.65, 0.35],
        "right_elbow":    [ 0.25, 0.65, 0.35],
        "left_wrist":     [-0.25, 0.40, 0.40],
        "right_wrist":    [ 0.25, 0.40, 0.40],
        "left_hip":       [-0.16, 0.85, 0.00],
        "right_hip":      [ 0.16, 0.85, 0.00],
        "left_knee":      [-0.16, 0.48, 0.05],
        "right_knee":     [ 0.16, 0.48, 0.05],
        "left_ankle":     [-0.16, 0.10, 0.05],
        "right_ankle":    [ 0.16, 0.10, 0.05],
    },
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
    "arms": [
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ],
}


def get_pose_skeleton(pose_type: str = "standing") -> Dict[str, list]:
    if pose_type not in POSE_SKELETONS:
        raise ValueError(f"不支持的姿态类型: {pose_type}")
    return POSE_SKELETONS[pose_type]
