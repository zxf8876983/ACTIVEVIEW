"""
骨架生成模块 —— skeleton.py
=============================

功能：
    根据人体位置、姿态类型和人体朝向，将局部骨架坐标旋转到世界坐标系。

v3.0 核心变化：
    - 从 action_pose_library.py 读取多姿态骨架
    - 支持按 human_yaw 旋转局部坐标到世界坐标
    - 新增 arms（手臂）关键点分组
"""

from typing import Dict
import numpy as np

from .action_pose_library import get_pose_skeleton, KEYPOINT_GROUPS


def rotate_local_point_by_yaw(local_point: np.ndarray, yaw: float) -> np.ndarray:
    """将局部坐标点绕 Y 轴旋转指定角度。

    参数：
        local_point: 局部三维坐标，shape=(3,)，原点在人体脚底中心。
        yaw: 人体朝向角（弧度制）。yaw=0 表示人体朝向 +Z。

    返回：
        旋转后的三维坐标，shape=(3,)。

    旋转矩阵（绕 Y 轴）：
        [cos(yaw),  0, sin(yaw)]
        [0,          1, 0       ]
        [-sin(yaw), 0, cos(yaw) ]

    用途：
        人体局部坐标定义中，正面朝向 +Z。当人体实际朝向（human_yaw）不同时，
        需要将所有关键点绕 Y 轴旋转。
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x = local_point[0] * cos_yaw + local_point[2] * sin_yaw
    y = local_point[1]
    z = -local_point[0] * sin_yaw + local_point[2] * cos_yaw
    return np.array([x, y, z], dtype=np.float32)


def get_skeleton(
    human_base_pos: np.ndarray,
    pose_type: str,
    human_yaw: float,
) -> Dict[str, np.ndarray]:
    """根据人体位置、姿态类型和人体朝向，生成世界坐标骨架。

    参数：
        human_base_pos: 人体脚底中心位置，shape=(3,)。
        pose_type: 姿态类型，如 "standing"/"sitting" 等。
        human_yaw: 人体朝向角（弧度制）。

    返回：
        字典 {关键点名称: 三维世界坐标}，共 15 个关键点。

    实现步骤：
        1. 从 pose_library 读取 pose_type 对应的局部骨架
        2. 对每个关键点局部坐标绕 Y 轴旋转 human_yaw
        3. 加上 human_base_pos 得到世界坐标
    """
    # 读取局部骨架
    local_skeleton = get_pose_skeleton(pose_type)

    # 旋转并平移到世界坐标
    skeleton = {}
    for name, local_pos in local_skeleton.items():
        local_np = np.array(local_pos, dtype=np.float32)
        rotated = rotate_local_point_by_yaw(local_np, human_yaw)
        world_pos = human_base_pos + rotated
        skeleton[name] = world_pos

    return skeleton
