"""
抽象人体骨架模块 —— skeleton.py
================================

功能：
    为 MVP0.1 提供抽象人体 3D 骨架关键点生成。

MVP0.1 约束：
    - 不使用真实 humanoid（真人形 avatar）
    - 不导入 AMASS、SMPL-X 等真实人体动作数据
    - 不调用 OpenPose、MediaPipe 等姿态估计模型
    - 只使用一个固定站姿（standing）的抽象骨架

骨架定义：
    包含 11 个关键点，分为三个组用于后续评分：
    - torso（躯干）：neck, pelvis, left_shoulder, right_shoulder, left_hip, right_hip
    - lower_body（下肢）：left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
    - head（头部）：head
    
    注意：left_hip 和 right_hip 同时属于 torso 和 lower_body 两个组，
    这是合理的设计 —— 髋关节是连接躯干和下肢的关键部位。
"""

from typing import Dict
import numpy as np

# ============================================================================
# 固定站姿骨架定义（相对坐标）
# ============================================================================
# 坐标说明：
#   - [X, Y, Z]，Y 轴向上
#   - [0, 0, 0] 表示人体脚底中心在地面上的位置
#   - 所有坐标单位：米
#   - 身高约 1.6 米（头顶高度）
#
# 关键点分布：
#   头部：      head    (0.00, 1.60, 0.00) —— 头顶
#   颈部：      neck    (0.00, 1.40, 0.00) —— 脖子位置
#   躯干中心：  pelvis  (0.00, 0.95, 0.00) —— 骨盆中心
#   肩部：      左右对称，距离中心 0.22m
#   髋部：      左右对称，距离中心 0.16m
#   膝盖：      左右对称，高度 0.50m
#   脚踝：      左右对称，高度 0.10m
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

# ============================================================================
# 关键点分组定义
# ============================================================================
# 在视角评分中，不同身体部位的权重不同：
#   - torso（躯干）：权重 0.4 —— 包含肩部、骨盆等核心部位
#   - lower_body（下肢）：权重 0.4 —— 包含髋部、膝部、脚踝
#   - head（头部）：权重 0.2 —— 只有头顶一个点
#
# 注意 left_hip 和 right_hip 同时出现在 torso 和 lower_body 中，
# 这反映了髋关节作为躯干和下肢连接处的双重角色。
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
    """
    在指定位置生成一个站立姿态的 3D 骨架。

    参数：
        human_base_pos: 人体脚底中心位置，shape=(3,)，即 (x, y, z)。
                        y 坐标通常在地面高度（navmesh 采样点的高度）。

    返回：
        字典，键为关键点名称，值为三维世界坐标（numpy 数组，shape=(3,)）。

    实现说明：
        将 SKELETON_STANDING 中定义的相对坐标偏移到 human_base_pos 处。
        例如如果 human_base_pos = [5.0, 0.0, 5.0]，
        则 head 的世界坐标为 [5.0, 1.6, 5.0]。

    关键点列表（11 个）：
        head, neck, pelvis,
        left_shoulder, right_shoulder,
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle
    """
    skeleton = {}
    for name, rel_pos in SKELETON_STANDING.items():
        # 相对坐标 + 人体基座位置 = 世界坐标
        skeleton[name] = human_base_pos + np.array(rel_pos, dtype=np.float32)
    return skeleton


def get_skeleton(
    human_base_pos: np.ndarray,
    pose_type: str = "standing",
) -> Dict[str, np.ndarray]:
    """
    根据姿态类型返回对应的骨架。
    MVP0.1 目前只支持 "standing"（站立姿态）。

    参数：
        human_base_pos: 人体脚底中心位置，shape=(3,)。
        pose_type: 姿态类型，目前仅支持 "standing"。
                   后续版本（MVP0.2）将加入：
                   - "sitting"（坐姿）
                   - "lying_fallen"（跌倒躺姿）
                   - "bending"（弯腰）

    返回：
        字典，键为关键点名称，值为三维世界坐标。

    抛出异常：
        ValueError: 当 pose_type 不支持时抛出。
    """
    if pose_type == "standing":
        return get_standing_skeleton(human_base_pos)
    else:
        raise ValueError(f"不支持的姿态类型: {pose_type}")
