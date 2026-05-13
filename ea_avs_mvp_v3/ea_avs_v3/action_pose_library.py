"""
动作姿态库模块 —— action_pose_library.py
==========================================

功能：
    定义四种人体姿态的骨架关键点相对坐标。

v3.0 支持的四种姿态：
    - standing（站立）：直立姿态
    - sitting（坐姿）：髋部降低，膝盖前伸，小腿向下
    - lying_fallen（跌倒躺姿）：人体沿地面方向展开
    - bending（弯腰）：头和躯干前倾

坐标系约定：
    - 相对坐标以人体脚底中心为原点
    - 人体正面方向为 +Z（即 yaw=0 时看向正 Z 方向）
    - Y 轴向上
    - 关键点局部坐标后续由 skeleton.py 根据 human_yaw 旋转到世界坐标系

关键点数量：
    共 15 个关键点（比 v2.0 多了左右肘和左右腕）
    head, neck, pelvis,
    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
    left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
"""

from typing import Dict

# =============================================================================
# 姿态骨架字典
# =============================================================================
# 每个姿态定义 15 个关键点的相对坐标 [X, Y, Z]
# 注：这些姿态不要求生物力学完全真实，只要能表达动作状态的空间差异即可。

POSE_SKELETONS: Dict[str, Dict[str, list]] = {
    # ------------------------------------------------------------------
    # 1. standing —— 直立姿态
    #    基本与 v2.0 相同，但新增了肘部和腕部
    #    身高约 1.60m，手臂自然下垂
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. sitting —— 坐姿
    #    髋部降低到约 0.50m（坐在椅子上）
    #    膝盖前伸到 Z+ 方向，小腿垂直向下
    #    躯干略微前倾，头部高度约 1.20m
    # ------------------------------------------------------------------
    "sitting": {
        "head":           [0.00, 1.20, 0.10],
        "neck":           [0.00, 1.05, 0.05],
        "pelvis":         [0.00, 0.50, 0.00],
        "left_shoulder":  [-0.22, 1.00, 0.05],
        "right_shoulder": [ 0.22, 1.00, 0.05],
        "left_elbow":     [-0.28, 0.70, 0.10],
        "right_elbow":    [ 0.28, 0.70, 0.10],
        "left_wrist":     [-0.25, 0.50, 0.15],
        "right_wrist":    [ 0.25, 0.50, 0.15],
        "left_hip":       [-0.16, 0.50, 0.00],
        "right_hip":      [ 0.16, 0.50, 0.00],
        "left_knee":      [-0.14, 0.55, 0.60],
        "right_knee":     [ 0.14, 0.55, 0.60],
        "left_ankle":     [-0.14, 0.10, 0.55],
        "right_ankle":    [ 0.14, 0.10, 0.55],
    },

    # ------------------------------------------------------------------
    # 3. lying_fallen —— 跌倒躺姿
    #    人体沿地面展开，关键点 Y 坐标接近地面高度
    #    假设面朝上平躺：
    #    - 头部和脚部高度相近（约 0.15-0.20m）
    #    - 躯干沿 Z 方向展开
    #    - 手臂自然放在身体两侧
    # ------------------------------------------------------------------
    "lying_fallen": {
        "head":           [0.00, 0.20, -0.70],
        "neck":           [0.00, 0.18, -0.55],
        "pelvis":         [0.00, 0.15,  0.00],
        "left_shoulder":  [-0.30, 0.18, -0.45],
        "right_shoulder": [ 0.30, 0.18, -0.45],
        "left_elbow":     [-0.35, 0.15, -0.25],
        "right_elbow":    [ 0.35, 0.15, -0.25],
        "left_wrist":     [-0.30, 0.12, -0.05],
        "right_wrist":    [ 0.30, 0.12, -0.05],
        "left_hip":       [-0.16, 0.15,  0.15],
        "right_hip":      [ 0.16, 0.15,  0.15],
        "left_knee":      [-0.14, 0.20,  0.50],
        "right_knee":     [ 0.14, 0.20,  0.50],
        "left_ankle":     [-0.12, 0.10,  0.80],
        "right_ankle":    [ 0.12, 0.10,  0.80],
    },

    # ------------------------------------------------------------------
    # 4. bending —— 弯腰姿态
    #    躯干前倾约 45°，头和肩部前移
    #    髋部和膝部接近站立但略微弯曲
    #    手臂自然前垂
    # ------------------------------------------------------------------
    "bending": {
        "head":           [0.00, 1.10, 0.60],
        "neck":           [0.00, 1.00, 0.50],
        "pelvis":         [0.00, 0.85, 0.00],
        "left_shoulder":  [-0.22, 0.95, 0.45],
        "right_shoulder": [ 0.22, 0.95, 0.45],
        "left_elbow":     [-0.27, 0.65, 0.35],
        "right_elbow":    [ 0.27, 0.65, 0.35],
        "left_wrist":     [-0.25, 0.45, 0.25],
        "right_wrist":    [ 0.25, 0.45, 0.25],
        "left_hip":       [-0.16, 0.80, 0.00],
        "right_hip":      [ 0.16, 0.80, 0.00],
        "left_knee":      [-0.16, 0.45, 0.05],
        "right_knee":     [ 0.16, 0.45, 0.05],
        "left_ankle":     [-0.16, 0.10, 0.05],
        "right_ankle":    [ 0.16, 0.10, 0.05],
    },
}

# 所有姿态共用的关键点分组
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


def get_pose_skeleton(pose_type: str) -> Dict[str, list]:
    """获取指定姿态的局部骨架相对坐标。

    参数：
        pose_type: 姿态类型，支持 "standing"/"sitting"/"lying_fallen"/"bending"。

    返回：
        字典 {关键点名称: [X, Y, Z] 相对坐标}，共 15 个关键点。

    抛出异常：
        ValueError: 不支持的姿态类型。
    """
    if pose_type not in POSE_SKELETONS:
        raise ValueError(f"不支持的姿态类型: {pose_type}，"
                         f"支持的类型: {list(POSE_SKELETONS.keys())}")
    return POSE_SKELETONS[pose_type]
