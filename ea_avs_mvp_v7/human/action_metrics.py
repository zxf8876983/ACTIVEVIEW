"""
人体动作真实性与运动学评价模块 —— action_metrics.py
===================================================

职责：
    1. 接收时序 16 关键点 3D 世界坐标与对应时间戳；
    2. 计算多维动力学指标：
       - root height change (Pelvis 垂直落差)
       - pelvis velocity (Pelvis 空间线速度)
       - body orientation change (水平朝向角变化)
       - torso angle change (躯干倾角变化)
       - joint displacement (全关节平均位移)
       - joint motion score (综合动力学运动得分)
       - dynamic motion (动态真实运动布尔判定)；
    3. 区分静止状态 (如 standing) 与动态过程 (如 fall_related / bending)。
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ActionMotionMetrics:
    """人体动作时序动力学指标。"""
    height_change: float            # Pelvis 垂直高度变化极差 (米)
    pelvis_velocity: float          # Pelvis 平均线速度 (米/秒)
    orientation_change: float       # 身体水平朝向最大转角 (度)
    torso_angle_change: float       # 躯干倾角最大偏角变化 (度)
    joint_displacement: float       # 全关节平均累积空间位移 (米)
    joint_motion_score: float       # 综合运动活力得分 [0.0, 1.0]
    dynamic_motion: bool            # 是否判定为有效动态运动

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两三维向量夹角 (度)。"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = np.dot(v1, v2) / (norm1 * norm2)
    dot = np.clip(dot, -1.0, 1.0)
    return float(math.degrees(math.acos(dot)))


def compute_action_motion_metrics(
    frames_gt_list: List[Dict[str, List[float]]],
    timestamps: Optional[List[float]] = None,
) -> ActionMotionMetrics:
    """从连续 3D 关节 GT 列表中计算多维动作运动指标。

    Args:
        frames_gt_list: 每帧包含 16 关键点 3D 坐标的字典列表
        timestamps: 每帧的时间戳 (秒)，若为空则默认按 30fps 均匀采样

    Returns:
        ActionMotionMetrics 动力学评价结果
    """
    num_frames = len(frames_gt_list)
    if num_frames == 0:
        return ActionMotionMetrics(
            height_change=0.0,
            pelvis_velocity=0.0,
            orientation_change=0.0,
            torso_angle_change=0.0,
            joint_displacement=0.0,
            joint_motion_score=0.0,
            dynamic_motion=False,
        )

    if timestamps is None or len(timestamps) != num_frames:
        timestamps = [i * (1.0 / 30.0) for i in range(num_frames)]

    # 1. 提取时序 Pelvis、左右髋关节与躯干向量
    pelvis_positions = []
    hip_angles = []
    torso_angles = []

    for f_idx, gt in enumerate(frames_gt_list):
        p = np.array(gt.get("pelvis", [0.0, 0.0, 0.0]), dtype=np.float32)
        pelvis_positions.append(p)

        # 左右髋水平朝向
        if "left_hip" in gt and "right_hip" in gt:
            l_hip = np.array(gt["left_hip"], dtype=np.float32)
            r_hip = np.array(gt["right_hip"], dtype=np.float32)
            hip_vec = r_hip - l_hip
            # X-Z 平面夹角 (航向角)
            yaw = math.atan2(float(hip_vec[2]), float(hip_vec[0]))
            hip_angles.append(math.degrees(yaw))

        # 躯干垂直倾角 (pelvis -> neck / head)
        upper = None
        if "neck" in gt:
            upper = np.array(gt["neck"], dtype=np.float32)
        elif "head" in gt:
            upper = np.array(gt["head"], dtype=np.float32)

        if upper is not None:
            torso_vec = upper - p
            # 与正垂直方向 [0, 1, 0] 的夹角
            vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            torso_ang = _compute_angle_between_vectors_deg(torso_vec, vertical)
            torso_angles.append(torso_ang)

    pelvis_arr = np.array(pelvis_positions)  # (N, 3)

    # 2. 计算各维度指标
    # 2.1 高度变化
    heights = pelvis_arr[:, 1]
    height_change = float(np.max(heights) - np.min(heights))

    # 2.2 Pelvis 速度
    velocities = []
    for i in range(1, num_frames):
        dt = max(1e-4, timestamps[i] - timestamps[i - 1])
        dp = np.linalg.norm(pelvis_arr[i] - pelvis_arr[i - 1])
        velocities.append(dp / dt)
    pelvis_velocity = float(np.mean(velocities)) if velocities else 0.0

    # 2.3 朝向角变化
    if hip_angles:
        # 处理角度环绕
        unwrapped = np.unwrap(np.radians(hip_angles))
        deg_unwrapped = np.degrees(unwrapped)
        orientation_change = float(np.max(deg_unwrapped) - np.min(deg_unwrapped))
    else:
        orientation_change = 0.0

    # 2.4 躯干倾角变化
    if torso_angles:
        torso_angle_change = float(np.max(torso_angles) - np.min(torso_angles))
    else:
        torso_angle_change = 0.0

    # 2.5 全关节位移
    common_keys = set(frames_gt_list[0].keys())
    for f_gt in frames_gt_list[1:]:
        common_keys = common_keys.intersection(f_gt.keys())

    displacements = []
    if common_keys and num_frames > 1:
        for k in common_keys:
            p0 = np.array(frames_gt_list[0][k])
            p_final = np.array(frames_gt_list[-1][k])
            displacements.append(np.linalg.norm(p_final - p0))
        joint_displacement = float(np.mean(displacements))
    else:
        joint_displacement = 0.0

    # 2.6 综合运动活力得分
    score_h = np.clip(height_change / 0.45, 0.0, 1.0)
    score_a = np.clip(torso_angle_change / 40.0, 0.0, 1.0)
    score_v = np.clip(pelvis_velocity / 0.8, 0.0, 1.0)
    score_d = np.clip(joint_displacement / 0.5, 0.0, 1.0)
    joint_motion_score = float(0.3 * score_h + 0.3 * score_a + 0.2 * score_v + 0.2 * score_d)

    # 2.7 动态动作判定 (显著高度下落、明显倾斜偏转或高活力得分)
    dynamic_motion = bool(
        height_change > 0.15
        or torso_angle_change > 20.0
        or joint_motion_score > 0.25
        or pelvis_velocity > 0.4
    )

    return ActionMotionMetrics(
        height_change=round(height_change, 4),
        pelvis_velocity=round(pelvis_velocity, 4),
        orientation_change=round(orientation_change, 2),
        torso_angle_change=round(torso_angle_change, 2),
        joint_displacement=round(joint_displacement, 4),
        joint_motion_score=round(joint_motion_score, 4),
        dynamic_motion=dynamic_motion,
    )
