"""
SMPL-X 到 Habitat KinematicHumanoid 关节映射与层级定义 —— joint_mapping.py
========================================================================

职责：
    1. 显式定义 SMPL-X 标准 54 关节层级与名称列表；
    2. 规范 162 维关节角 (54 * 3 Rodrigues) 到 216 维四元数 (54 * 4 Quaternions) 的维度关系；
    3. 提供关节数据切片与四元数归一化校验函数；
    4. 明确 SMPL-X 与 Habitat Humanoid 关节骨架对应关系。
"""

from typing import Dict, List, Tuple
import numpy as np

# SMPL-X 标准 54 关节名称列表 (按 index 排序)
SMPLX_JOINT_NAMES: List[str] = [
    "pelvis",            # 0 (Root)
    "left_hip",          # 1
    "right_hip",         # 2
    "spine1",            # 3
    "left_knee",         # 4
    "right_knee",        # 5
    "spine2",            # 6
    "left_ankle",        # 7
    "right_ankle",       # 8
    "spine3",            # 9
    "left_foot",         # 10
    "right_foot",        # 11
    "neck",              # 12
    "left_collar",       # 13
    "right_collar",      # 14
    "head",              # 15
    "left_shoulder",     # 16
    "right_shoulder",    # 17
    "left_elbow",        # 18
    "right_elbow",       # 19
    "left_wrist",        # 20
    "right_wrist",       # 21
    "jaw",               # 22
    "left_eye_smplhf",   # 23
    "right_eye_smplhf",  # 24
    "left_index1",       # 25
    "left_index2",       # 26
    "left_index3",       # 27
    "left_middle1",      # 28
    "left_middle2",      # 29
    "left_middle3",      # 30
    "left_pinky1",       # 31
    "left_pinky2",       # 32
    "left_pinky3",       # 33
    "left_ring1",        # 34
    "left_ring2",        # 35
    "left_ring3",        # 36
    "left_thumb1",       # 37
    "left_thumb2",       # 38
    "left_thumb3",       # 39
    "right_index1",      # 40
    "right_index2",      # 41
    "right_index3",      # 42
    "right_middle1",     # 43
    "right_middle2",     # 44
    "right_middle3",     # 45
    "right_pinky1",      # 46
    "right_pinky2",      # 47
    "right_pinky3",      # 48
    "right_ring1",       # 49
    "right_ring2",       # 50
    "right_ring3",       # 51
    "right_thumb1",      # 52
    "right_thumb2",      # 53
    "right_thumb3",      # 54
]

NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)  # 55 包括 root，或 54 body/hand joints
SMPLX_RODRIGUES_DIM = 54 * 3               # 162 维 (非 root 的 54 关节)
HABITAT_HUMANOID_QUAT_DIM = 54 * 4         # 216 维 (54 关节 * 4 四元数)


def get_joint_index(joint_name: str) -> int:
    """获取 SMPL-X 关节名称对应的序号。"""
    if joint_name not in SMPLX_JOINT_NAMES:
        raise KeyError(f"Unknown SMPL-X joint: {joint_name}")
    return SMPLX_JOINT_NAMES.index(joint_name)


def get_joint_slice(joint_index: int) -> slice:
    """获取非 root 关节在 162 维数组中的切片范围。"""
    if joint_index < 1 or joint_index >= len(SMPLX_JOINT_NAMES):
        raise IndexError(f"Joint index {joint_index} out of range [1, {len(SMPLX_JOINT_NAMES)-1}]")
    start = (joint_index - 1) * 3
    return slice(start, start + 3)


def validate_motion_quaternions(joints_array: np.ndarray, tolerance: float = 1e-2) -> bool:
    """校验转换后的 216 维关节四元数数组的模长是否归一化。"""
    if joints_array.ndim != 2 or joints_array.shape[1] != HABITAT_HUMANOID_QUAT_DIM:
        raise ValueError(
            f"Expected shape (N, {HABITAT_HUMANOID_QUAT_DIM}), got {joints_array.shape}"
        )

    num_frames = joints_array.shape[0]
    quats = joints_array.reshape(num_frames, -1, 4)
    norms = np.linalg.norm(quats, axis=2)

    # 允许全零四元数 (rest 状态未设置) 或模长接近 1.0 的有效四元数
    is_zero = norms < 1e-6
    is_unit = np.abs(norms - 1.0) < tolerance
    valid = np.all(is_zero | is_unit)
    return bool(valid)
