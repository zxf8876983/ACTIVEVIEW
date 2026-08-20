"""
EA-AVS-MVP v7.0 动作加载、规范化与回放子模块
"""

from .amass_loader import AMASSLoader, NormalizedMotion, load_amass_motion
from .joint_mapping import (
    SMPLX_JOINT_NAMES,
    SMPLX_RODRIGUES_DIM,
    HABITAT_HUMANOID_QUAT_DIM,
    get_joint_index,
    get_joint_slice,
    validate_motion_quaternions,
)
from .motion_converter import MotionConverter, convert_normalized_motion_to_pkl
from .motion_player import MotionPlayer

__all__ = [
    "AMASSLoader",
    "NormalizedMotion",
    "load_amass_motion",
    "SMPLX_JOINT_NAMES",
    "SMPLX_RODRIGUES_DIM",
    "HABITAT_HUMANOID_QUAT_DIM",
    "get_joint_index",
    "get_joint_slice",
    "validate_motion_quaternions",
    "MotionConverter",
    "convert_normalized_motion_to_pkl",
    "MotionPlayer",
]
