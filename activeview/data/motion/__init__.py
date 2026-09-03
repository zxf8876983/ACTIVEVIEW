"""文件用途：提供 AMASS/BABEL 动作读取、筛选和 Habitat 转换工具。

主要输入：BABEL 标注、AMASS 动作文件与 humanoid URDF。
主要输出：规范化动作、关节映射、grounding 信息和离线 skeleton 生成器。
项目角色：数据链中负责 motion 资产与转换的子模块。
"""

from .amass_loader import AMASSLoader, NormalizedMotion, load_amass_motion
from .joint_mapping import (
    SMPLX_JOINT_NAMES,
    SMPLX_RODRIGUES_DIM,
    HABITAT_HUMANOID_QUAT_DIM,
    get_joint_index,
    get_joint_slice,
    validate_motion_quaternions,
    validate_habitat_motion_dict,
)
from .motion_converter import MotionConverter, convert_normalized_motion_to_pkl

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
    "validate_habitat_motion_dict",
    "MotionConverter",
    "convert_normalized_motion_to_pkl",
]
