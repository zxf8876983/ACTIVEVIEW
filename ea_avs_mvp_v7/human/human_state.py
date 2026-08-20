"""
人体物理状态与骨架 Ground-Truth —— human_state.py
================================================

功能：
    1. 封装时刻 t 人体的空间位置、朝向、帧号与动作标签；
    2. 记录人体 16 个核心关节在世界坐标系下的 3D 真实坐标 (Ground-Truth)。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HumanState:
    """人体空间状态与 3D 骨架真值。"""
    position: List[float]                          # [x, y, z] 基座世界坐标
    orientation_yaw_rad: float                     # 绕 Y 轴朝向弧度
    frame_id: int                                  # 当前动作帧序号
    timestamp: float                               # 当前时刻 (秒)
    action_class: str                              # 动作类别 (如 fall_related)
    action_label: str                              # 语义标签 (如 fall to the ground)
    joint_positions_3d_world: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
