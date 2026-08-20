"""
观测元数据规范 —— metadata.py
==============================

功能：
    1. 定义单帧观测元数据 FrameMetadata；
    2. 定义完整时序序列元数据 SequenceMetadata。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FrameMetadata:
    """单帧多模态感知元数据。"""
    frame_index: int
    timestamp: float
    action_class: str
    action_label: str
    babel_sid: Union[int, str]
    camera_position: List[float]
    camera_yaw_deg: float
    camera_pose_matrix: List[List[float]]
    camera_intrinsics: Dict[str, float]
    human_base_position: List[float]
    human_base_yaw: float
    human_pose_gt_world: Dict[str, List[float]]
    rgb_relative_path: Optional[str] = None
    depth_relative_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceMetadata:
    """整段动作序列观测元数据。"""
    sequence_id: str
    action_class: str
    action_label: str
    babel_sid: Union[int, str]
    num_frames: int
    camera_position: List[float]
    camera_yaw_deg: float
    frames: List[FrameMetadata] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
