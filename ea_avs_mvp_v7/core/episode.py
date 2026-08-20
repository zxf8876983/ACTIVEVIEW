"""
Episode 数据结构定义 —— episode.py
=================================

功能：
    1. 定义单帧观测 EpisodeFrame (RGB/Depth 路径、相机位姿、人体 3D GT、动作标注、时间戳)；
    2. 定义完整仿真 Episode 数据结构 (包含多帧时序观测与全局元数据)；
    3. 支持标准字典与 JSON 序列化。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class EpisodeFrame:
    """Episode 中的单帧多模态感知数据。"""
    frame_index: int
    timestamp: float
    camera_position: List[float]
    camera_yaw_deg: float
    camera_pose_matrix: List[List[float]]
    camera_intrinsics: Dict[str, float]
    human_base_position: List[float]
    human_base_yaw: float
    human_pose_gt_world: Dict[str, List[float]]
    action_class: str
    action_label: str
    babel_sid: Union[int, str]
    rgb_relative_path: Optional[str] = None
    depth_relative_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    """一次完整的主动感知仿真 Episode。"""
    episode_id: str
    scene_id: str
    motion_id: str
    action_class: str
    action_label: str
    num_frames: int
    camera_view_id: str
    camera_initial_position: List[float]
    camera_initial_yaw_deg: float
    human_initial_position: List[float]
    human_initial_yaw_deg: float
    frames: List[EpisodeFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
