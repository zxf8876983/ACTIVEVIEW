"""
v10 核心数据结构与实体定义 —— types.py
====================================

职责：
    1. ActionClassV10: 定义 v10.0 支持的 6 大核心人体动作类别；
    2. CameraPose & CameraIntrinsics: 针孔相机内外参描述；
    3. CandidateViewpointV10: 候选视点空间几何描述；
    4. V10Sample: Phase 1 标准数据集样本元数据结构。
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ActionClassV10(str, Enum):
    """v10.0 支持的 6 大动作类别枚举。"""
    STANDING = "standing"
    WALKING = "walking"
    SITTING = "sitting"
    BENDING = "bending"
    REACHING = "reaching"
    FALLING = "falling"

    @classmethod
    def from_str(cls, s: str) -> "ActionClassV10":
        s_lower = s.lower().strip()
        if "fall" in s_lower:
            return cls.FALLING
        for item in cls:
            if item.value in s_lower or s_lower in item.value:
                return item
        raise ValueError(f"Unknown action string '{s}' for ActionClassV10")


@dataclass
class CameraIntrinsics:
    """相机几何内参。"""
    width: int = 640
    height: int = 480
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0
    hfov_deg: float = 90.0
    clip_near: float = 0.01
    clip_far: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CameraPose:
    """相机在世界坐标系下的位姿与内参。"""
    position: List[float]               # [x, y, z]
    rotation_quat: List[float]          # [x, y, z, w]
    yaw_deg: float = 0.0
    intrinsics: Optional[CameraIntrinsics] = None
    matrix_4x4: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "position": [round(float(x), 4) for x in self.position],
            "rotation_quat": [round(float(x), 4) for x in self.rotation_quat],
            "yaw_deg": round(float(self.yaw_deg), 2),
        }
        if self.intrinsics:
            d["intrinsics"] = self.intrinsics.to_dict()
        if self.matrix_4x4 is not None:
            d["matrix_4x4"] = [[round(float(c), 4) for c in row] for row in self.matrix_4x4]
        return d


@dataclass
class CandidateViewpointV10:
    """候选视点空间几何描述。"""
    viewpoint_id: str
    radius: float                       # 环绕半径 (meters)
    angle_deg: float                    # 极角偏航角 (0~360 degrees)
    height: float                       # 相机相对地面高度 (meters)
    position: List[float]               # 相机绝对世界坐标 [x, y, z]
    yaw_deg: float                      # 相机对准人体的注视偏航角
    feasible: bool = True               # 是否通过场景碰撞与空间约束检查

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "radius": round(float(self.radius), 2),
            "angle_deg": round(float(self.angle_deg), 1),
            "height": round(float(self.height), 2),
            "position": [round(float(x), 3) for x in self.position],
            "yaw_deg": round(float(self.yaw_deg), 1),
            "feasible": self.feasible,
        }


@dataclass
class V10Sample:
    """v10.0 Phase 1 标准数据集样本元数据。"""
    sample_id: str
    scene_id: str
    motion_id: str
    action_label: str
    frame_idx: int
    view_id: str
    camera_pose: CameraPose
    rgb_path: str                       # 相对数据集根目录相对路径
    depth_path: str                     # 相对数据集根目录相对路径
    gt_skeleton_path: Optional[str] = None  # 仅供 Oracle/验证，严禁模型前向输入
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "motion_id": self.motion_id,
            "action_label": self.action_label,
            "frame_idx": self.frame_idx,
            "view_id": self.view_id,
            "camera_pose": self.camera_pose.to_dict(),
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "gt_skeleton_path": self.gt_skeleton_path,
            "metadata": self.metadata,
        }
