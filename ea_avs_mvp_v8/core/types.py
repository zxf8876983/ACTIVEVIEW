"""
v8.0 核心数据类型定义 —— types.py
=================================

包含：
    1. CandidateViewpoint: 机器人候选观察视点数据结构 (包含 position, yaw, camera_pose)；
    2. ViewpointQuality: 候选视点观测质量评价数据结构；
    3. HumanPlacementPose: 人体放置位姿数据结构。
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class HumanPlacementPose:
    """人体放置位姿结构。"""
    position: List[float]          # [x, y, z] 世界坐标
    yaw_deg: float = 0.0           # 朝向角 (度)
    scale: float = 1.0             # 缩放比例
    is_valid: bool = True          # 是否通过有效性校验
    scene_id: str = "apartment_1"  # 所属场景
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [float(x) for x in self.position],
            "yaw_deg": float(self.yaw_deg),
            "scale": float(self.scale),
            "is_valid": bool(self.is_valid),
            "scene_id": self.scene_id,
            "metadata": self.metadata,
        }


@dataclass
class CandidateViewpoint:
    """机器人候选观察视点数据结构。"""
    viewpoint_id: str              # 视点标识符 (如 vp_000_r1.5_a000)
    position: List[float]          # 机器人底盘三维坐标 [x, ground_height, z]
    yaw_deg: float                 # 机器人朝向角 (度，正对人体)
    radius: float                  # 采样半径 (米)
    angle_deg: float               # 相对人体水平方位角 (0-360度)
    camera_height: float = 1.2     # 相机安装高度 (米)
    ground_height: float = -1.60   # 地面标高 (米)
    camera_pose: Optional[Dict[str, Any]] = None  # 相机空间位姿
    feasible: bool = True          # 是否在 NavMesh 可行域内且可达
    reason: str = "valid"          # 状态说明 (valid, not_navigable, unreachable)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        cam_pos = [
            float(self.position[0]),
            float(self.position[1]) + float(self.camera_height),
            float(self.position[2]),
        ]
        yaw_rad = math.radians(float(self.yaw_deg))
        half_y = yaw_rad / 2.0
        default_cam_rot = [0.0, float(math.sin(half_y)), 0.0, float(math.cos(half_y))]

        c_pose = self.camera_pose or {
            "position": cam_pos,
            "rotation": default_cam_rot,
        }

        return {
            "viewpoint_id": self.viewpoint_id,
            "position": [float(x) for x in self.position],
            "yaw": float(self.yaw_deg),
            "yaw_deg": float(self.yaw_deg),
            "radius": float(self.radius),
            "angle_deg": float(self.angle_deg),
            "camera_height": float(self.camera_height),
            "camera_pose": c_pose,
            "feasible": bool(self.feasible),
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class ViewpointQuality:
    """视点观测质量评价结构。"""
    viewpoint_id: str              # 视点标识符
    distance: float                # 视点到人体的欧氏距离 (米)
    viewing_angle_deg: float       # 观测视线与人体正面的夹角 (度)
    visible_joints_count: int      # 视锥内可见关节点数量 (16 关节点基准)
    visibility_score: float        # 综合可见性得分 [0.0, 1.0]
    occlusion_ratio: float = 0.0   # 遮挡比例 [0.0, 1.0]
    pose_coverage: float = 1.0     # 骨骼姿态覆盖率 [0.0, 1.0]
    is_valid: bool = True          # 是否为有效观察视点
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "distance": float(self.distance),
            "viewing_angle_deg": float(self.viewing_angle_deg),
            "visible_joints_count": int(self.visible_joints_count),
            "visibility_score": float(self.visibility_score),
            "occlusion_ratio": float(self.occlusion_ratio),
            "pose_coverage": float(self.pose_coverage),
            "is_valid": bool(self.is_valid),
            "metadata": self.metadata,
        }
