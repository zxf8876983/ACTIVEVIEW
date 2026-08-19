"""
人体状态估计数据结构模块 —— estimated_human_state.py
===================================================

功能：
    定义封装完整的 EstimatedHumanState 数据结构及其序列化方法。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .depth_lifter import EstimatedJoint3D


@dataclass
class EstimatedHumanState:
    """由当前 RGB-D 观测估计出的人体状态。"""
    valid: bool

    # 1. 人体世界位置（通常对齐脚底/基座中心）
    human_position_world: Optional[np.ndarray]
    human_position_confidence: float = 0.0
    human_position_source: str = "invalid"

    # 2. 人体水平朝向（Yaw，弧度）
    human_yaw: Optional[float] = None
    yaw_confidence: float = 0.0
    yaw_source: str = "invalid"

    # 3. 人体尺度比例
    body_scale: Optional[float] = None
    body_scale_confidence: float = 0.0
    body_scale_source: str = "invalid"

    # 4. 关键点与骨架表示
    joints: Dict[str, EstimatedJoint3D] = field(default_factory=dict)
    observed_skeleton: Dict[str, np.ndarray] = field(default_factory=dict)
    proxy_full_skeleton: Dict[str, np.ndarray] = field(default_factory=dict)

    # 5. 可见性与补全划分列表
    visible_2d_keypoints: List[str] = field(default_factory=list)
    observable_3d_keypoints: List[str] = field(default_factory=list)
    missing_keypoints: List[str] = field(default_factory=list)
    template_completed_keypoints: List[str] = field(default_factory=list)

    # 6. 置信度与检测分数
    pose_detection_score: float = 0.0
    state_confidence: float = 0.0

    # 7. 失败/失效原因记录
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典，用于 JSONL 记录与调试文件。"""
        return {
            "valid": bool(self.valid),
            "human_position_world": (
                self.human_position_world.tolist()
                if self.human_position_world is not None
                else None
            ),
            "human_position_confidence": float(self.human_position_confidence),
            "human_position_source": self.human_position_source,
            "human_yaw": float(self.human_yaw) if self.human_yaw is not None else None,
            "human_yaw_deg": (
                float(np.rad2deg(self.human_yaw))
                if self.human_yaw is not None
                else None
            ),
            "yaw_confidence": float(self.yaw_confidence),
            "yaw_source": self.yaw_source,
            "body_scale": float(self.body_scale) if self.body_scale is not None else None,
            "body_scale_confidence": float(self.body_scale_confidence),
            "body_scale_source": self.body_scale_source,
            "visible_2d_keypoints": list(self.visible_2d_keypoints),
            "observable_3d_keypoints": list(self.observable_3d_keypoints),
            "missing_keypoints": list(self.missing_keypoints),
            "template_completed_keypoints": list(self.template_completed_keypoints),
            "pose_detection_score": float(self.pose_detection_score),
            "state_confidence": float(self.state_confidence),
            "failure_reason": self.failure_reason,
            "observed_skeleton": {
                k: v.tolist() for k, v in self.observed_skeleton.items()
            },
            "proxy_full_skeleton": {
                k: v.tolist() for k, v in self.proxy_full_skeleton.items()
            },
        }
