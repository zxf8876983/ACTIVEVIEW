"""
Humanoid 状态模块 —— humanoid_state.py
=======================================

功能：
    保存 Humanoid 当前 GT 状态（供调试与 GT-State 支路使用）。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HumanoidState:
    base_position: np.ndarray
    base_yaw: float
    requested_yaw: float = 0.0
    actual_base_yaw: float = 0.0
    pose_name: str = "standing"
    motion_frame: Optional[int] = None
    semantic_id: int = 100

    def to_dict(self) -> dict:
        return {
            "base_position": (
                self.base_position.tolist()
                if hasattr(self.base_position, "tolist")
                else list(self.base_position)
            ),
            "base_yaw": float(self.base_yaw),
            "requested_yaw": float(self.requested_yaw),
            "actual_base_yaw": float(self.actual_base_yaw),
            "pose_name": self.pose_name,
            "motion_frame": self.motion_frame,
            "semantic_id": int(self.semantic_id),
        }
