"""
Humanoid 状态模块 —— humanoid_state.py
=======================================

功能：
    保存 Humanoid 当前 GT 状态（供调试与 GT-State 支路使用）。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HumanoidState:
    """Humanoid 当前 GT 状态。

    属性：
        base_position: 人体脚底中心位置，shape=(3,)。
        base_yaw: 人体基准朝向（弧度），即 set_base_pose 请求的 yaw。
        actual_base_yaw: 从 agent 实际读回的朝向（可能因 mesh/controller 有偏差）。
        requested_yaw: episode 请求的 human_yaw（随机化采样值）。
        pose_name: 当前姿态/动作名称，如 "standing" / "walking"。
        motion_frame: 动画帧索引；非动画姿态时为 None。
        semantic_id: Humanoid 语义标号（调试用）。
    """
    base_position: np.ndarray
    base_yaw: float
    requested_yaw: float = 0.0
    actual_base_yaw: float = 0.0
    pose_name: str = "standing"
    motion_frame: Optional[int] = None
    semantic_id: int = 100

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化字典（用于 debug/metrics）。"""
        return {
            "base_position": self.base_position.tolist()
            if hasattr(self.base_position, "tolist")
            else list(self.base_position),
            "base_yaw": float(self.base_yaw),
            "requested_yaw": float(self.requested_yaw),
            "actual_base_yaw": float(self.actual_base_yaw),
            "pose_name": self.pose_name,
            "motion_frame": self.motion_frame,
            "semantic_id": int(self.semantic_id),
        }