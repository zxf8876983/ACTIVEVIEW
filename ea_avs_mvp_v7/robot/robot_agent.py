"""
移动机器人代理 —— robot_agent.py
===============================

职责：
    1. 管理移动机器人在仿真场景中的底盘基座位置与航向角；
    2. 控制 Habitat AgentState 变换。
"""

import math
from typing import List, Optional, Union

import numpy as np

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError:
    habitat_sim = None
    quat_from_angle_axis = None


class RobotAgent:
    """移动机器人底盘代理封装。"""

    def __init__(self, sim, agent_id: int = 0):
        self.sim = sim
        self.agent_id = agent_id
        self._current_pos = np.zeros(3, dtype=np.float32)
        self._current_yaw_deg = 0.0

    @property
    def agent(self):
        if self.sim is None:
            return None
        return self.sim.get_agent(self.agent_id)

    @property
    def position(self) -> np.ndarray:
        return self._current_pos.copy()

    @property
    def yaw_deg(self) -> float:
        return self._current_yaw_deg

    def set_pose(
        self,
        position: Union[List[float], np.ndarray],
        yaw_deg: float = 0.0,
    ) -> None:
        """设置机器人位置 [x, y, z] 与绕 Y 轴航向角 (度)。"""
        pos = np.asarray(position, dtype=np.float32)
        self._current_pos = pos
        self._current_yaw_deg = float(yaw_deg)

        if self.sim is not None and habitat_sim is not None and quat_from_angle_axis is not None:
            ag = self.sim.get_agent(self.agent_id)
            state = habitat_sim.AgentState()
            state.position = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
            yaw_rad = math.radians(float(yaw_deg))
            state.rotation = quat_from_angle_axis(yaw_rad, np.array([0.0, 1.0, 0.0]))
            ag.set_state(state)
