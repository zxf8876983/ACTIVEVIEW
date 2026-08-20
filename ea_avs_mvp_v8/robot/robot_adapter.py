"""
v8 机器人与传感器适配器 —— robot_adapter.py
==========================================

职责：
    1. 模块化复用 ea_avs_mvp_v7.robot.robot_agent 与 rgbd_sensor；
    2. 提供便捷将 CandidateViewpoint 应用至机器人底盘与相机的统一接口。
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from ea_avs_mvp_v8.core.types import CandidateViewpoint


class V8RobotAdapter:
    """v8 机器人与传感器控制适配器。"""

    def __init__(self, sim, sensor_cfg: Optional[Dict[str, Any]] = None):
        self.sim = sim
        self.robot = RobotAgent(sim)
        self.sensor = RGBDSensor(sim, sensor_cfg)

    def set_viewpoint(self, viewpoint: CandidateViewpoint):
        """将机器人移动并旋转对准指定候选视点。"""
        self.robot.set_pose(
            position=viewpoint.position,
            yaw_deg=viewpoint.yaw_deg,
        )

    def capture_observation(self) -> Dict[str, Any]:
        """捕获当前视点的 RGB-D 观测。"""
        return self.sensor.capture()
