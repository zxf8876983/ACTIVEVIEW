"""
v8 机器人与传感器适配器 —— robot_adapter.py
==========================================

职责：
    1. 模块化复用 ea_avs_mvp_v7.robot.robot_agent 与 rgbd_sensor；
    2. 提供将 CandidateViewpoint 应用至机器人底盘与相机的统一接口；
    3. 严格同步 Robot Base 与 Camera Sensor 位姿，并打印 [V8 Camera Debug] 日志。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from ea_avs_mvp_v8.core.types import CandidateViewpoint

logger = logging.getLogger(__name__)


class V8RobotAdapter:
    """v8 机器人底盘与 RGB-D 相机传感器适配器。"""

    def __init__(self, sim, sensor_cfg: Optional[Dict[str, Any]] = None, agent_id: int = 0):
        self.sim = sim
        self.agent_id = agent_id
        self.sensor_cfg = sensor_cfg or {}
        self.robot = RobotAgent(sim, agent_id=agent_id)
        self.sensor = RGBDSensor(sim, sensor_cfg=self.sensor_cfg, agent_id=agent_id)

    def set_viewpoint(self, viewpoint: CandidateViewpoint, verbose: bool = True) -> Dict[str, Any]:
        """将机器人移动并旋转对准指定候选视点，并同步相机传感器。"""
        # 1. 更新机器人底盘位姿
        self.robot.set_pose(
            position=viewpoint.position,
            yaw_deg=viewpoint.yaw_deg,
        )

        # 2. 提取同步后的 Agent 与 Camera 位姿
        cam_pos = [viewpoint.position[0], viewpoint.position[1] + viewpoint.camera_height, viewpoint.position[2]]
        cam_rot_list = [0.0, 0.0, 0.0, 1.0]

        if self.sim is not None:
            try:
                ag = self.sim.get_agent(self.agent_id)
                ag_state = ag.get_state()
                if "color_sensor" in ag_state.sensor_states:
                    c_state = ag_state.sensor_states["color_sensor"]
                    cam_pos = [float(x) for x in c_state.position]
                    cam_rot = c_state.rotation
                    cam_rot_list = [float(cam_rot.x), float(cam_rot.y), float(cam_rot.z), float(cam_rot.w)]
                else:
                    cam_pos = [float(x) for x in ag_state.position]
            except Exception as e:
                logger.debug("Could not read agent sensor state directly: %s", e)

        # 3. 打印标准化 [V8 Camera Debug]
        if verbose:
            print("\n[V8 Camera Debug]")
            print(f"Robot position:  {[round(x, 3) for x in viewpoint.position]}")
            print(f"Robot yaw:       {viewpoint.yaw_deg:.1f} deg")
            print(f"Camera position: {[round(x, 3) for x in cam_pos]}")
            print(f"Camera rotation: {[round(x, 3) for x in cam_rot_list]}")

        return {
            "robot_position": list(viewpoint.position),
            "robot_yaw_deg": float(viewpoint.yaw_deg),
            "camera_position": cam_pos,
            "camera_rotation": cam_rot_list,
        }

    def capture_observation(self) -> Dict[str, np.ndarray]:
        """捕获当前视点的 RGB-D 观测。"""
        return self.sensor.capture()

    def get_camera_pose_matrix(self) -> np.ndarray:
        """获取相机 4x4 外参矩阵。"""
        return self.sensor.get_camera_pose_matrix()
