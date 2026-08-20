"""
移动机器人传感器与相机配置器 —— robot_sensor.py
=============================================

功能：
    1. 管理机器人搭载相机的位姿控制 (Position, Yaw, Pitch)；
    2. 计算相机内参 (Intrinsics) 与外参 (Camera-to-World / World-to-Camera)；
    3. 获取同步传感器图像数据 (RGB, Depth, Semantic)。
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError:
    habitat_sim = None
    quat_from_angle_axis = None


class RobotSensorRig:
    """机器人相机传感器总成。"""

    def __init__(self, sim, config: Dict[str, Any]):
        self.sim = sim
        self.config = config
        self.camera_cfg = config.get("camera", {})

        self.width = int(self.camera_cfg.get("width", 640))
        self.height = int(self.camera_cfg.get("height", 480))
        self.hfov_deg = float(self.camera_cfg.get("hfov_deg", 90))
        self.camera_height = float(self.camera_cfg.get("camera_height", 1.2))

        self._agent = self.sim.get_agent(0) if self.sim else None
        self._current_pos = np.zeros(3, dtype=np.float32)
        self._current_yaw = 0.0

    @property
    def intrinsics(self) -> Dict[str, float]:
        """计算相机内参 (fx, fy, cx, cy)。"""
        hfov_rad = math.radians(self.hfov_deg)
        fx = self.width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": self.width,
            "height": self.height,
            "hfov_deg": self.hfov_deg,
        }

    def set_pose(
        self,
        position: Union[List[float], np.ndarray],
        yaw_deg: float = 0.0,
    ) -> None:
        """设置机器人相机底盘位置与航向角。"""
        pos = np.asarray(position, dtype=np.float32)
        self._current_pos = pos
        self._current_yaw = float(yaw_deg)

        if self._agent is not None and quat_from_angle_axis is not None:
            state = habitat_sim.AgentState()
            state.position = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
            yaw_rad = math.radians(float(yaw_deg))
            state.rotation = quat_from_angle_axis(yaw_rad, np.array([0.0, 1.0, 0.0]))
            self._agent.set_state(state)

    def get_camera_pose_matrix(self) -> np.ndarray:
        """获取当前相机在世界坐标系下的 4x4 变换矩阵。"""
        if self._agent is None:
            return np.eye(4, dtype=np.float32)

        # 从 agent sensor node 获取准确相机绝对变换
        sensor_node = self._agent.scene_node
        mat = np.array(sensor_node.transformation, dtype=np.float32)
        return mat

    def get_observation(self) -> Dict[str, np.ndarray]:
        """从模拟器提取当前视角的 RGB 与 Depth。"""
        if self.sim is None:
            return {}

        obs = self.sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        depth = obs.get("depth_sensor")
        semantic = obs.get("semantic_sensor")

        # 确保 rgb 为 (H, W, 3) uint8
        if rgb is not None and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        return {
            "rgb": rgb,
            "depth": depth,
            "semantic": semantic,
        }
