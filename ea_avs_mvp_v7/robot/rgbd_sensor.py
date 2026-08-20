"""
机器人 RGB-D 传感器 —— rgbd_sensor.py
====================================

职责：
    1. 计算相机几何内参 (fx, fy, cx, cy)；
    2. 从仿真场景中提取当前视角的 4x4 相机外参矩阵；
    3. 获取同步的 RGB 图像 (uint8) 与 Depth 深度图 (float32 meters)。
"""

import math
from typing import Any, Dict, Optional

import numpy as np


class RGBDSensor:
    """机器人挂载的 RGB-D 相机传感器。"""

    def __init__(self, sim, sensor_cfg: Optional[Dict[str, Any]] = None, agent_id: int = 0):
        self.sim = sim
        self.sensor_cfg = sensor_cfg or {}
        self.agent_id = agent_id

        self.width = int(self.sensor_cfg.get("width", 640))
        self.height = int(self.sensor_cfg.get("height", 480))
        self.hfov_deg = float(self.sensor_cfg.get("hfov_deg", 90.0))
        self.camera_height = float(self.sensor_cfg.get("camera_height", 1.2))

    @staticmethod
    def compute_intrinsics_from_config(sensor_cfg: Dict[str, Any]) -> Dict[str, float]:
        """从配置字典计算针孔相机内参。"""
        w = float(sensor_cfg.get("width", 640))
        h = float(sensor_cfg.get("height", 480))
        hfov_deg = float(sensor_cfg.get("hfov_deg", 90.0))
        hfov_rad = math.radians(hfov_deg)
        fx = w / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": w,
            "height": h,
            "hfov_deg": hfov_deg,
        }

    @property
    def intrinsics(self) -> Dict[str, float]:
        """计算针孔相机内参矩阵参数。"""
        return self.compute_intrinsics_from_config(self.sensor_cfg)

    def get_camera_pose_matrix(self) -> np.ndarray:
        """获取相机在世界坐标系下的 4x4 变换矩阵。"""
        if self.sim is None:
            return np.eye(4, dtype=np.float32)

        ag = self.sim.get_agent(self.agent_id)
        mat = np.array(ag.scene_node.transformation, dtype=np.float32)
        return mat

    def capture(self) -> Dict[str, np.ndarray]:
        """从模拟器捕获当前视角的传感器观测。

        Returns:
            {
                "rgb": (H, W, 3) uint8 ndarray,
                "depth": (H, W) float32 ndarray (meters),
            }
        """
        if self.sim is None:
            return {}

        obs = self.sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        depth = obs.get("depth_sensor")

        # RGBA 转 RGB
        if rgb is not None and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        return {
            "rgb": rgb,
            "depth": depth,
        }
