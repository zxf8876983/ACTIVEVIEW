"""
机器人 RGB-D 传感器 —— rgbd_sensor.py
====================================

职责：
    1. 计算相机几何内参 (fx, fy, cx, cy)；
    2. 从仿真场景中提取当前视角的 4x4 相机外参矩阵；
    3. 获取同步的 RGB 图像 (uint8) 与 Depth 深度图 (float32 meters)；
    4. 提供 check_object_in_view 检查目标空间点是否落在当前相机视锥内。
"""

import math
from typing import Any, Dict, List, Optional, Union

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
        self.clip_near = float(self.sensor_cfg.get("clip_near", 0.01))
        self.clip_far = float(self.sensor_cfg.get("clip_far", 10.0))

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

    def check_object_in_view(
        self,
        target_pos_3d: Union[List[float], np.ndarray],
        camera_mat: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """检查空间 3D 目标点是否落在当前相机视锥与图像范围内。

        Args:
            target_pos_3d: 世界坐标系下的目标 3D 坐标 [x, y, z]
            camera_mat: 相机 4x4 外参矩阵 (若为空则自动获取当前相机位姿)

        Returns:
            {
                "visible": bool,
                "distance": float (meters),
                "angle": float (degrees),
                "pixel_coord": [u, v],
                "in_frustum": bool,
            }
        """
        if camera_mat is None:
            camera_mat = self.get_camera_pose_matrix()

        pt_w = np.array([float(target_pos_3d[0]), float(target_pos_3d[1]), float(target_pos_3d[2]), 1.0], dtype=np.float32)
        inv_cam = np.linalg.inv(camera_mat)
        pt_c = inv_cam @ pt_w  # 相机坐标系 (Habitat 约定: -Z 为前向, +X 为右, +Y 为上)

        # 距离
        dist = float(np.linalg.norm(pt_c[:3]))
        z_cam = float(-pt_c[2])  # 前向深度

        # 与主光轴 (-Z) 的偏角
        if dist > 1e-6:
            forward_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            dot = float(np.dot(pt_c[:3] / dist, forward_vec))
            dot = np.clip(dot, -1.0, 1.0)
            angle_deg = float(math.degrees(math.acos(dot)))
        else:
            angle_deg = 0.0

        intr = self.intrinsics
        if z_cam > 0.05:
            u = intr["cx"] + intr["fx"] * (float(pt_c[0]) / z_cam)
            v = intr["cy"] - intr["fy"] * (float(pt_c[1]) / z_cam)
            in_bounds = (0.0 <= u <= intr["width"]) and (0.0 <= v <= intr["height"])
            in_range = (self.clip_near <= z_cam <= self.clip_far)
            visible = bool(in_bounds and in_range)
        else:
            u = -1.0
            v = -1.0
            visible = False

        return {
            "visible": visible,
            "distance": round(dist, 3),
            "angle": round(angle_deg, 2),
            "pixel_coord": [round(u, 1), round(v, 1)],
            "in_frustum": visible,
        }

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
