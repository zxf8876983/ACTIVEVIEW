"""
RGB-D 传感器与图像采集模块 —— rgbd_capture.py
============================================

职责：
    1. 封装 Habitat-Sim RGB (Color) 与 Depth 深度传感器；
    2. 实时捕获同步 RGB 图像 (uint8) 与高精度深度图 (float32, meters)；
    3. 提取相机几何内参 (CameraIntrinsics) 与世界坐标系位姿 (CameraPose, 4x4 变换矩阵)；
    4. 提供持久化保存方法 (RGB -> PNG, Depth -> NPY / Visual PNG, CameraPose -> JSON)。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose

logger = logging.getLogger(__name__)


class RGBDCapture:
    """机器人 RGB-D 传感器捕获与相机位姿提取器。"""

    def __init__(
        self,
        sim: Optional[Any] = None,
        sensor_cfg: Optional[Dict[str, Any]] = None,
        agent_id: int = 0,
    ):
        self.sim = sim
        self.sensor_cfg = sensor_cfg or {}
        self.agent_id = agent_id

        self.width = int(self.sensor_cfg.get("width", 640))
        self.height = int(self.sensor_cfg.get("height", 480))
        self.hfov_deg = float(self.sensor_cfg.get("hfov_deg", 90.0))
        self.camera_height = float(self.sensor_cfg.get("camera_height", 1.20))
        self.clip_near = float(self.sensor_cfg.get("clip_near", 0.01))
        self.clip_far = float(self.sensor_cfg.get("clip_far", 10.0))

        self.intrinsics = self._compute_intrinsics()

    def _compute_intrinsics(self) -> CameraIntrinsics:
        """从几何参数计算针孔相机内参。"""
        hfov_rad = math.radians(self.hfov_deg)
        fx = self.width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            hfov_deg=self.hfov_deg,
            clip_near=self.clip_near,
            clip_far=self.clip_far,
        )

    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraPose]:
        """
        从当前仿真环境中捕获一帧 RGB 与 Depth 数据及对应相机位姿。

        Returns:
            rgb: np.ndarray, shape (H, W, 3), dtype uint8
            depth: np.ndarray, shape (H, W), dtype float32 (meters)
            camera_pose: CameraPose 实例
        """
        if self.sim is None:
            # 离线模拟数据
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            depth = np.ones((self.height, self.width), dtype=np.float32) * 2.0
            cam_pose = CameraPose(
                position=[0.0, self.camera_height, 2.0],
                rotation_quat=[0.0, 0.0, 0.0, 1.0],
                yaw_deg=0.0,
                intrinsics=self.intrinsics,
                matrix_4x4=np.eye(4, dtype=np.float32).tolist(),
            )
            return rgb, depth, cam_pose

        obs = self.sim.get_sensor_observations(self.agent_id)

        # 1. 提取 RGB
        if "color_sensor" in obs:
            raw_rgb = obs["color_sensor"]
            if raw_rgb.shape[-1] == 4:
                rgb = raw_rgb[:, :, :3].astype(np.uint8)
            else:
                rgb = raw_rgb.astype(np.uint8)
        else:
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 2. 提取 Depth
        if "depth_sensor" in obs:
            raw_depth = obs["depth_sensor"]
            if raw_depth.ndim == 3 and raw_depth.shape[-1] == 1:
                depth = raw_depth[:, :, 0].astype(np.float32)
            else:
                depth = raw_depth.astype(np.float32)
        else:
            depth = np.ones((self.height, self.width), dtype=np.float32) * 2.0

        # 3. 提取相机外参和位姿
        ag = self.sim.get_agent(self.agent_id)
        ag_state = ag.get_state()

        if "color_sensor" in ag_state.sensor_states:
            s_state = ag_state.sensor_states["color_sensor"]
            cam_pos = [float(x) for x in s_state.position]
            q = s_state.rotation
            cam_quat = [float(q.x), float(q.y), float(q.z), float(q.w)]
        else:
            cam_pos = [float(x) for x in ag_state.position]
            q = ag_state.rotation
            cam_quat = [float(q.x), float(q.y), float(q.z), float(q.w)]

        # 计算底盘 yaw 角 (度)
        siny_cosp = 2 * (cam_quat[3] * cam_quat[1] + cam_quat[0] * cam_quat[2])
        cosy_cosp = 1 - 2 * (cam_quat[0] * cam_quat[0] + cam_quat[1] * cam_quat[1])
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad) % 360.0

        mat_4x4 = np.array(ag.scene_node.transformation, dtype=np.float32).tolist()

        cam_pose = CameraPose(
            position=cam_pos,
            rotation_quat=cam_quat,
            yaw_deg=yaw_deg,
            intrinsics=self.intrinsics,
            matrix_4x4=mat_4x4,
        )

        return rgb, depth, cam_pose

    @staticmethod
    def save_rgb_image(rgb_array: np.ndarray, save_path: Union[str, Path]) -> Path:
        """保存 RGB 图像为 PNG 格式。"""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(rgb_array)
        img.save(p)
        return p

    @staticmethod
    def save_depth_array(depth_array: np.ndarray, save_path: Union[str, Path]) -> Path:
        """保存原始深度浮点数组为 .npy 格式。"""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, depth_array.astype(np.float32))
        return p

    @staticmethod
    def save_depth_visual(depth_array: np.ndarray, save_path: Union[str, Path], clip_max: float = 5.0) -> Path:
        """保存归一化灰度深度可视化图为 PNG 格式。"""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        norm_d = np.clip(depth_array, 0.0, clip_max) / max(1e-4, clip_max)
        gray_img = (norm_d * 255.0).astype(np.uint8)
        img = Image.fromarray(gray_img)
        img.save(p)
        return p

    @staticmethod
    def save_camera_pose(camera_pose: CameraPose, save_path: Union[str, Path]) -> Path:
        """保存相机位姿与内参为 JSON 格式。"""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(camera_pose.to_dict(), f, indent=2, ensure_ascii=False)
        return p
