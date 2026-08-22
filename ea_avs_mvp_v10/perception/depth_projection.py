"""
深度反投影与 3D 关节几何恢复 —— depth_projection.py
===================================================

职责：
    1. 接收 2D 关键点坐标 (u, v)、深度图与相机几何内参 (fx, fy, cx, cy)；
    2. 执行局部窗口邻域自适应滤波 (Patch Depth Filtering)，剔除边缘深度混叠与非正值异常；
    3. 基于针孔相机逆投影模型恢复相机坐标系下的 3D 关节坐标：
       X_cam = (u - cx) * Z / fx
       Y_cam = (v - cy) * Z / fy
       Z_cam = depth(u, v)
    4. 结合相机外参矩阵 (Camera Extrinsics) 将关节映射至世界坐标系；
    5. 计算局部深度一致性置信度 (depth_confidence) 与有效掩码。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose

logger = logging.getLogger(__name__)


@dataclass
class DepthProjectionResult:
    """深度反投影 3D 姿态结果容器。"""
    joints_3d_cam: np.ndarray          # 形状 (N, 3), 相机坐标系三维坐标 (米)
    joints_3d_world: np.ndarray        # 形状 (N, 3), 世界坐标系三维坐标 (米)
    depth_values: np.ndarray           # 形状 (N,), 各关键点采样深度 Z (米)
    depth_confidence: np.ndarray       # 形状 (N,), 深度局部平滑与有效性置信度 [0.0, 1.0]
    valid_mask: np.ndarray             # 形状 (N,), bool 是否为有效深度几何

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joints_3d_cam": self.joints_3d_cam.tolist(),
            "joints_3d_world": self.joints_3d_world.tolist(),
            "depth_values": self.depth_values.tolist(),
            "depth_confidence": self.depth_confidence.tolist(),
            "valid_mask": self.valid_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthProjectionResult":
        return cls(
            joints_3d_cam=np.array(data["joints_3d_cam"], dtype=np.float32),
            joints_3d_world=np.array(data["joints_3d_world"], dtype=np.float32),
            depth_values=np.array(data["depth_values"], dtype=np.float32),
            depth_confidence=np.array(data["depth_confidence"], dtype=np.float32),
            valid_mask=np.array(data["valid_mask"], dtype=bool),
        )


class DepthProjector:
    """深度图逆投影与 3D 关节重建器。"""

    def __init__(
        self,
        patch_radius: int = 2,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        max_patch_std: float = 0.35,
    ):
        """
        Args:
            patch_radius: 采样邻域半径 (窗口大小为 2*r+1，默认 5x5)
            min_depth: 最小有效深度 (米)
            max_depth: 最大有效深度 (米)
            max_patch_std: 允许的最大邻域标准差 (超过则降低深度置信度)
        """
        self.patch_radius = int(patch_radius)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_patch_std = float(max_patch_std)

    def project_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pose: Optional[CameraPose] = None,
    ) -> DepthProjectionResult:
        """
        将 2D 像素坐标数组反投影为相机系与世界系 3D 坐标。

        Args:
            keypoints_2d: (N, 2) 关键点像素坐标 [u, v]
            depth_map: (H, W) 深度图 (float32, 米)
            intrinsics: 相机内参 (fx, fy, cx, cy)
            camera_pose: 相机外参位姿 (可选，用于世界坐标转换)

        Returns:
            DepthProjectionResult
        """
        kpts = np.asarray(keypoints_2d, dtype=np.float32)
        n_joints = kpts.shape[0]
        h, w = depth_map.shape[:2]

        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.cx)
        cy = float(intrinsics.cy)

        joints_3d_cam = np.zeros((n_joints, 3), dtype=np.float32)
        depth_values = np.zeros(n_joints, dtype=np.float32)
        depth_conf = np.zeros(n_joints, dtype=np.float32)
        valid_mask = np.zeros(n_joints, dtype=bool)

        r = self.patch_radius

        for i in range(n_joints):
            u, v = kpts[i, 0], kpts[i, 1]

            # 边界检查
            if np.isnan(u) or np.isnan(v) or u < 0 or u >= w or v < 0 or v >= h:
                depth_values[i] = 0.0
                depth_conf[i] = 0.0
                valid_mask[i] = False
                continue

            ui = int(round(u))
            vi = int(round(v))

            # 提取局部窗口
            u_min = max(0, ui - r)
            u_max = min(w, ui + r + 1)
            v_min = max(0, vi - r)
            v_max = min(h, vi + r + 1)

            patch = depth_map[v_min:v_max, u_min:u_max]
            # 过滤有效深度像素
            valid_pixels = patch[(patch >= self.min_depth) & (patch <= self.max_depth) & (~np.isnan(patch))]

            if len(valid_pixels) == 0:
                # 无效深度 (例如落在背景虚空或遮挡极小边缘)
                z = float(depth_map[vi, ui]) if (0 <= vi < h and 0 <= ui < w) else 0.0
                if z < self.min_depth or z > self.max_depth or np.isnan(z):
                    z = 0.0
                    c_z = 0.0
                    is_valid = False
                else:
                    c_z = 0.5
                    is_valid = True
            else:
                # 采用中值深度抵抗噪声
                z = float(np.median(valid_pixels))
                patch_std = float(np.std(valid_pixels))
                # 邻域一致性置信度
                c_z = max(0.0, min(1.0, 1.0 - (patch_std / self.max_patch_std)))
                is_valid = True

            depth_values[i] = z
            depth_conf[i] = c_z
            valid_mask[i] = is_valid

            if is_valid and z > 0.0:
                # 针孔反投影标准公式
                x_cam = (u - cx) * z / fx
                y_cam = (v - cy) * z / fy
                z_cam = z
                joints_3d_cam[i] = [x_cam, y_cam, z_cam]
            else:
                joints_3d_cam[i] = [0.0, 0.0, 0.0]

        # 计算世界坐标系转换
        if camera_pose is not None and camera_pose.matrix_4x4 is not None:
            c2w = np.array(camera_pose.matrix_4x4, dtype=np.float32)
            # 齐次坐标变换
            ones = np.ones((n_joints, 1), dtype=np.float32)
            homo_cam = np.hstack([joints_3d_cam, ones])  # (N, 4)
            homo_world = (c2w @ homo_cam.T).T            # (N, 4)
            joints_3d_world = homo_world[:, :3]
            # 无效关节置 0
            joints_3d_world[~valid_mask] = 0.0
        else:
            joints_3d_world = joints_3d_cam.copy()

        return DepthProjectionResult(
            joints_3d_cam=joints_3d_cam,
            joints_3d_world=joints_3d_world,
            depth_values=depth_values,
            depth_confidence=depth_conf,
            valid_mask=valid_mask,
        )
