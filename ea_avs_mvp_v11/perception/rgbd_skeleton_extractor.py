"""
成熟 RGB-D 人体 3D 骨架提取器 —— rgbd_skeleton_extractor.py
==========================================================

职责：
    1. 接收 RGB (HxWx3, uint8) + Depth (HxW, float32, 米) + 相机内参与位姿；
    2. 基于 MediaPipe 2D 姿态检测 + 深度图多尺度中值空间对齐 + 针孔逆投影模型恢复 Camera 3D 骨架；
    3. 重建的 3D 骨架保证与二维视觉观测的几何一致性 (The reconstructed 3D skeleton is geometrically consistent with the observed 2D keypoints)；
    4. 严格读取统一骨架拓扑规范 `configs/skeleton_definition.json`；
    5. 输出结构化的 EstimatedSkeleton3D 实体对象。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


class BaseRGBDSkeletonExtractor(ABC):
    """RGB-D 人体 3D 骨架提取器抽象基类。"""

    @abstractmethod
    def extract(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        camera_pose: Optional[CameraPose] = None,
    ) -> EstimatedSkeleton3D:
        """从 RGB-D 数据估计提取 3D 人体骨架。"""
        pass


class MediaPipeRGBDSkeletonExtractor(BaseRGBDSkeletonExtractor):
    """
    基于 MediaPipe 2D 关键点检测与深度图空间融合的几何一致性 3D 人体骨架提取器。

    特性：
        - 重建的 3D 骨架保证与二维视觉观测的几何一致性 (Geometrically consistent with 2D observations)；
        - 读取全局统一骨架拓扑 `configs/skeleton_definition.json`；
        - 自适应局部深度中值采样与骨骼运动学深度平滑；
        - 输出标准相机右手系 3D 坐标 (+X: 右, +Y: 上, +Z: 前/深度)。
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_patch_radius: int = 3,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.depth_patch_radius = depth_patch_radius
        self.skel_def = skel_def or get_skeleton_definition()
        self._pose = None

    def _lazy_init(self):
        if self._pose is not None:
            return
        import mediapipe as mp
        logger.info("Initializing MediaPipe Pose for RGB-D 3D Extraction (model_complexity=%d)...", self.model_complexity)
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def extract(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        camera_pose: Optional[CameraPose] = None,
    ) -> EstimatedSkeleton3D:
        self._lazy_init()

        if isinstance(rgb, Image.Image):
            rgb_arr = np.array(rgb)
        else:
            rgb_arr = rgb

        h, w = rgb_arr.shape[:2]
        depth_arr = np.asarray(depth, dtype=np.float32)

        # 1. 执行 MediaPipe 2D 姿态检测
        results = self._pose.process(rgb_arr)
        num_joints = self.skel_def.joint_num

        if not results.pose_landmarks:
            logger.debug("MediaPipe: No human pose detected in frame.")
            return EstimatedSkeleton3D(
                joint_format=self.skel_def.backend,
                joints_3d_camera=np.zeros((num_joints, 3), dtype=np.float32),
                joints_3d_world=np.zeros((num_joints, 3), dtype=np.float32),
                joints_2d=np.zeros((num_joints, 2), dtype=np.float32),
                perception_confidence=np.zeros(num_joints, dtype=np.float32),
                uncertainty_mask=np.ones(num_joints, dtype=bool),
                part_confidence={k: 0.0 for k in self.skel_def.part_groups.keys()},
                joint_names=self.skel_def.joint_names,
            )

        landmarks = results.pose_landmarks.landmark

        # 2. 提取 2D 像素坐标与 2D 检测置信度
        joints_2d = np.zeros((num_joints, 2), dtype=np.float32)
        conf_2d = np.zeros(num_joints, dtype=np.float32)
        for i in range(num_joints):
            lm = landmarks[i]
            joints_2d[i] = [float(lm.x * w), float(lm.y * h)]
            conf_2d[i] = float(lm.visibility)

        # 3. 采样各关节处的深度测量值 (局部邻域自适应滤波)
        depth_values = np.zeros(num_joints, dtype=np.float32)
        conf_depth = np.zeros(num_joints, dtype=np.float32)
        r = self.depth_patch_radius

        for i in range(num_joints):
            ui = int(np.clip(joints_2d[i, 0], 0, w - 1))
            vi = int(np.clip(joints_2d[i, 1], 0, h - 1))

            u_min, u_max = max(0, ui - r), min(w, ui + r + 1)
            v_min, v_max = max(0, vi - r), min(h, vi + r + 1)

            patch = depth_arr[v_min:v_max, u_min:u_max]
            valid_pixels = patch[(patch >= 0.2) & (patch <= 8.0) & (~np.isnan(patch))]

            if len(valid_pixels) > 0:
                z = float(np.median(valid_pixels))
                p_std = float(np.std(valid_pixels))
                c_z = max(0.2, min(1.0, 1.0 - (p_std / 0.35)))
            else:
                raw_z = float(depth_arr[vi, ui])
                if 0.2 <= raw_z <= 8.0 and not np.isnan(raw_z):
                    z = raw_z
                    c_z = 0.5
                else:
                    z = 0.0
                    c_z = 0.0

            depth_values[i] = z
            conf_depth[i] = c_z

        # 4. 骨盆根节点深度与未命中关节深度插值平滑
        root_indices = self.skel_def.root_joints
        valid_root_depths = [depth_values[idx] for idx in root_indices if depth_values[idx] > 0]
        z_root = float(np.median(valid_root_depths)) if len(valid_root_depths) > 0 else 2.0

        # 对无效深度关节赋予合理的骨骼连续性深度
        for i in range(num_joints):
            if depth_values[i] <= 0.0:
                parent_idx = self.skel_def.joints[i].parent
                if parent_idx is not None and depth_values[parent_idx] > 0:
                    depth_values[i] = depth_values[parent_idx]
                else:
                    depth_values[i] = z_root
                conf_depth[i] = 0.2

        # 5. 针孔相机反投影到相机坐标系 (X: 右, Y: 上, Z: 前)
        if camera_pose and camera_pose.intrinsics:
            fx = float(camera_pose.intrinsics.fx)
            fy = float(camera_pose.intrinsics.fy)
            cx = float(camera_pose.intrinsics.cx)
            cy = float(camera_pose.intrinsics.cy)
        else:
            fx = fy = float(w / 2.0)
            cx = float(w / 2.0)
            cy = float(h / 2.0)

        joints_3d_cam = np.zeros((num_joints, 3), dtype=np.float32)
        for i in range(num_joints):
            u, v = joints_2d[i]
            z = depth_values[i]
            x_cam = (u - cx) * z / fx
            y_cam = -(v - cy) * z / fy  # 计算机视觉中 v 向下增长，因此相机坐标系向上为 -(v - cy)
            z_cam = z
            joints_3d_cam[i] = [x_cam, y_cam, z_cam]

        # 6. 计算世界坐标系 3D 骨架 (结合相机外参)
        if camera_pose and camera_pose.matrix_4x4 is not None:
            c2w = np.array(camera_pose.matrix_4x4, dtype=np.float32)
            homo_cam = np.hstack([joints_3d_cam, np.ones((num_joints, 1), dtype=np.float32)])
            homo_world = (c2w @ homo_cam.T).T
            joints_3d_world = homo_world[:, :3]
        else:
            joints_3d_world = joints_3d_cam.copy()

        # 7. 计算复合感知置信度与不确定性掩码
        conf_composite = conf_2d * conf_depth
        uncertainty_mask = conf_composite < 0.35

        part_conf: Dict[str, float] = {}
        for part_name, indices in self.skel_def.part_groups.items():
            sub_confs = [conf_composite[idx] for idx in indices]
            part_conf[part_name] = float(np.mean(sub_confs))

        return EstimatedSkeleton3D(
            joint_format=self.skel_def.backend,
            joints_3d_camera=joints_3d_cam.astype(np.float32),
            joints_3d_world=joints_3d_world.astype(np.float32),
            joints_2d=joints_2d.astype(np.float32),
            perception_confidence=conf_composite.astype(np.float32),
            uncertainty_mask=uncertainty_mask,
            part_confidence=part_conf,
            joint_names=self.skel_def.joint_names,
        )


class MockRGBDSkeletonExtractor(BaseRGBDSkeletonExtractor):
    """供单元测试与离线模拟的 Mock 3D 骨架提取器。"""

    def __init__(self, default_confidence: float = 0.95, skel_def: Optional[SkeletonDefinition] = None):
        self.default_confidence = default_confidence
        self.skel_def = skel_def or get_skeleton_definition()

    def extract(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        camera_pose: Optional[CameraPose] = None,
    ) -> EstimatedSkeleton3D:
        num_joints = self.skel_def.joint_num
        j_cam = np.zeros((num_joints, 3), dtype=np.float32)
        j_cam[:, 2] = 2.0  # 默认 2.0m 深度

        # 模拟生成合理的站立人体 3D 关键点
        j_cam[0] = [0.0, 0.45, 2.0]     # nose
        j_cam[11] = [-0.18, 0.25, 2.0]  # left_shoulder
        j_cam[12] = [0.18, 0.25, 2.0]   # right_shoulder
        j_cam[13] = [-0.25, -0.05, 2.0] # left_elbow
        j_cam[14] = [0.25, -0.05, 2.0]  # right_elbow
        j_cam[15] = [-0.28, -0.35, 2.0] # left_wrist
        j_cam[16] = [0.28, -0.35, 2.0]  # right_wrist
        j_cam[23] = [-0.12, -0.25, 2.0] # left_hip
        j_cam[24] = [0.12, -0.25, 2.0]  # right_hip
        j_cam[25] = [-0.12, -0.65, 2.0] # left_knee
        j_cam[26] = [0.12, -0.65, 2.0]  # right_knee
        j_cam[27] = [-0.12, -1.05, 2.0] # left_ankle
        j_cam[28] = [0.12, -1.05, 2.0]  # right_ankle

        confs = np.full(num_joints, self.default_confidence, dtype=np.float32)
        j_2d = np.ones((num_joints, 2), dtype=np.float32) * 240.0

        if camera_pose and camera_pose.matrix_4x4 is not None:
            c2w = np.array(camera_pose.matrix_4x4, dtype=np.float32)
            homo_cam = np.hstack([j_cam, np.ones((num_joints, 1), dtype=np.float32)])
            j_world = (c2w @ homo_cam.T).T[:, :3]
        else:
            j_world = j_cam.copy()

        return EstimatedSkeleton3D(
            joint_format=self.skel_def.backend,
            joints_3d_camera=j_cam,
            joints_3d_world=j_world,
            joints_2d=j_2d,
            perception_confidence=confs,
            uncertainty_mask=confs < 0.35,
            part_confidence={k: self.default_confidence for k in self.skel_def.part_groups.keys()},
            joint_names=self.skel_def.joint_names,
        )


def RGBDSkeletonExtractor(backend: str = "mediapipe", **kwargs) -> BaseRGBDSkeletonExtractor:
    """RGB-D 骨架提取器工厂函数。"""
    if backend in ["mediapipe", "mediapipe_33"]:
        return MediaPipeRGBDSkeletonExtractor(**kwargs)
    elif backend == "mock":
        return MockRGBDSkeletonExtractor(**kwargs)
    else:
        raise ValueError(f"Unsupported RGB-D Skeleton Extractor backend: {backend}")
