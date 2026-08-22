"""
成熟 RGB-D 人体 3D 骨架提取器 —— rgbd_skeleton_extractor.py
==========================================================

职责：
    1. 接收 RGB (HxWx3, uint8) + Depth (HxW, float32, 米) + 相机位姿与内参；
    2. 基于成熟工业级 RGB-D/3D 姿态估计器 (MediaPipe BlazePose 3D / 运动学约束求解器) 提取具备真实人体解剖学刚性与对称性的 3D 骨架；
    3. 杜绝手工单点反投影导致的骨骼拉伸变形、深度抖动与异常撕裂；
    4. 默认保留 Extractor 原生关节拓扑 (MediaPipe 33 关键点，带官方骨骼连接)；
    5. 输出结构化的 EstimatedSkeleton3D 实体对象。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)

# MediaPipe 33 关键点标准名称定义
MEDIAPIPE_33_KEYPOINTS: List[str] = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]

# MediaPipe 官方标准运动学骨骼连接拓扑
MEDIAPIPE_33_SKELETON_PAIRS: List[Tuple[int, int]] = [
    # 面部与头部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 上肢与躯干
    (11, 12),              # 双肩
    (11, 13), (13, 15),    # 左臂
    (15, 17), (15, 19), (15, 21), (17, 19),  # 左手
    (12, 14), (14, 16),    # 右臂
    (16, 18), (16, 20), (16, 22), (18, 20),  # 右手
    (11, 23), (12, 24),    # 躯干侧边
    (23, 24),              # 骨盆跨部
    # 下肢
    (23, 25), (25, 27),    # 左腿
    (27, 29), (29, 31), (27, 31),  # 左脚
    (24, 26), (26, 28),    # 右腿
    (28, 30), (30, 32), (28, 32),  # 右脚
]

# 部位分组 (MediaPipe 33)
MEDIAPIPE_BODY_PART_GROUPS: Dict[str, List[int]] = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "torso": [11, 12, 23, 24],
    "left_arm": [11, 13, 15, 17, 19, 21],
    "right_arm": [12, 14, 16, 18, 20, 22],
    "left_leg": [23, 25, 27, 29, 31],
    "right_leg": [24, 26, 28, 30, 32],
}


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
    基于 MediaPipe BlazePose 3D 运动学模型与深度图空间融合的 3D 人体骨架提取器。

    特性：
        - 输出 33 关键点标准人体运动学拓扑；
        - 直接输出解剖学真实比例的 3D 骨架，根节点位于骨盆中心；
        - 结合深度图恢复局部相机坐标系真实测距与空间定位；
        - 肢体长度稳定对称，无单点逆投影深度抖动与撕裂。
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_patch_radius: int = 4,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.depth_patch_radius = depth_patch_radius
        self._pose = None

    def _lazy_init(self):
        if self._pose is not None:
            return
        import mediapipe as mp
        logger.info("Initializing MediaPipe BlazePose 3D (model_complexity=%d)...", self.model_complexity)
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
        import cv2

        if isinstance(rgb, Image.Image):
            rgb_arr = np.array(rgb)
        else:
            rgb_arr = rgb

        h, w = rgb_arr.shape[:2]
        depth_arr = np.asarray(depth, dtype=np.float32)

        # 执行 3D 人体姿态估计
        results = self._pose.process(rgb_arr)

        num_joints = len(MEDIAPIPE_33_KEYPOINTS)

        if not results.pose_world_landmarks or not results.pose_landmarks:
            logger.debug("MediaPipe: No human pose detected in frame.")
            return EstimatedSkeleton3D(
                joint_format="MediaPipe33",
                joints_3d_camera=np.zeros((num_joints, 3), dtype=np.float32),
                joints_3d_world=np.zeros((num_joints, 3), dtype=np.float32),
                joints_2d=np.zeros((num_joints, 2), dtype=np.float32),
                perception_confidence=np.zeros(num_joints, dtype=np.float32),
                uncertainty_mask=np.ones(num_joints, dtype=bool),
                part_confidence={k: 0.0 for k in MEDIAPIPE_BODY_PART_GROUPS.keys()},
                joint_names=list(MEDIAPIPE_33_KEYPOINTS),
            )

        world_landmarks = results.pose_world_landmarks.landmark
        image_landmarks = results.pose_landmarks.landmark

        joints_2d = np.zeros((num_joints, 2), dtype=np.float32)
        joints_3d_rel = np.zeros((num_joints, 3), dtype=np.float32)
        confs = np.zeros(num_joints, dtype=np.float32)

        for i in range(num_joints):
            img_lm = image_landmarks[i]
            w_lm = world_landmarks[i]

            u = float(img_lm.x * w)
            v = float(img_lm.y * h)
            joints_2d[i] = [u, v]

            # MediaPipe world landmarks: x: right, y: down, z: forward (meters from hip)
            # 转换为标准机器人相机系右手系 (X: 右, Y: 上, Z: 前):
            joints_3d_rel[i] = [w_lm.x, -w_lm.y, w_lm.z]
            confs[i] = float(img_lm.visibility)

        # 2. 结合深度图恢复骨盆根节点 (hip center) 的绝对相机距离 Z_root
        # 骨盆中点在图像中的位置
        left_hip_2d = joints_2d[23]
        right_hip_2d = joints_2d[24]
        root_u = int(np.clip((left_hip_2d[0] + right_hip_2d[0]) / 2.0, 0, w - 1))
        root_v = int(np.clip((left_hip_2d[1] + right_hip_2d[1]) / 2.0, 0, h - 1))

        r = self.depth_patch_radius
        u_min, u_max = max(0, root_u - r), min(w, root_u + r + 1)
        v_min, v_max = max(0, root_v - r), min(h, root_v + r + 1)

        patch = depth_arr[v_min:v_max, u_min:u_max]
        valid_depths = patch[(patch >= 0.2) & (patch <= 8.0) & (~np.isnan(patch))]

        if len(valid_depths) > 0:
            z_root = float(np.median(valid_depths))
        else:
            # 备用方案：使用全躯干区域深度
            torso_u = int(np.clip((joints_2d[11, 0] + joints_2d[12, 0] + root_u) / 3.0, 0, w - 1))
            torso_v = int(np.clip((joints_2d[11, 1] + joints_2d[12, 1] + root_v) / 3.0, 0, h - 1))
            z_val = float(depth_arr[torso_v, torso_u])
            z_root = z_val if (0.2 <= z_val <= 8.0) else 2.0

        # 相机几何内参反解根节点在相机系下的中心 (X_root, Y_root, Z_root)
        if camera_pose and camera_pose.intrinsics:
            fx = float(camera_pose.intrinsics.fx)
            fy = float(camera_pose.intrinsics.fy)
            cx = float(camera_pose.intrinsics.cx)
            cy = float(camera_pose.intrinsics.cy)
            x_root = (root_u - cx) * z_root / fx
            y_root = -(root_v - cy) * z_root / fy
        else:
            x_root = (root_u - w / 2.0) * z_root / (w / 2.0)
            y_root = -(root_v - h / 2.0) * z_root / (h / 2.0)

        # 3. 生成相机系绝对 3D 骨架
        joints_3d_camera = joints_3d_rel.copy()
        joints_3d_camera[:, 0] += x_root
        joints_3d_camera[:, 1] += y_root
        joints_3d_camera[:, 2] += z_root

        # 4. 生成世界系 3D 骨架
        if camera_pose and camera_pose.matrix_4x4 is not None:
            c2w = np.array(camera_pose.matrix_4x4, dtype=np.float32)
            ones = np.ones((num_joints, 1), dtype=np.float32)
            homo_cam = np.hstack([joints_3d_camera, ones])
            homo_world = (c2w @ homo_cam.T).T
            joints_3d_world = homo_world[:, :3]
        else:
            joints_3d_world = joints_3d_camera.copy()

        # 5. 不确定性掩码与部位平均置信度
        uncertainty_mask = confs < 0.4
        part_conf: Dict[str, float] = {}
        for part_name, joint_indices in MEDIAPIPE_BODY_PART_GROUPS.items():
            sub_confs = [confs[j] for j in joint_indices]
            part_conf[part_name] = float(np.mean(sub_confs))

        return EstimatedSkeleton3D(
            joint_format="MediaPipe33",
            joints_3d_camera=joints_3d_camera.astype(np.float32),
            joints_3d_world=joints_3d_world.astype(np.float32),
            joints_2d=joints_2d.astype(np.float32),
            perception_confidence=confs.astype(np.float32),
            uncertainty_mask=uncertainty_mask,
            part_confidence=part_conf,
            joint_names=list(MEDIAPIPE_33_KEYPOINTS),
        )


class MockRGBDSkeletonExtractor(BaseRGBDSkeletonExtractor):
    """供单元测试与离线模拟的 Mock 3D 骨架提取器。"""

    def __init__(self, default_confidence: float = 0.95):
        self.default_confidence = default_confidence

    def extract(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        camera_pose: Optional[CameraPose] = None,
    ) -> EstimatedSkeleton3D:
        num_joints = len(MEDIAPIPE_33_KEYPOINTS)
        j_cam = np.zeros((num_joints, 3), dtype=np.float32)
        j_cam[:, 2] = 2.0  # 默认 2.0m 深度

        # 模拟生成合理的站立人体 3D 关键点
        j_cam[0] = [0.0, 0.65, 2.0]     # nose
        j_cam[11] = [-0.2, 0.45, 2.0]   # left_shoulder
        j_cam[12] = [0.2, 0.45, 2.0]    # right_shoulder
        j_cam[13] = [-0.3, 0.15, 2.0]   # left_elbow
        j_cam[14] = [0.3, 0.15, 2.0]    # right_elbow
        j_cam[15] = [-0.35, -0.15, 2.0] # left_wrist
        j_cam[16] = [0.35, -0.15, 2.0]  # right_wrist
        j_cam[23] = [-0.15, 0.0, 2.0]   # left_hip
        j_cam[24] = [0.15, 0.0, 2.0]    # right_hip
        j_cam[25] = [-0.15, -0.45, 2.0] # left_knee
        j_cam[26] = [0.15, -0.45, 2.0]  # right_knee
        j_cam[27] = [-0.15, -0.9, 2.0]  # left_ankle
        j_cam[28] = [0.15, -0.9, 2.0]   # right_ankle

        confs = np.full(num_joints, self.default_confidence, dtype=np.float32)
        j_2d = np.ones((num_joints, 2), dtype=np.float32) * 240.0

        if camera_pose and camera_pose.matrix_4x4 is not None:
            c2w = np.array(camera_pose.matrix_4x4, dtype=np.float32)
            homo_cam = np.hstack([j_cam, np.ones((num_joints, 1), dtype=np.float32)])
            j_world = (c2w @ homo_cam.T).T[:, :3]
        else:
            j_world = j_cam.copy()

        return EstimatedSkeleton3D(
            joint_format="MediaPipe33",
            joints_3d_camera=j_cam,
            joints_3d_world=j_world,
            joints_2d=j_2d,
            perception_confidence=confs,
            uncertainty_mask=confs < 0.35,
            part_confidence={k: self.default_confidence for k in MEDIAPIPE_BODY_PART_GROUPS.keys()},
            joint_names=list(MEDIAPIPE_33_KEYPOINTS),
        )


def RGBDSkeletonExtractor(backend: str = "mediapipe", **kwargs) -> BaseRGBDSkeletonExtractor:
    """RGB-D 骨架提取器工厂函数。"""
    if backend == "mediapipe":
        return MediaPipeRGBDSkeletonExtractor(**kwargs)
    elif backend == "mock":
        return MockRGBDSkeletonExtractor(**kwargs)
    else:
        raise ValueError(f"Unsupported RGB-D Skeleton Extractor backend: {backend}")
