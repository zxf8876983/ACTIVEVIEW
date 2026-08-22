"""
骨架拓扑适配器接口 —— skeleton_adapter.py
======================================

职责：
    1. 提供可选的骨架拓扑适配接口 (MediaPipe-33 -> COCO-17 / NTU-25 等)；
    2. 供未来 Phase 3 (ST-GCN / 动作识别模型) 按需适配特定图卷积拓扑；
    3. Phase 2 默认保持 Extractor 原生格式 (MediaPipe-33)，不执行强制转换。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .pose_estimator import COCO_KEYPOINTS
from .skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)

# NTU RGB+D 25 关键点标准定义
NTU_25_JOINT_NAMES: List[str] = [
    "base_of_spine",     # 0 (pelvis center)
    "middle_of_spine",   # 1
    "neck",              # 2
    "head",              # 3
    "left_shoulder",     # 4
    "left_elbow",        # 5
    "left_wrist",        # 6
    "left_hand",         # 7
    "right_shoulder",    # 8
    "right_elbow",       # 9
    "right_wrist",       # 10
    "right_hand",        # 11
    "left_hip",          # 12
    "left_knee",         # 13
    "left_ankle",        # 14
    "left_foot",         # 15
    "right_hip",         # 16
    "right_knee",        # 17
    "right_ankle",       # 18
    "right_foot",        # 19
    "spine_shoulder",    # 20
    "left_hand_tip",     # 21
    "left_thumb",        # 22
    "right_hand_tip",    # 23
    "right_thumb",       # 24
]


class BaseSkeletonAdapter(ABC):
    """骨架格式适配器基类。"""

    @abstractmethod
    def adapt(self, skeleton: EstimatedSkeleton3D) -> EstimatedSkeleton3D:
        """将输入骨架转换为目标拓扑骨架。"""
        pass


class MediaPipe33ToCOCO17Adapter(BaseSkeletonAdapter):
    """将 MediaPipe 33 关键点标准映射为 COCO-17 拓扑。"""

    # MediaPipe -> COCO-17 映射字典
    MAP: List[Tuple[int, int]] = [
        (0, 0),    # nose
        (1, 2),    # left_eye
        (2, 5),    # right_eye
        (3, 7),    # left_ear
        (4, 8),    # right_ear
        (5, 11),   # left_shoulder
        (6, 12),   # right_shoulder
        (7, 13),   # left_elbow
        (8, 14),   # right_elbow
        (9, 15),   # left_wrist
        (10, 16),  # right_wrist
        (11, 23),  # left_hip
        (12, 24),  # right_hip
        (13, 25),  # left_knee
        (14, 26),  # right_knee
        (15, 27),  # left_ankle
        (16, 28),  # right_ankle
    ]

    def adapt(self, skeleton: EstimatedSkeleton3D) -> EstimatedSkeleton3D:
        j_cam = skeleton.joints_3d_camera
        j_world = skeleton.joints_3d_world
        j_2d = skeleton.joints_2d
        conf = skeleton.perception_confidence
        j_norm = skeleton.joints_3d_normalized

        coco_cam = np.zeros((17, 3), dtype=np.float32)
        coco_world = np.zeros((17, 3), dtype=np.float32)
        coco_2d = np.zeros((17, 2), dtype=np.float32)
        coco_conf = np.zeros(17, dtype=np.float32)
        coco_norm = np.zeros((17, 3), dtype=np.float32) if j_norm is not None else None

        for coco_idx, mp_idx in self.MAP:
            coco_cam[coco_idx] = j_cam[mp_idx]
            coco_world[coco_idx] = j_world[mp_idx]
            coco_2d[coco_idx] = j_2d[mp_idx]
            coco_conf[coco_idx] = conf[mp_idx]
            if coco_norm is not None:
                coco_norm[coco_idx] = j_norm[mp_idx]

        return EstimatedSkeleton3D(
            joint_format="COCO17",
            joints_3d_camera=coco_cam,
            joints_3d_world=coco_world,
            joints_2d=coco_2d,
            perception_confidence=coco_conf,
            uncertainty_mask=coco_conf < 0.35,
            part_confidence=skeleton.part_confidence,
            joints_3d_normalized=coco_norm,
            joint_names=list(COCO_KEYPOINTS),
        )


class MediaPipe33ToNTU25Adapter(BaseSkeletonAdapter):
    """将 MediaPipe 33 关键点映射为 NTU-25 拓扑结构。"""

    def adapt(self, skeleton: EstimatedSkeleton3D) -> EstimatedSkeleton3D:
        j_cam = skeleton.joints_3d_camera
        j_world = skeleton.joints_3d_world
        conf = skeleton.perception_confidence

        ntu_cam = np.zeros((25, 3), dtype=np.float32)
        ntu_world = np.zeros((25, 3), dtype=np.float32)
        ntu_conf = np.zeros(25, dtype=np.float32)

        # 0: base of spine (hip center)
        ntu_cam[0] = (j_cam[23] + j_cam[24]) / 2.0
        ntu_world[0] = (j_world[23] + j_world[24]) / 2.0
        ntu_conf[0] = (conf[23] + conf[24]) / 2.0

        # 20: spine_shoulder (shoulder center)
        ntu_cam[20] = (j_cam[11] + j_cam[12]) / 2.0
        ntu_world[20] = (j_world[11] + j_world[12]) / 2.0
        ntu_conf[20] = (conf[11] + conf[12]) / 2.0

        # 1: middle of spine
        ntu_cam[1] = (ntu_cam[0] + ntu_cam[20]) / 2.0
        ntu_world[1] = (ntu_world[0] + ntu_world[20]) / 2.0
        ntu_conf[1] = (ntu_conf[0] + ntu_conf[20]) / 2.0

        # 2: neck
        ntu_cam[2] = ntu_cam[20]
        ntu_world[2] = ntu_world[20]
        ntu_conf[2] = ntu_conf[20]

        # 3: head (nose)
        ntu_cam[3] = j_cam[0]
        ntu_world[3] = j_world[0]
        ntu_conf[3] = conf[0]

        # 直接肢体对应 (MediaPipe 33)
        direct_map = [
            (4, 11),  # left_shoulder
            (5, 13),  # left_elbow
            (6, 15),  # left_wrist
            (7, 19),  # left_hand (index)
            (8, 12),  # right_shoulder
            (9, 14),  # right_elbow
            (10, 16), # right_wrist
            (11, 20), # right_hand (index)
            (12, 23), # left_hip
            (13, 25), # left_knee
            (14, 27), # left_ankle
            (15, 31), # left_foot
            (16, 24), # right_hip
            (17, 26), # right_knee
            (18, 28), # right_ankle
            (19, 32), # right_foot
            (21, 17), # left_hand_tip (pinky)
            (22, 21), # left_thumb
            (23, 18), # right_hand_tip (pinky)
            (24, 22), # right_thumb
        ]

        for ntu_idx, mp_idx in direct_map:
            ntu_cam[ntu_idx] = j_cam[mp_idx]
            ntu_world[ntu_idx] = j_world[mp_idx]
            ntu_conf[ntu_idx] = conf[mp_idx]

        return EstimatedSkeleton3D(
            joint_format="NTU25",
            joints_3d_camera=ntu_cam,
            joints_3d_world=ntu_world,
            joints_2d=np.zeros((25, 2), dtype=np.float32),
            perception_confidence=ntu_conf,
            uncertainty_mask=ntu_conf < 0.35,
            part_confidence={},
            joint_names=list(NTU_25_JOINT_NAMES),
        )


class COCO17ToNTU25Adapter(BaseSkeletonAdapter):
    """将 COCO-17 关键点映射为 NTU-25 拓扑结构 (兼容旧模型接口)。"""

    def adapt(self, skeleton: EstimatedSkeleton3D) -> EstimatedSkeleton3D:
        j_cam = skeleton.joints_3d_camera
        j_world = skeleton.joints_3d_world
        conf = skeleton.perception_confidence

        ntu_cam = np.zeros((25, 3), dtype=np.float32)
        ntu_world = np.zeros((25, 3), dtype=np.float32)
        ntu_conf = np.zeros(25, dtype=np.float32)

        # 0: base of spine (hip center)
        ntu_cam[0] = (j_cam[11] + j_cam[12]) / 2.0
        ntu_world[0] = (j_world[11] + j_world[12]) / 2.0
        ntu_conf[0] = (conf[11] + conf[12]) / 2.0

        # 20: spine_shoulder (shoulder center)
        ntu_cam[20] = (j_cam[5] + j_cam[6]) / 2.0
        ntu_world[20] = (j_world[5] + j_world[6]) / 2.0
        ntu_conf[20] = (conf[5] + conf[6]) / 2.0

        # 1: middle of spine
        ntu_cam[1] = (ntu_cam[0] + ntu_cam[20]) / 2.0
        ntu_world[1] = (ntu_world[0] + ntu_world[20]) / 2.0
        ntu_conf[1] = (ntu_conf[0] + ntu_conf[20]) / 2.0

        # 2: neck
        ntu_cam[2] = ntu_cam[20]
        ntu_world[2] = ntu_world[20]
        ntu_conf[2] = ntu_conf[20]

        # 3: head (nose)
        ntu_cam[3] = j_cam[0]
        ntu_world[3] = j_world[0]
        ntu_conf[3] = conf[0]

        # 直接肢体对应
        direct_map = [
            (4, 5),   # left_shoulder
            (5, 7),   # left_elbow
            (6, 9),   # left_wrist
            (7, 9),   # left_hand (approx)
            (8, 6),   # right_shoulder
            (9, 8),   # right_elbow
            (10, 10), # right_wrist
            (11, 10), # right_hand (approx)
            (12, 11), # left_hip
            (13, 13), # left_knee
            (14, 15), # left_ankle
            (15, 15), # left_foot (approx)
            (16, 12), # right_hip
            (17, 14), # right_knee
            (18, 16), # right_ankle
            (19, 16), # right_foot (approx)
        ]

        for ntu_idx, coco_idx in direct_map:
            ntu_cam[ntu_idx] = j_cam[coco_idx]
            ntu_world[ntu_idx] = j_world[coco_idx]
            ntu_conf[ntu_idx] = conf[coco_idx]

        return EstimatedSkeleton3D(
            joint_format="NTU25",
            joints_3d_camera=ntu_cam,
            joints_3d_world=ntu_world,
            joints_2d=np.zeros((25, 2), dtype=np.float32),
            perception_confidence=ntu_conf,
            uncertainty_mask=ntu_conf < 0.35,
            part_confidence={},
            joint_names=list(NTU_25_JOINT_NAMES),
        )

