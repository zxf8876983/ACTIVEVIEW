"""
骨架拓扑适配器接口 —— skeleton_adapter.py
======================================

职责：
    1. 提供可选的骨架拓扑适配接口 (COCO-17 -> NTU-25 / 16-Joint Kinematic Tree 等)；
    2. 供未来 Phase 3 (ST-GCN / 动作识别模型) 按需适配特定图卷积拓扑；
    3. Phase 2 默认保持原生 COCO-17 格式，不执行默认转换。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


class COCO17ToNTU25Adapter(BaseSkeletonAdapter):
    """将 COCO-17 关键点插值/映射为 NTU-25 拓扑结构 (供 ST-GCN NTU 预训练模型适配)。"""

    def adapt(self, skeleton: EstimatedSkeleton3D) -> EstimatedSkeleton3D:
        if skeleton.joint_format != "COCO17":
            logger.warning("Input skeleton format is %s, expected COCO17", skeleton.joint_format)

        j_cam = skeleton.joints_3d_cam      # (17, 3)
        j_world = skeleton.joints_3d_world  # (17, 3)
        conf = skeleton.confidence          # (17,)

        ntu_cam = np.zeros((25, 3), dtype=np.float32)
        ntu_world = np.zeros((25, 3), dtype=np.float32)
        ntu_conf = np.zeros(25, dtype=np.float32)

        # 0: base of spine (left_hip 11 + right_hip 12 center)
        ntu_cam[0] = (j_cam[11] + j_cam[12]) / 2.0
        ntu_world[0] = (j_world[11] + j_world[12]) / 2.0
        ntu_conf[0] = (conf[11] + conf[12]) / 2.0

        # 20: spine_shoulder (left_shoulder 5 + right_shoulder 6 center)
        ntu_cam[20] = (j_cam[5] + j_cam[6]) / 2.0
        ntu_world[20] = (j_world[5] + j_world[6]) / 2.0
        ntu_conf[20] = (conf[5] + conf[6]) / 2.0

        # 1: middle_of_spine (base_of_spine 0 与 spine_shoulder 20 center)
        ntu_cam[1] = (ntu_cam[0] + ntu_cam[20]) / 2.0
        ntu_world[1] = (ntu_world[0] + ntu_world[20]) / 2.0
        ntu_conf[1] = (ntu_conf[0] + ntu_conf[20]) / 2.0

        # 2: neck
        ntu_cam[2] = ntu_cam[20]
        ntu_world[2] = ntu_world[20]
        ntu_conf[2] = ntu_conf[20]

        # 3: head (nose 0)
        ntu_cam[3] = j_cam[0]
        ntu_world[3] = j_world[0]
        ntu_conf[3] = conf[0]

        # 直接肢体对应
        direct_map = [
            (4, 5),   # left_shoulder <- COCO 5
            (5, 7),   # left_elbow    <- COCO 7
            (6, 9),   # left_wrist    <- COCO 9
            (7, 9),   # left_hand     <- COCO 9 (approx)
            (8, 6),   # right_shoulder<- COCO 6
            (9, 8),   # right_elbow   <- COCO 8
            (10, 10), # right_wrist   <- COCO 10
            (11, 10), # right_hand    <- COCO 10 (approx)
            (12, 11), # left_hip      <- COCO 11
            (13, 13), # left_knee     <- COCO 13
            (14, 15), # left_ankle    <- COCO 15
            (15, 15), # left_foot     <- COCO 15 (approx)
            (16, 12), # right_hip     <- COCO 12
            (17, 14), # right_knee    <- COCO 14
            (18, 16), # right_ankle   <- COCO 16
            (19, 16), # right_foot    <- COCO 16 (approx)
            (21, 9),  # left_hand_tip <- COCO 9
            (22, 9),  # left_thumb    <- COCO 9
            (23, 10), # right_hand_tip<- COCO 10
            (24, 10), # right_thumb   <- COCO 10
        ]

        for ntu_idx, coco_idx in direct_map:
            ntu_cam[ntu_idx] = j_cam[coco_idx]
            ntu_world[ntu_idx] = j_world[coco_idx]
            ntu_conf[ntu_idx] = conf[coco_idx]

        return EstimatedSkeleton3D(
            joint_format="NTU25",
            joints_3d_cam=ntu_cam,
            joints_3d_world=ntu_world,
            joints_2d=np.zeros((25, 2), dtype=np.float32),
            confidence=ntu_conf,
            occluded_mask=ntu_conf < 0.35,
            part_confidence={},
            joint_names=list(NTU_25_JOINT_NAMES),
        )
