#!/usr/bin/env python3
"""
动作感知 3D 骨架时序提取器 —— pose_extraction.py
=============================================

职责：
    1. 接收连续 RGB 图像序列 List[np.ndarray]；
    2. 调用统一 `Pose3DEstimator` 逐帧提取 3D 骨架序列 (T, V, 3)；
    3. 调用 `SkeletonNormalizer` 执行根节点去中心与尺度归一化；
    4. 输出结构化的 (T, V, 3) 归一化姿态时序与置信度。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from ea_avs_mvp_v10.perception.pose3d_estimator import (
    BasePose3DEstimator,
    create_pose3d_estimator,
)
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer

logger = logging.getLogger("pose_extraction")


class SequencePoseExtractor:
    """序列 3D 骨架特征提取器。"""

    def __init__(
        self,
        estimator: Optional[BasePose3DEstimator] = None,
        estimator_type: str = "mediapipe",
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.skel_def = skel_def or get_skeleton_definition()
        self.estimator = estimator or create_pose3d_estimator(estimator_type=estimator_type, skel_def=self.skel_def)
        self.normalizer = SkeletonNormalizer(skel_def=self.skel_def)

    def extract_and_normalize(
        self,
        rgb_frames: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从多帧 RGB 序列提取并归一化 3D 骨架。

        Returns:
            normalized_skeletons: (T, V, 3) 归一化 3D 骨架序列
            confidences: (T, V) 置信度序列
        """
        raw_skeletons, confidences = self.estimator.estimate_sequence(rgb_frames)
        norm_skeletons = self.normalizer.normalize_sequence(raw_skeletons)
        return norm_skeletons.astype(np.float32), confidences.astype(np.float32)
