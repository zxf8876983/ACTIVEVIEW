"""
3D 骨架归一化器 —— skeleton_normalizer.py
======================================

职责：
    1. 根节点去中心化 (Root Normalization)：
       以骨盆/跨部中心 (hip_center) 为原点进行坐标平移：
       p_i' = p_i - p_root
    2. 躯干尺度归一化 (Scale Normalization)：
       基于躯干长度 (torso_length = ||shoulder_center - hip_center||) 消除不同个体身高与体型差异：
       p_i'' = p_i' / (scale + eps)
    3. 严格禁止相机朝向旋转归一化 (No Camera-Facing Rotation Normalization)：
       主动视角研究的核心依赖视角本身带来的运动学特征差异，严禁旋转消除视角变化；
    4. 规范化输出 EstimatedSkeleton3D 中的 joints_3d_normalized 矩阵。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)


class SkeletonNormalizer:
    """3D 骨架空间根节点平移与体型尺度归一化器。"""

    def __init__(self, default_scale_fallback: float = 0.5, eps: float = 1e-5):
        self.default_scale_fallback = float(default_scale_fallback)
        self.eps = float(eps)

    def normalize(
        self,
        skeleton: EstimatedSkeleton3D,
        use_world_coords: bool = False,
    ) -> EstimatedSkeleton3D:
        """
        对骨架 3D 关节执行根节点平移与尺度归一化。

        Args:
            skeleton: 输入的 EstimatedSkeleton3D (MediaPipe33 或 COCO-17)
            use_world_coords: 是否使用世界坐标进行归一化 (默认使用相机系局部坐标)

        Returns:
            带有 joints_3d_normalized 属性的 EstimatedSkeleton3D
        """
        if use_world_coords:
            raw_joints = skeleton.joints_3d_world.copy()
        else:
            raw_joints = skeleton.joints_3d_camera.copy()

        conf = skeleton.perception_confidence

        if skeleton.joint_format == "MediaPipe33":
            # MediaPipe 33 索引: 11: left_sh, 12: right_sh, 23: left_hip, 24: right_hip
            left_hip, right_hip = raw_joints[23], raw_joints[24]
            conf_hips = (conf[23] + conf[24]) / 2.0

            if conf_hips >= 0.1 and np.linalg.norm(left_hip) > 0 and np.linalg.norm(right_hip) > 0:
                root_pos = (left_hip + right_hip) / 2.0
            else:
                valid_idx = np.where(conf >= 0.2)[0]
                root_pos = np.mean(raw_joints[valid_idx], axis=0) if len(valid_idx) > 0 else np.zeros(3, dtype=np.float32)

            centered_joints = raw_joints - root_pos

            left_sh, right_sh = raw_joints[11], raw_joints[12]
            sh_center = (left_sh + right_sh) / 2.0
            torso_len = float(np.linalg.norm(sh_center - root_pos))
            scale = torso_len if torso_len >= 0.1 else self.default_scale_fallback
            scale = max(scale, self.default_scale_fallback * 0.5)

            normalized_joints = centered_joints / (scale + self.eps)

        elif skeleton.joint_format == "COCO17":
            # COCO-17 索引: 5: left_sh, 6: right_sh, 11: left_hip, 12: right_hip
            left_hip, right_hip = raw_joints[11], raw_joints[12]
            conf_hips = (conf[11] + conf[12]) / 2.0

            if conf_hips >= 0.1 and np.linalg.norm(left_hip) > 0 and np.linalg.norm(right_hip) > 0:
                root_pos = (left_hip + right_hip) / 2.0
            else:
                valid_idx = np.where(conf >= 0.2)[0]
                root_pos = np.mean(raw_joints[valid_idx], axis=0) if len(valid_idx) > 0 else np.zeros(3, dtype=np.float32)

            centered_joints = raw_joints - root_pos

            left_sh, right_sh = raw_joints[5], raw_joints[6]
            sh_center = (left_sh + right_sh) / 2.0
            torso_len = float(np.linalg.norm(sh_center - root_pos))
            scale = torso_len if torso_len >= 0.1 else self.default_scale_fallback
            scale = max(scale, self.default_scale_fallback * 0.5)

            normalized_joints = centered_joints / (scale + self.eps)

        else:
            # 通用备用方案
            valid_idx = np.where(conf >= 0.2)[0]
            if len(valid_idx) > 0:
                root_pos = np.mean(raw_joints[valid_idx], axis=0)
                centered = raw_joints - root_pos
                scale = float(np.max(np.linalg.norm(centered[valid_idx], axis=1))) + self.eps
                normalized_joints = centered / scale
            else:
                normalized_joints = raw_joints.copy()

        skeleton.joints_3d_normalized = normalized_joints.astype(np.float32)
        return skeleton
