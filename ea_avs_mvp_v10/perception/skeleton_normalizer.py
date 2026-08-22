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
    4. 规范化输出 EstimatedSkeleton3D 中的 joints_3d_normalized 矩阵 (17, 3)。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)


class SkeletonNormalizer:
    """3D 骨架空间根节点平移与体型尺度归一化器。"""

    def __init__(self, default_scale_fallback: float = 0.5, eps: float = 1e-5):
        """
        Args:
            default_scale_fallback: 当躯干被完全遮挡时使用的默认参考尺度 (米)
            eps: 防止除以 0 的微小扰动值
        """
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
            skeleton: 输入的 EstimatedSkeleton3D (默认 COCO-17)
            use_world_coords: 是否使用世界坐标进行归一化 (默认使用相机系局部坐标)

        Returns:
            带有 joints_3d_normalized (17, 3) 属性的 EstimatedSkeleton3D
        """
        if use_world_coords:
            raw_joints = skeleton.joints_3d_world.copy()
        else:
            raw_joints = skeleton.joints_3d_cam.copy()

        conf = skeleton.confidence

        if skeleton.joint_format == "COCO17":
            # COCO-17 索引:
            # 5: left_shoulder, 6: right_shoulder
            # 11: left_hip, 12: right_hip
            left_hip, right_hip = raw_joints[11], raw_joints[12]
            conf_hips = (conf[11] + conf[12]) / 2.0

            if conf_hips >= 0.1 and np.linalg.norm(left_hip) > 0 and np.linalg.norm(right_hip) > 0:
                root_pos = (left_hip + right_hip) / 2.0
            else:
                # 跨部遮挡时退化为有效关节均值
                valid_idx = np.where(conf >= 0.2)[0]
                if len(valid_idx) > 0:
                    root_pos = np.mean(raw_joints[valid_idx], axis=0)
                else:
                    root_pos = np.zeros(3, dtype=np.float32)

            # 1. 根节点平移: p_i' = p_i - p_root
            centered_joints = raw_joints - root_pos

            # 2. 躯干尺度计算: ||shoulder_center - hip_center||
            left_sh, right_sh = raw_joints[5], raw_joints[6]
            sh_center = (left_sh + right_sh) / 2.0
            torso_vec = sh_center - root_pos
            torso_len = float(np.linalg.norm(torso_vec))

            if torso_len >= 0.1:
                scale = torso_len
            else:
                # 躯干过小或退化，采用整体有效关节点方差尺度
                valid_centered = centered_joints[conf >= 0.2]
                if len(valid_centered) > 0:
                    scale = float(np.max(np.linalg.norm(valid_centered, axis=1)))
                else:
                    scale = self.default_scale_fallback

            scale = max(scale, self.default_scale_fallback * 0.5)

            # 3. 尺度归一化: p_i'' = p_i' / scale
            normalized_joints = centered_joints / (scale + self.eps)

            # 将完全无效/零置信度关节置 0
            invalid_mask = conf < 0.05
            normalized_joints[invalid_mask] = 0.0

        else:
            # 通用备用方案：以有效关节质心为原点，最大半径归一化
            valid_idx = np.where(conf >= 0.2)[0]
            if len(valid_idx) > 0:
                root_pos = np.mean(raw_joints[valid_idx], axis=0)
                centered = raw_joints - root_pos
                scale = float(np.max(np.linalg.norm(centered[valid_idx], axis=1))) + self.eps
                normalized_joints = centered / scale
            else:
                normalized_joints = raw_joints.copy()

        # 更新并返回
        skeleton.joints_3d_normalized = normalized_joints.astype(np.float32)
        return skeleton
