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
    3. 严格禁止相机朝向旋转归一化 (No Camera-Facing Rotation Normalization)；
    4. 统一从 `configs/skeleton_definition.json` 获取根节点与躯干关键点索引。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .skeleton_converter import EstimatedSkeleton3D
from .skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


class SkeletonNormalizer:
    """3D 骨架空间根节点平移与体型尺度归一化器。"""

    def __init__(
        self,
        default_scale_fallback: float = 0.5,
        eps: float = 1e-5,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.default_scale_fallback = float(default_scale_fallback)
        self.eps = float(eps)
        self.skel_def = skel_def or get_skeleton_definition()

    def normalize(
        self,
        skeleton: EstimatedSkeleton3D,
        use_world_coords: bool = False,
    ) -> EstimatedSkeleton3D:
        """
        对骨架 3D 关节执行根节点平移与尺度归一化。

        Args:
            skeleton: 输入的 EstimatedSkeleton3D
            use_world_coords: 是否使用世界坐标进行归一化 (默认使用相机系局部坐标)

        Returns:
            带有 joints_3d_normalized 属性的 EstimatedSkeleton3D
        """
        if use_world_coords:
            raw_joints = skeleton.joints_3d_world.copy()
        else:
            raw_joints = skeleton.joints_3d_camera.copy()

        conf = skeleton.perception_confidence

        root_indices = self.skel_def.root_joints
        torso_indices = self.skel_def.torso_joints

        # 计算骨盆根节点位置
        valid_root_idx = [idx for idx in root_indices if conf[idx] >= 0.2 and np.linalg.norm(raw_joints[idx]) > 0.01]
        if len(valid_root_idx) > 0:
            root_pos = np.mean(raw_joints[valid_root_idx], axis=0)
        else:
            valid_idx = np.where(conf >= 0.2)[0]
            root_pos = np.mean(raw_joints[valid_idx], axis=0) if len(valid_idx) > 0 else np.zeros(3, dtype=np.float32)

        # 根节点去中心化平移
        centered_joints = raw_joints - root_pos

        # 计算躯干尺度 (双肩中心与骨盆中心距离)
        # 双肩索引在 mediapipe 中为 11, 12，在 COCO 中为 5, 6
        shoulder_indices = [idx for idx in torso_indices if idx not in root_indices]
        valid_sh_idx = [idx for idx in shoulder_indices if conf[idx] >= 0.2 and np.linalg.norm(raw_joints[idx]) > 0.01]

        if len(valid_sh_idx) > 0:
            sh_center = np.mean(raw_joints[valid_sh_idx], axis=0)
            torso_len = float(np.linalg.norm(sh_center - root_pos))
            scale = torso_len if torso_len >= 0.1 else self.default_scale_fallback
        else:
            scale = self.default_scale_fallback

        scale = max(scale, self.default_scale_fallback * 0.5)

        # 尺度归一化
        normalized_joints = centered_joints / (scale + self.eps)

        skeleton.joints_3d_normalized = normalized_joints.astype(np.float32)
        return skeleton
