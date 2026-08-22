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
    4. 统一从 `configs/skeleton_definition.json` 获取根节点与躯干关键点索引；
    5. 支持单帧骨架对象归一化与批量时序矩阵序列 (T, V, C) / (N, C, T, V, M) 归一化。
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

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        对形状为 (T, V, 3) 的三维骨架时间序列进行逐帧空间归一化。

        Args:
            sequence: (T, V, 3) 原始相机系骨骼时序数据

        Returns:
            normalized_sequence: (T, V, 3) 归一化后时序数据
        """
        T, V, C = sequence.shape
        root_indices = self.skel_def.root_joints
        torso_indices = self.skel_def.torso_joints
        shoulder_indices = [idx for idx in torso_indices if idx not in root_indices]

        normalized_seq = np.zeros_like(sequence, dtype=np.float32)
        for t in range(T):
            frame = sequence[t] # (V, 3)
            # 根节点 (双髋中点)
            root_pos = np.mean(frame[root_indices], axis=0)
            centered = frame - root_pos

            # 躯干尺度
            sh_center = np.mean(frame[shoulder_indices], axis=0)
            torso_len = float(np.linalg.norm(sh_center - root_pos))
            scale = torso_len if torso_len >= 0.1 else self.default_scale_fallback
            scale = max(scale, self.default_scale_fallback * 0.5)

            normalized_seq[t] = centered / (scale + self.eps)

        return normalized_seq

    def normalize_tensor_batch(self, tensor_data: np.ndarray) -> np.ndarray:
        """
        对形状为 (N, C, T, V, M) 的 ST-GCN 标准五维张量进行批量归一化。
        """
        N, C, T, V, M = tensor_data.shape
        out = np.zeros_like(tensor_data, dtype=np.float32)
        for n in range(N):
            for m in range(M):
                seq_t_v_c = np.transpose(tensor_data[n, :, :, :, m], (1, 2, 0)) # (T, V, C)
                norm_seq = self.normalize_sequence(seq_t_v_c)
                out[n, :, :, :, m] = np.transpose(norm_seq, (2, 0, 1))
        return out
