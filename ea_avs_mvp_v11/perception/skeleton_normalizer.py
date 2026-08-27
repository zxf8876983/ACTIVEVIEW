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
    3. 可选的人体偏航角归一化仅消除 Yaw，严格保留相对重力的 Roll/Pitch；
    4. 统一从 `configs/skeleton_definition.json` 获取根节点与躯干关键点索引；
    5. 支持单帧骨架对象归一化与批量时序矩阵序列 (T, V, C) / (N, C, T, V, M) 归一化。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
        skeleton: Any,
        use_world_coords: bool = False,
    ) -> Any:
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

    def align_to_canonical_frame(self, skeleton_seq: np.ndarray) -> np.ndarray:
        """
        将 (T, V, 3) 骨架序列对齐至标准偏航角，同时保留重力姿态。

        Only rotation around the vertical Y axis is removed. Full 3D body-frame
        alignment would rotate a falling or lying torso upright and erase the
        signal required by household safety-action recognition.
        """
        T, _, _ = skeleton_seq.shape
        aligned_seq = np.zeros_like(skeleton_seq, dtype=np.float32)

        name_to_id = self.skel_def.name_to_id
        left_hip = name_to_id["left_hip"]
        right_hip = name_to_id["right_hip"]
        left_shoulder = name_to_id["left_shoulder"]
        right_shoulder = name_to_id["right_shoulder"]
        raw_yaws = np.zeros(T, dtype=np.float32)
        for t, frame in enumerate(skeleton_seq):
            hip_axis = frame[right_hip] - frame[left_hip]
            shoulder_axis = frame[right_shoulder] - frame[left_shoulder]
            lateral_axis = hip_axis + shoulder_axis
            dx, dz = float(lateral_axis[0]), float(lateral_axis[2])
            if np.hypot(dx, dz) > self.eps:
                raw_yaws[t] = np.arctan2(-dz, -dx)

        yaws = np.unwrap(raw_yaws)
        smooth_window = 5
        if T >= smooth_window:
            pad_size = smooth_window // 2
            padded = np.pad(yaws, (pad_size, pad_size), mode="edge")
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            yaws = np.convolve(padded, kernel, mode="valid")[:T]

        for t, frame in enumerate(skeleton_seq):
            cos_yaw = np.cos(-float(yaws[t]))
            sin_yaw = np.sin(-float(yaws[t]))
            rotation = np.array(
                [
                    [cos_yaw, 0.0, -sin_yaw],
                    [0.0, 1.0, 0.0],
                    [sin_yaw, 0.0, cos_yaw],
                ],
                dtype=np.float32,
            )
            aligned_seq[t] = np.matmul(frame, rotation.T)

        return aligned_seq

    def normalize_sequence(self, sequence: np.ndarray, align_canonical: bool = False) -> np.ndarray:
        """
        对形状为 (T, V, 3) 的三维骨架时间序列进行逐帧空间归一化。

        Args:
            sequence: (T, V, 3) 原始相机系骨骼时序数据
            align_canonical: 是否旋转对齐至人体标准正向参考系

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

        if align_canonical:
            normalized_seq = self.align_to_canonical_frame(normalized_seq)

        return normalized_seq

    def normalize_tensor_batch(self, tensor_data: np.ndarray, align_canonical: bool = False) -> np.ndarray:
        """
        对形状为 (N, C, T, V, M) 的 ST-GCN 标准五维张量进行批量归一化。
        """
        N, C, T, V, M = tensor_data.shape
        out = np.zeros_like(tensor_data, dtype=np.float32)
        for n in range(N):
            for m in range(M):
                seq_t_v_c = np.transpose(tensor_data[n, :, :, :, m], (1, 2, 0)) # (T, V, C)
                norm_seq = self.normalize_sequence(seq_t_v_c, align_canonical=align_canonical)
                out[n, :, :, :, m] = np.transpose(norm_seq, (2, 0, 1))
        return out
