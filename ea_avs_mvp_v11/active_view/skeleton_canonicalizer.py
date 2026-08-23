"""
Canonical Skeleton Alignment 模块 —— skeleton_canonicalizer.py
=============================================================

职责：
    1. 将任意机器人观察角度（Camera Coordinate System）下的 3D 人体骨架，
       统一转换至人体自身标准正规化坐标系（Human Canonical Coordinate System）；
    2. 基于人体骨盆（pelvis）、左右髋关节（left/right hip）和左右肩关节（left/right shoulder）
       构造人体本征旋转矩阵 R_body；
    3. 消除相机视角造成的 Yaw 刚体旋转与空间平移，实现视角不变性（View-Invariance）；
    4. 对动作时序执行角度连续展开（Unwrap）与时序平滑（Rotation Smoothing），避免逐帧抖动；
    5. 支持 (T, V, 3), (V, 3), (N, C, T, V, M) 等多种输入张量维度，严格保证数值稳定无 NaN。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger("skeleton_canonicalizer")


class CanonicalSkeletonAligner:
    """人体 3D 骨架坐标系正规化与视角不变性对齐器。"""

    def __init__(
        self,
        smooth_window: int = 5,
        eps: float = 1e-6,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.smooth_window = max(1, int(smooth_window))
        self.eps = float(eps)
        self.skel_def = skel_def or get_skeleton_definition()

        # 获取关键解剖学关节索引 (MediaPipe-33 标准)
        name_to_id = self.skel_def.name_to_id
        self.left_shoulder_idx = name_to_id.get("left_shoulder", 11)
        self.right_shoulder_idx = name_to_id.get("right_shoulder", 12)
        self.left_hip_idx = name_to_id.get("left_hip", 23)
        self.right_hip_idx = name_to_id.get("right_hip", 24)

    def compute_yaw_trajectory(self, sequence: np.ndarray, smooth: bool = True) -> np.ndarray:
        """
        计算骨架时序中每一帧相对于正向观察基准的人体偏航角 (Yaw Angle, 弧度)。

        Args:
            sequence: (T, V, 3) 骨架时序矩阵
            smooth: 是否应用时序平滑

        Returns:
            yaws: (T,) 展开并平滑后的偏航角时序 (弧度)
        """
        T = sequence.shape[0]
        raw_yaws = np.zeros(T, dtype=np.float32)

        for t in range(T):
            frame = sequence[t]
            # 计算左右髋关节与肩关节横向向量
            p_lh = frame[self.left_hip_idx]
            p_rh = frame[self.right_hip_idx]
            p_ls = frame[self.left_shoulder_idx]
            p_rs = frame[self.right_shoulder_idx]

            v_hip = p_rh - p_lh
            v_sh = p_rs - p_ls
            v_lat = v_hip + v_sh

            dx, dz = float(v_lat[0]), float(v_lat[2])
            norm = math.sqrt(dx * dx + dz * dz)

            if norm > self.eps:
                # 在正视相机坐标系下, 右髋位于 -X 方向, 左髋位于 +X 方向, v_lat 主轴为 [-1, 0, 0]
                # 经 Y 轴旋转 θ 后: v_rot = [-cos θ, 0, -sin θ]
                # 因此 θ = atan2(-dz, -dx)
                yaw = math.atan2(-dz, -dx)
            else:
                yaw = 0.0

            raw_yaws[t] = yaw

        # 展开角度跳变 (-π 到 π 的跳变消除)
        unwrapped = np.unwrap(raw_yaws)

        if smooth and T >= self.smooth_window and self.smooth_window > 1:
            pad_size = self.smooth_window // 2
            pad_yaws = np.pad(unwrapped, (pad_size, pad_size), mode="edge")
            kernel = np.ones(self.smooth_window, dtype=np.float32) / self.smooth_window
            smoothed = np.convolve(pad_yaws, kernel, mode="valid")
            return smoothed[:T].astype(np.float32)

        return unwrapped.astype(np.float32)

    def align(
        self,
        skeleton: np.ndarray,
        smooth: bool = True,
        center_pelvis: bool = True,
    ) -> np.ndarray:
        """
        将 3D 骨架序列正规化到人体标准朝向坐标系 (Canonical Frame)。

        Args:
            skeleton: (T, V, 3) 或 (V, 3) 形状的 NumPy 数组
            smooth: 是否对时序旋转角进行平滑
            center_pelvis: 是否以骨盆中心为原点平移

        Returns:
            canonical_skeleton: (T, V, 3) 或 (V, 3) 标准正视坐标系下的骨骼矩阵
        """
        orig_shape = skeleton.shape

        # 处理单帧 (V, 3)
        if skeleton.ndim == 2:
            seq = skeleton[np.newaxis, ...]  # (1, V, 3)
            is_single = True
        elif skeleton.ndim == 3:
            seq = skeleton.copy()
            is_single = False
        else:
            raise ValueError(f"Expected skeleton with 2 or 3 dims, got shape {orig_shape}")

        T, V, C = seq.shape
        assert C == 3, f"Expected 3 spatial coordinates (X, Y, Z), got {C}"

        yaws = self.compute_yaw_trajectory(seq, smooth=smooth and not is_single)
        canonical_seq = np.zeros_like(seq, dtype=np.float32)

        for t in range(T):
            frame = seq[t]
            p_lh = frame[self.left_hip_idx]
            p_rh = frame[self.right_hip_idx]

            if center_pelvis:
                center = (p_lh + p_rh) * 0.5
                centered = frame - center
            else:
                centered = frame

            yaw = float(yaws[t])
            cos_y = math.cos(-yaw)
            sin_y = math.sin(-yaw)

            # 绕 Y 轴反向旋转消除偏航
            R_inv = np.array(
                [
                    [cos_y, 0.0, -sin_y],
                    [0.0, 1.0, 0.0],
                    [sin_y, 0.0, cos_y],
                ],
                dtype=np.float32,
            )

            # (V, 3) = (centered @ R_inv.T)
            canonical_frame = np.matmul(centered, R_inv.T)
            canonical_seq[t] = canonical_frame

        # 安全防爆：去除任何潜在的 NaN 或 Inf
        np.nan_to_num(canonical_seq, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if is_single:
            return canonical_seq[0]
        return canonical_seq

    def align_tensor_sequence(self, tensor_seq: np.ndarray) -> np.ndarray:
        """
        对 ST-GCN 标准输入张量 (C, T, V, M) 或 (C, T, V) 进行 Canonical Alignment。

        Args:
            tensor_seq: (C, T, V, M) 或 (C, T, V) 格式张量

        Returns:
            aligned_tensor: 相同形状的 Canonical 张量
        """
        if tensor_seq.ndim == 4:
            C, T, V, M = tensor_seq.shape
            out = np.zeros_like(tensor_seq)
            for m in range(M):
                raw_skel = np.transpose(tensor_seq[:, :, :, m], (1, 2, 0))  # (T, V, C)
                aligned_skel = self.align(raw_skel)
                out[:, :, :, m] = np.transpose(aligned_skel, (2, 0, 1))
            return out
        elif tensor_seq.ndim == 3:
            C, T, V = tensor_seq.shape
            raw_skel = np.transpose(tensor_seq, (1, 2, 0))  # (T, V, C)
            aligned_skel = self.align(raw_skel)
            return np.transpose(aligned_skel, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported tensor shape {tensor_seq.shape}")


# 全局单例
_GLOBAL_CANONICAL_ALIGNER: Optional[CanonicalSkeletonAligner] = None


def get_canonical_skeleton_aligner() -> CanonicalSkeletonAligner:
    """获取全局骨架正规化单例。"""
    global _GLOBAL_CANONICAL_ALIGNER
    if _GLOBAL_CANONICAL_ALIGNER is None:
        _GLOBAL_CANONICAL_ALIGNER = CanonicalSkeletonAligner()
    return _GLOBAL_CANONICAL_ALIGNER
