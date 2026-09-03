"""
AMASS 动作加载与数据规范化器 —— amass_loader.py
=============================================

职责：
    1. 读取 AMASS .npz 原始动作文件；
    2. 自适应兼容 Schema A (显式 root_orient) 与 Schema B (root_orient 位于 poses[:, :3])；
    3. 将人体身体关节零填充至 SMPL-X 标准 54 关节维度 (162 维)；
    4. 支持按帧范围 (start_frame:end_frame) 规范化切片；
    5. 输出统一的 NormalizedMotion 对象。

边界约束：
    - 纯 Python / NumPy 实现，严禁依赖 Habitat 物理引擎或机器人控制。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

NUM_SMPLX_JOINTS = 54
SMPLX_JOINT_DIM = 162  # 54 joints * 3 dof (Rodrigues)


@dataclass
class NormalizedMotion:
    """标准化的统一动作数据结构。"""
    translation: np.ndarray        # (N, 3) float32
    root_rotation: np.ndarray      # (N, 3) float32 (Rodrigues 旋转矢量)
    body_pose: np.ndarray          # (N, 162) float32 (54 关节 * 3)
    fps: float                     # 动作采样率
    num_frames: int                # 帧数
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.translation.ndim == 2 and self.translation.shape[1] == 3
        assert self.root_rotation.ndim == 2 and self.root_rotation.shape[1] == 3
        assert self.body_pose.ndim == 2 and self.body_pose.shape[1] == SMPLX_JOINT_DIM
        assert self.translation.shape[0] == self.num_frames
        assert self.root_rotation.shape[0] == self.num_frames
        assert self.body_pose.shape[0] == self.num_frames


class AMASSLoader:
    """AMASS .npz 动作数据文件读取与标准化加载器。"""

    @staticmethod
    def load(
        npz_path: Union[str, Path],
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NormalizedMotion:
        """加载并标准化 AMASS .npz 动作。"""
        p = Path(npz_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"AMASS file not found: {p}")

        data = np.load(p, allow_pickle=True)

        if "trans" not in data or "poses" not in data:
            raise ValueError(f"Invalid AMASS file (missing 'trans' or 'poses'): {p}")

        trans_raw = data["trans"].astype(np.float32)
        poses_raw = data["poses"].astype(np.float32)
        total_raw_frames = poses_raw.shape[0]

        # 1. 解析帧率
        fps = 30.0
        for fps_key in ("mocap_framerate", "mocap_frame_rate", "frame_rate", "framerate", "fps"):
            if fps_key in data:
                val = data[fps_key]
                fps = float(val) if np.ndim(val) == 0 else float(val[0])
                break

        # 2. 解析 Root Rotation 与 Body Pose (Schema A vs Schema B)
        if "root_orient" in data and data["root_orient"].shape[0] == total_raw_frames:
            root_rot_raw = data["root_orient"].astype(np.float32)
            body_pose_raw = poses_raw[:, 3:] if poses_raw.shape[1] > 3 else poses_raw
        else:
            # Schema B: root orient 包含在 poses 前 3 维
            root_rot_raw = poses_raw[:, :3]
            body_pose_raw = poses_raw[:, 3:]

        # 3. 补齐 SMPL-X 关节至 162 维 (54 joints * 3)
        padded_body_pose = np.zeros((total_raw_frames, SMPLX_JOINT_DIM), dtype=np.float32)
        valid_dims = min(body_pose_raw.shape[1], SMPLX_JOINT_DIM)
        padded_body_pose[:, :valid_dims] = body_pose_raw[:, :valid_dims]

        # 4. 时序帧切片
        sf = max(0, int(start_frame)) if start_frame is not None else 0
        ef = min(total_raw_frames - 1, int(end_frame)) if end_frame is not None else total_raw_frames - 1
        if sf > ef:
            sf, ef = 0, total_raw_frames - 1

        trans_sliced = trans_raw[sf : ef + 1]
        root_rot_sliced = root_rot_raw[sf : ef + 1]
        body_pose_sliced = padded_body_pose[sf : ef + 1]
        num_frames = ef - sf + 1

        meta = dict(metadata or {})
        meta.update({
            "source_file": str(p),
            "start_frame": sf,
            "end_frame": ef,
            "total_raw_frames": total_raw_frames,
            "fps": fps,
        })

        return NormalizedMotion(
            translation=trans_sliced,
            root_rotation=root_rot_sliced,
            body_pose=body_pose_sliced,
            fps=fps,
            num_frames=num_frames,
            metadata=meta,
        )


def load_amass_motion(
    npz_path: Union[str, Path],
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> NormalizedMotion:
    """快捷加载单条 AMASS 动作为 NormalizedMotion。"""
    return AMASSLoader.load(npz_path, start_frame, end_frame, metadata)
