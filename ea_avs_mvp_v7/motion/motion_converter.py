"""
Habitat SMPL-X 动作格式转换器 —— motion_converter.py
===================================================

职责：
    1. 接收 NormalizedMotion 标准动作对象；
    2. 基于 Habitat 官方 MotionConverterSMPLX 执行 URDF 坐标系变换与 54 关节四元数求解；
    3. 校验输出四元数维度 (216 维) 与根变换矩阵 (4x4)；
    4. 输出并持久化标准 Habitat .pkl 格式文件。

边界约束：
    - 仅负责数学与几何转换，不控制 Habitat 仿真世界或机器人传感器。
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import magnum as mn
    from habitat.utils.humanoid_utils import MotionConverterSMPLX
except ImportError:
    mn = None
    MotionConverterSMPLX = None

from .amass_loader import NormalizedMotion, load_amass_motion
from .joint_mapping import (
    HABITAT_HUMANOID_QUAT_DIM,
    SMPLX_RODRIGUES_DIM,
    validate_motion_quaternions,
)

logger = logging.getLogger(__name__)


class MotionConverter:
    """NormalizedMotion 到 Habitat SMPL-X 动作格式的转换器。"""

    def __init__(self, urdf_path: Union[str, Path]):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Humanoid URDF not found at {self.urdf_path}")

        if MotionConverterSMPLX is None:
            raise RuntimeError(
                "habitat.utils.humanoid_utils.MotionConverterSMPLX is unavailable. "
                "Please run inside the habitat conda environment."
            )

        self._converter = MotionConverterSMPLX(str(self.urdf_path))

    def convert(self, motion: NormalizedMotion) -> Dict[str, Any]:
        """将 NormalizedMotion 转换为 Habitat 格式动作字典。"""
        if motion.body_pose.shape[1] != SMPLX_RODRIGUES_DIM:
            raise ValueError(
                f"NormalizedMotion body_pose must have {SMPLX_RODRIGUES_DIM} dims, got {motion.body_pose.shape[1]}"
            )

        transform_array = []
        joints_array = []

        for f_idx in range(motion.num_frames):
            root_t, root_r, pose_quat = self._converter.convert_pose_to_rotation(
                motion.translation[f_idx],
                motion.root_rotation[f_idx],
                motion.body_pose[f_idx],
            )
            transform_mat = np.array(mn.Matrix4.from_(root_r, root_t), dtype=np.float32)
            transform_array.append(transform_mat[None, :])
            joints_array.append(np.array(pose_quat, dtype=np.float32)[None, :])

        transform_array = np.concatenate(transform_array, axis=0)
        joints_array = np.concatenate(joints_array, axis=0)

        # 校验四元数与变换矩阵维度
        if joints_array.shape[1] != HABITAT_HUMANOID_QUAT_DIM:
            raise ValueError(
                f"Converted joints array shape mismatch: expected (N, {HABITAT_HUMANOID_QUAT_DIM}), got {joints_array.shape}"
            )
        if transform_array.shape[1:] != (4, 4):
            raise ValueError(
                f"Converted transform array shape mismatch: expected (N, 4, 4), got {transform_array.shape}"
            )

        validate_motion_quaternions(joints_array)

        meta = dict(motion.metadata)
        meta.update({
            "num_frames": motion.num_frames,
            "fps": motion.fps,
            "duration_seconds": motion.num_frames / motion.fps if motion.fps > 0 else 0.0,
        })

        return {
            "pose_motion": {
                "joints_array": joints_array,
                "transform_array": transform_array,
                "displacement": None,
                "fps": motion.fps,
            },
            "metadata": meta,
        }

    def convert_and_save(
        self,
        motion: NormalizedMotion,
        output_pkl_path: Union[str, Path],
    ) -> Path:
        """转换并保存为 .pkl 文件。"""
        data = self.convert(motion)
        out_p = Path(output_pkl_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved converted motion PKL to: %s", out_p)
        return out_p


def convert_normalized_motion_to_pkl(
    motion: NormalizedMotion,
    urdf_path: Union[str, Path],
    output_pkl_path: Union[str, Path],
) -> Path:
    """快捷转换并保存函数。"""
    converter = MotionConverter(urdf_path)
    return converter.convert_and_save(motion, output_pkl_path)
