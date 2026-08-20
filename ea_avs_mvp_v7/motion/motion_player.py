"""
动作回放控制器 —— motion_player.py
=================================

职责：
    1. 加载 Habitat 转换后的 .pkl 动作文件；
    2. 管理时序帧指针，控制动作单步执行 (step)、跳转 (seek) 与重置 (reset)；
    3. 输出当前时刻的关节姿态与 4x4 根变换矩阵。

边界约束：
    - 纯时序数据提供器，不直接操作 Habitat 仿真世界。
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MotionPlayer:
    """人体动作序列回放播放器。"""

    def __init__(
        self,
        motion_pkl_path: Union[str, Path],
        playback_fps: float = 30.0,
    ):
        self.pkl_path = Path(motion_pkl_path).resolve()
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Motion PKL not found: {self.pkl_path}")

        with open(self.pkl_path, "rb") as f:
            raw_data = pickle.load(f)

        if "pose_motion" not in raw_data:
            raise ValueError(f"Invalid motion file (missing 'pose_motion'): {self.pkl_path}")

        motion_info = raw_data["pose_motion"]
        self.metadata = raw_data.get("metadata", {})

        self.joints_array = np.asarray(motion_info["joints_array"], dtype=np.float32)
        self.transform_array = np.asarray(motion_info["transform_array"], dtype=np.float32)
        self.fps = float(motion_info.get("fps", 30.0))
        self.playback_fps = playback_fps
        self.num_frames = self.joints_array.shape[0]

        self.current_frame = 0

    @property
    def total_frames(self) -> int:
        return self.num_frames

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    @property
    def action_class(self) -> str:
        return self.metadata.get("target_class", "unknown_action")

    @property
    def action_label(self) -> str:
        return self.metadata.get("proc_label", self.metadata.get("raw_label", "action"))

    def is_finished(self) -> bool:
        return self.current_frame >= self.num_frames - 1

    def reset(self) -> None:
        """重置播放进度至第 0 帧。"""
        self.current_frame = 0

    def seek(self, frame_index: int) -> int:
        """跳转至指定帧序号。"""
        self.current_frame = max(0, min(int(frame_index), self.num_frames - 1))
        return self.current_frame

    def step(self, advance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前帧的关节四元数数组与 4x4 根变换矩阵。

        Returns:
            (joints_pose (216,), root_transform_mat (4, 4))
        """
        f = self.current_frame
        joints = self.joints_array[f]
        transform_mat = self.transform_array[f]

        if advance and not self.is_finished():
            self.current_frame += 1

        return joints, transform_mat

    def get_current_pose(self) -> Dict[str, Any]:
        """获取当前帧的姿态信息字典。"""
        joints, transform_mat = self.step(advance=False)
        return {
            "frame_index": self.current_frame,
            "total_frames": self.num_frames,
            "timestamp": self.current_frame / self.fps if self.fps > 0 else 0.0,
            "joints_pose": joints,
            "root_transform": transform_mat,
            "action_class": self.action_class,
            "action_label": self.action_label,
            "metadata": self.metadata,
        }
