"""
Humanoid 动作序列回放播放器 —— motion_player.py
=============================================

功能：
    1. 加载 Habitat SMPL-X 动作 .pkl 文件；
    2. 控制动作帧步进、跳转 (seek) 与循环/单次播放；
    3. 提取当前帧的关节角度 (joint positions) 与根变换 (root transformation)；
    4. 供 HumanoidAgent 实时同步至 Habitat 物理世界。
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import magnum as mn
    from habitat.articulated_agent_controllers.humanoid_seq_pose_controller import HumanoidSeqPoseController
except ImportError:
    mn = None
    HumanoidSeqPoseController = None

logger = logging.getLogger(__name__)


class MotionPlayer:
    """Humanoid 动作序列播放器。"""

    def __init__(
        self,
        motion_pkl_path: Union[str, Path],
        playback_fps: float = 30.0,
        base_offset: Tuple[float, float, float] = (0.0, 0.9, 0.0),
    ):
        self.pkl_path = Path(motion_pkl_path).resolve()
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Converted motion PKL not found: {self.pkl_path}")

        with open(self.pkl_path, "rb") as f:
            raw_data = pickle.load(f)

        if "pose_motion" not in raw_data:
            raise ValueError(f"Invalid motion file format (missing 'pose_motion'): {self.pkl_path}")

        self.motion_info = raw_data["pose_motion"]
        self.metadata = raw_data.get("metadata", {})

        self.joints_array = np.asarray(self.motion_info["joints_array"], dtype=np.float32)
        self.transform_array = np.asarray(self.motion_info["transform_array"], dtype=np.float32)
        self.fps = float(self.motion_info.get("fps", 30.0))
        self.playback_fps = playback_fps
        self.num_frames = self.joints_array.shape[0]

        self.current_frame = 0
        self.base_offset = base_offset

        # 封装底层 Habitat 控制器（若可用）
        self._controller: Optional[HumanoidSeqPoseController] = None
        if HumanoidSeqPoseController is not None:
            try:
                self._controller = HumanoidSeqPoseController(
                    motion_pose_path=str(self.pkl_path),
                    motion_fps=playback_fps,
                    base_offset=base_offset,
                )
            except Exception as e:
                logger.warning("HumanoidSeqPoseController init fallback: %s", e)

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

    def reset(self, initial_base_transform: Optional[Any] = None) -> None:
        """重置播放进度至第一帧。"""
        self.current_frame = 0
        if self._controller is not None:
            base_t = initial_base_transform if initial_base_transform is not None else (mn.Matrix4() if mn else None)
            if base_t is not None:
                self._controller.reset(base_t)

    def seek(self, frame_index: int) -> int:
        """跳转至指定帧。"""
        self.current_frame = max(0, min(int(frame_index), self.num_frames - 1))
        if self._controller is not None:
            self._controller.motion_frame = self.current_frame
            self._controller.calculate_pose(advance_pose=False)
        return self.current_frame

    def step(self, advance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前帧的关节角度与根变换，并前进一步。

        Returns:
            (joints_pose, root_transform_matrix_4x4)
        """
        f = self.current_frame
        joints = self.joints_array[f]
        transform_mat = self.transform_array[f]

        if advance and not self.is_finished():
            self.current_frame += 1
            if self._controller is not None:
                self._controller.calculate_pose(advance_pose=True)

        return joints, transform_mat

    def get_current_pose(self) -> Dict[str, Any]:
        """获取当前帧的综合姿态字典。"""
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
