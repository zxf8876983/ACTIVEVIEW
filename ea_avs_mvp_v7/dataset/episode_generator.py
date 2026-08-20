"""
Episode 数据集生成器 —— episode_generator.py
===========================================

职责：
    1. 编排 Habitat 室内场景、Humanoid 动作回放、移动机器人多视角观测与数据录制；
    2. 生成符合 v7.0 规范的 Episode 目录结构：
       data/ActiveView/runs/
       └── episode_xxx/
           ├── rgb/
           │   └── frame_000000.png
           ├── depth/
           │   └── frame_000000.npy
           ├── human_pose/
           │   └── frame_000000.json
           └── metadata.json
    3. 输出标准 Episode 对象并支持全量数据集构建。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v7.core.episode import Episode, EpisodeFrame
from ea_avs_mvp_v7.core.paths import get_data_root, to_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from ea_avs_mvp_v7.observation.metadata import FrameMetadata
from ea_avs_mvp_v7.observation.recorder import ObservationRecorder

logger = logging.getLogger(__name__)


class EpisodeGenerator:
    """标准主动感知 Episode 数据集生成器。"""

    def __init__(
        self,
        env: HabitatEnv,
        humanoid: HumanoidAgent,
        robot: RobotAgent,
        sensor: RGBDSensor,
        recorder: Optional[ObservationRecorder] = None,
    ):
        self.env = env
        self.humanoid = humanoid
        self.robot = robot
        self.sensor = sensor
        self.recorder = recorder or ObservationRecorder()

    def generate_single_episode(
        self,
        episode_id: str,
        motion_player: MotionPlayer,
        camera_position: Union[List[float], np.ndarray],
        camera_yaw_deg: float,
        human_position: Union[List[float], np.ndarray],
        human_yaw_rad: float = 0.0,
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
    ) -> Episode:
        """生成单次动作回放与视点观测 Episode。"""
        out_base = Path(output_dir) if output_dir else get_data_root() / "runs"
        ep_dir = out_base / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 1. 设置 Humanoid 与 Robot 初始位姿
        self.humanoid.set_base_pose(human_position, human_yaw_rad)
        self.robot.set_pose(camera_position, camera_yaw_deg)

        # 计算底盘四元数
        yaw_rad = math.radians(float(camera_yaw_deg))
        robot_rotation_quat = [0.0, float(math.sin(yaw_rad / 2.0)), 0.0, float(math.cos(yaw_rad / 2.0))]

        # 2. 初始化动作播放器
        motion_player.reset()
        total_frames = motion_player.total_frames
        frame_indices = list(range(0, total_frames, max(1, int(frame_step))))
        if max_frames and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

        action_class = motion_player.action_class
        action_label = motion_player.action_label
        babel_sid = motion_player.metadata.get("babel_sid", "0")
        motion_id = f"{action_class}_{babel_sid}"
        avatar_name = self.humanoid.config.get("avatar_name", "neutral_0")

        episode_frames: List[EpisodeFrame] = []
        frames_metadata_list: List[Dict[str, Any]] = []

        logger.info("Generating episode '%s' (%d frames)...", episode_id, len(frame_indices))

        for idx, f_idx in enumerate(frame_indices):
            motion_player.seek(f_idx)
            pose_info = motion_player.get_current_pose()

            # 驱动 Humanoid 模型 (使用相对位移补偿)
            t0 = motion_player.transform_array[0, :3, 3] if hasattr(motion_player, "transform_array") else None
            self.humanoid.apply_motion_frame(
                joints_pose=pose_info["joints_pose"],
                root_transform_mat=pose_info["root_transform"],
                reference_root_translation=t0,
            )

            # 机器人相机渲染
            obs = self.sensor.capture()
            gt_joints = self.humanoid.get_gt_joint_positions()
            cam_mat = self.sensor.get_camera_pose_matrix().tolist()

            f_meta = FrameMetadata(
                frame_index=idx,
                timestamp=float(pose_info["timestamp"]),
                action_class=action_class,
                action_label=action_label,
                babel_sid=babel_sid,
                camera_position=[float(x) for x in camera_position],
                camera_yaw_deg=float(camera_yaw_deg),
                camera_pose_matrix=cam_mat,
                camera_intrinsics=self.sensor.intrinsics,
                human_base_position=[float(x) for x in human_position],
                human_base_yaw=float(human_yaw_rad),
                human_pose_gt_world=gt_joints,
            )

            # 录制 RGB/Depth/Human Pose 到磁盘
            self.recorder.record_frame(
                target_dir=ep_dir,
                frame_idx=idx,
                rgb=obs.get("rgb"),
                depth=obs.get("depth"),
                metadata=f_meta,
                human_pose_gt=gt_joints,
            )

            frame_entry = {
                "frame_id": idx,
                "timestamp": float(pose_info["timestamp"]),
                "rgb_path": f_meta.rgb_relative_path,
                "depth_path": f_meta.depth_relative_path,
                "human_pose_gt": gt_joints,
            }
            frames_metadata_list.append(frame_entry)

            ep_frame = EpisodeFrame(
                frame_index=idx,
                timestamp=float(pose_info["timestamp"]),
                camera_position=f_meta.camera_position,
                camera_yaw_deg=f_meta.camera_yaw_deg,
                camera_pose_matrix=f_meta.camera_pose_matrix,
                camera_intrinsics=f_meta.camera_intrinsics,
                human_base_position=f_meta.human_base_position,
                human_base_yaw=f_meta.human_base_yaw,
                human_pose_gt_world=f_meta.human_pose_gt_world,
                action_class=f_meta.action_class,
                action_label=f_meta.action_label,
                babel_sid=f_meta.babel_sid,
                rgb_relative_path=f_meta.rgb_relative_path,
                depth_relative_path=f_meta.depth_relative_path,
            )
            episode_frames.append(ep_frame)

        # 3. 输出符合标准 schema 的顶层 metadata.json
        full_metadata = {
            "scene_id": self.env.scene_id,
            "episode_id": episode_id,
            "human": {
                "avatar": avatar_name,
                "motion_id": motion_id,
                "action_class": action_class,
                "action_label": action_label,
            },
            "robot": {
                "initial_pose": {
                    "position": [float(x) for x in camera_position],
                    "yaw_deg": float(camera_yaw_deg),
                    "rotation_quat": robot_rotation_quat,
                },
                "camera_pose": {
                    "extrinsic": self.sensor.get_camera_pose_matrix().tolist(),
                    "intrinsic": self.sensor.intrinsics,
                },
            },
            "frames": frames_metadata_list,
        }
        self.recorder.record_episode_metadata(ep_dir, full_metadata)

        episode = Episode(
            episode_id=episode_id,
            scene_id=self.env.scene_id,
            motion_id=motion_id,
            action_class=action_class,
            action_label=action_label,
            num_frames=len(episode_frames),
            camera_view_id="standard_view",
            camera_initial_position=[float(x) for x in camera_position],
            camera_initial_yaw_deg=float(camera_yaw_deg),
            human_initial_position=[float(x) for x in human_position],
            human_initial_yaw_deg=float(human_yaw_rad),
            frames=episode_frames,
            metadata={
                "source_motion_metadata": motion_player.metadata,
                "episode_dir": to_relative_data_path(ep_dir),
            },
        )

        return episode
