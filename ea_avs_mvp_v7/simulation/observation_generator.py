"""
RGB-D 观测与标注数据集生成器 —— observation_generator.py
======================================================

功能：
    1. 驱动 Humanoid 回放指定动作序列；
    2. 同步控制移动机器人相机在指定或环绕视点采集 RGB-D 图像；
    3. 提取对应的 3D 关节位置 Ground-Truth、相机外参、动作标签与时间戳；
    4. 将观测与元数据持久化输出至指定外部数据目录。
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from tools.motion_assets.data_paths import get_data_root, to_relative_data_path

logger = logging.getLogger(__name__)


@dataclass
class ObservationRecord:
    """单帧多模态观测记录。"""
    frame_index: int
    timestamp: float
    action_class: str
    action_label: str
    babel_sid: Union[int, str]
    camera_position: List[float]
    camera_yaw_deg: float
    camera_pose_matrix: List[List[float]]
    camera_intrinsics: Dict[str, float]
    human_base_position: List[float]
    human_base_yaw: float
    human_pose_gt_world: Dict[str, List[float]]
    rgb_relative_path: Optional[str] = None
    depth_relative_path: Optional[str] = None


class ObservationGenerator:
    """RGB-D 动作感知观测数据集生成器。"""

    def __init__(
        self,
        scene_loader,
        humanoid_agent,
        robot_sensor,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.scene_loader = scene_loader
        self.humanoid_agent = humanoid_agent
        self.robot_sensor = robot_sensor
        self.config = config or {}

    def capture_single_frame(
        self,
        motion_player,
        camera_pos: Union[List[float], np.ndarray],
        camera_yaw_deg: float,
        frame_index: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], ObservationRecord]:
        """在特定视点下捕获当前动作帧的观测与标注。"""
        if frame_index is not None:
            motion_player.seek(frame_index)

        # 1. 提取当前动作姿态并更新 Humanoid
        pose_info = motion_player.get_current_pose()
        self.humanoid_agent.apply_motion_frame(
            joints_pose=pose_info["joints_pose"],
            root_transform_mat=pose_info["root_transform"],
        )

        # 2. 设置机器人相机位姿并渲染
        self.robot_sensor.set_pose(camera_pos, camera_yaw_deg)
        obs = self.robot_sensor.get_observation()

        # 3. 提取 Ground-Truth 关节世界坐标
        gt_joints_raw = self.humanoid_agent.get_gt_joint_positions()
        gt_joints = {k: v.tolist() for k, v in gt_joints_raw.items()}

        cam_mat = self.robot_sensor.get_camera_pose_matrix().tolist()
        meta = pose_info.get("metadata", {})

        record = ObservationRecord(
            frame_index=pose_info["frame_index"],
            timestamp=pose_info["timestamp"],
            action_class=pose_info["action_class"],
            action_label=pose_info["action_label"],
            babel_sid=meta.get("babel_sid", "0"),
            camera_position=[float(x) for x in camera_pos],
            camera_yaw_deg=float(camera_yaw_deg),
            camera_pose_matrix=cam_mat,
            camera_intrinsics=self.robot_sensor.intrinsics,
            human_base_position=self.humanoid_agent.base_position.tolist(),
            human_base_yaw=self.humanoid_agent.base_yaw,
            human_pose_gt_world=gt_joints,
        )

        return obs, record

    def run_sequence(
        self,
        motion_player,
        camera_pos: Union[List[float], np.ndarray],
        camera_yaw_deg: float,
        output_dir: Optional[Union[str, Path]] = None,
        frame_step: int = 1,
        max_frames: Optional[int] = None,
    ) -> List[ObservationRecord]:
        """运行完整动作序列，逐帧采集并保存观测数据。"""
        out_base = Path(output_dir) if output_dir else get_data_root() / "runs" / "v70_observations"
        action_name = f"{motion_player.action_class}_{motion_player.metadata.get('babel_sid', '0')}"
        seq_dir = out_base / action_name
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        meta_dir = seq_dir / "metadata"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        records = []
        total = motion_player.total_frames
        motion_player.reset()

        frame_indices = list(range(0, total, max(1, int(frame_step))))
        if max_frames and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

        logger.info(
            "Starting sequence generation: %s (%d frames to capture)",
            action_name,
            len(frame_indices),
        )

        for f_idx in frame_indices:
            obs, rec = self.capture_single_frame(
                motion_player=motion_player,
                camera_pos=camera_pos,
                camera_yaw_deg=camera_yaw_deg,
                frame_index=f_idx,
            )

            # 保存 RGB
            rgb_fname = f"{f_idx:04d}.png"
            rgb_path = rgb_dir / rgb_fname
            if obs.get("rgb") is not None:
                img = Image.fromarray(obs["rgb"])
                img.save(rgb_path)
                rec.rgb_relative_path = to_relative_data_path(rgb_path)

            # 保存 Depth (npy 原始深度)
            depth_fname = f"{f_idx:04d}.npy"
            depth_path = depth_dir / depth_fname
            if obs.get("depth") is not None:
                np.save(depth_path, obs["depth"].astype(np.float32))
                rec.depth_relative_path = to_relative_data_path(depth_path)

            # 保存单帧元数据
            meta_fname = f"{f_idx:04d}.json"
            meta_path = meta_dir / meta_fname
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(asdict(rec), f, indent=2, ensure_ascii=False)

            records.append(rec)

        # 保存整段序列汇总清单
        summary_path = seq_dir / "sequence_summary.json"
        summary_data = {
            "action_class": motion_player.action_class,
            "action_label": motion_player.action_label,
            "babel_sid": motion_player.metadata.get("babel_sid", "0"),
            "num_captured_frames": len(records),
            "camera_position": [float(x) for x in camera_pos],
            "camera_yaw_deg": float(camera_yaw_deg),
            "records": [asdict(r) for r in records],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved sequence dataset to %s (%d frames)", seq_dir, len(records))
        return records
