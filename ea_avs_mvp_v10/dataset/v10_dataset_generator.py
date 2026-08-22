"""
v10 Phase 1 RGB-D 数据集生成引擎 —— v10_dataset_generator.py
============================================================

职责：
    1. 编排 Habitat 仿真环境 (场景 + 人形模型 + 动作回放 + 机器人传感器)；
    2. 对指定动作序列进行多视点 (Multi-Viewpoint) RGB-D 观测采集与渲染；
    3. 同步提取相机位姿 (CameraPose) 与 GT 骨骼姿态 (仅供 Oracle/验证)；
    4. 输出规范的 v10 样本集与 metadata 索引清单；
    5. 严格隔离：Phase 1 仅提供仿真数据基础，不涉及 ST-GCN 或主动选点策略。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v10.core.config import V10Config, load_v10_config
from ea_avs_mvp_v10.core.paths import get_v10_dataset_root
from ea_avs_mvp_v10.core.types import ActionClassV10, CandidateViewpointV10, V10Sample
from ea_avs_mvp_v10.dataset.sample_builder import V10SampleBuilder
from ea_avs_mvp_v10.motion.motion_manager import MotionManager
from ea_avs_mvp_v10.sensors.rgbd_capture import RGBDCapture
from ea_avs_mvp_v10.viewpoint.candidate_generator import CandidateViewpointGeneratorV10

logger = logging.getLogger(__name__)


class V10DatasetGenerator:
    """v10.0 RGB-D 多视角动作数据集生成流水线。"""

    def __init__(
        self,
        config: Optional[V10Config] = None,
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        self.config = config or load_v10_config()
        self.dataset_root = Path(dataset_root) if dataset_root else get_v10_dataset_root()

        self.motion_mgr = MotionManager()
        self.vp_gen = CandidateViewpointGeneratorV10(
            radii=self.config.viewpoint.get("radii", [1.5, 2.0, 2.5]),
            num_angles=int(self.config.viewpoint.get("num_angles", 8)),
            camera_height=float(self.config.camera.get("camera_height", 1.20)),
            ground_height=float(self.config.viewpoint.get("ground_height", -1.60)),
        )
        self.builder = V10SampleBuilder(self.dataset_root)

    def generate_single_sample_offline(
        self,
        sample_id: str,
        scene_id: str,
        motion_id: str,
        action_label: str,
        viewpoint: CandidateViewpointV10,
        frame_idx: int = 0,
        gt_skeleton: Optional[Dict[str, List[float]]] = None,
    ) -> V10Sample:
        """纯离线模式生成单样本 (用于测试或无物理引擎环境)。"""
        capture = RGBDCapture(sim=None, sensor_cfg=self.config.camera)
        rgb, depth, cam_pose = capture.capture_frame()

        cam_pose.position = viewpoint.position
        cam_pose.yaw_deg = viewpoint.yaw_deg

        sample = self.builder.build_and_save_sample(
            sample_id=sample_id,
            scene_id=scene_id,
            motion_id=motion_id,
            action_label=action_label,
            frame_idx=frame_idx,
            view_id=viewpoint.viewpoint_id,
            rgb_array=rgb,
            depth_array=depth,
            camera_pose=cam_pose,
            gt_skeleton=gt_skeleton,
            extra_metadata={"viewpoint_radius": viewpoint.radius, "viewpoint_angle_deg": viewpoint.angle_deg},
        )
        return sample

    def generate_motion_dataset(
        self,
        motion_ids: Optional[List[str]] = None,
        human_position: Optional[List[float]] = None,
        human_yaw_deg: float = 0.0,
        frame_step: int = 10,
        max_frames_per_motion: int = 5,
        max_viewpoints: Optional[int] = 4,
    ) -> List[V10Sample]:
        """
        在 Habitat 仿真环境中执行完整动作序列并采集多视角 RGB-D 样本。
        """
        env_adapter = V8EnvironmentAdapter(self.config.scene, self.config.camera)
        sim = env_adapter.start()

        humanoid = HumanoidAgent(sim, self.config.human)
        humanoid.load()
        humanoid.set_visibility(True)

        robot = RobotAgent(sim, agent_id=0)
        capture = RGBDCapture(sim, sensor_cfg=self.config.camera, agent_id=0)

        h_pos = human_position or self.config.human.get("default_position", [1.5, -1.60, 4.0])
        human_yaw_rad = math.radians(human_yaw_deg)
        humanoid.set_base_pose(h_pos, yaw_rad=human_yaw_rad)

        candidates = self.vp_gen.generate_candidates(h_pos, human_yaw_deg=human_yaw_deg)

        # 过滤可导航且视线无物理阻挡的有效视点 (排除墙体/隔断阻挡)
        from ea_avs_mvp_v8.constraints.line_of_sight_constraint import LineOfSightConstraint
        from ea_avs_mvp_v8.core.types import CandidateViewpoint

        los_checker = LineOfSightConstraint(env_adapter=env_adapter)
        valid_candidates = []
        for vp in candidates:
            if not env_adapter.is_navigable(vp.position):
                continue
            v8_vp = CandidateViewpoint(
                viewpoint_id=vp.viewpoint_id,
                position=vp.position,
                yaw_deg=vp.yaw_deg,
                radius=vp.radius,
                angle_deg=vp.angle_deg,
                camera_height=vp.height,
            )
            is_los, _ = los_checker.evaluate(v8_vp, human_position=h_pos)
            if is_los:
                valid_candidates.append(vp)

        if not valid_candidates:
            valid_candidates = [vp for vp in candidates if env_adapter.is_navigable(vp.position)] or candidates

        if max_viewpoints and len(valid_candidates) > max_viewpoints:
            # 均匀下采样候选视点
            indices = np.linspace(0, len(valid_candidates) - 1, max_viewpoints, dtype=int)
            sampled_candidates = [valid_candidates[i] for i in indices]
        else:
            sampled_candidates = valid_candidates

        target_motions = motion_ids or self.motion_mgr.list_available_motions()[:6]
        all_samples: List[V10Sample] = []
        sample_counter = 0

        scene_id = self.config.scene.get("scene_id", "apartment_1")

        logger.info(
            "Starting v10 dataset generation: %d motions, %d candidate viewpoints...",
            len(target_motions),
            len(sampled_candidates),
        )

        try:
            for m_idx, m_id in enumerate(target_motions):
                player = self.motion_mgr.get_motion_player(m_id)
                player.reset()

                # 提取动作类别与标签
                try:
                    act_class = ActionClassV10.from_str(player.action_class).value
                except ValueError:
                    act_class = player.action_class

                act_label = player.action_label
                total_f = player.total_frames
                frame_indices = list(range(0, total_f, max(1, frame_step)))
                if max_frames_per_motion and len(frame_indices) > max_frames_per_motion:
                    frame_indices = frame_indices[:max_frames_per_motion]

                logger.info(
                    "[%d/%d] Processing motion '%s' (class: %s, %d frames)...",
                    m_idx + 1,
                    len(target_motions),
                    m_id,
                    act_class,
                    len(frame_indices),
                )

                for f_idx in frame_indices:
                    player.seek(f_idx)
                    joints_pose, root_trans = player.step(advance=False)

                    # 驱动 Humanoid 动作
                    humanoid.apply_motion_frame(joints_pose, root_transform_mat=root_trans)
                    gt_keypoints = humanoid.get_gt_joint_positions()

                    # 在各候选视点下渲染并采集 RGB-D
                    for vp in sampled_candidates:
                        # 移动机器人到底盘位置并朝向人体
                        robot.set_pose(position=vp.position, yaw_deg=vp.yaw_deg)

                        rgb, depth, cam_pose = capture.capture_frame()

                        sample_id = f"v10_sample_{sample_counter:06d}"
                        sample_counter += 1

                        s = self.builder.build_and_save_sample(
                            sample_id=sample_id,
                            scene_id=scene_id,
                            motion_id=m_id,
                            action_label=act_class,
                            frame_idx=f_idx,
                            view_id=vp.viewpoint_id,
                            rgb_array=rgb,
                            depth_array=depth,
                            camera_pose=cam_pose,
                            gt_skeleton=gt_keypoints,
                            extra_metadata={
                                "action_detail_label": act_label,
                                "viewpoint_radius": vp.radius,
                                "viewpoint_angle_deg": vp.angle_deg,
                                "human_position": [round(float(x), 3) for x in h_pos],
                                "human_yaw_deg": round(float(human_yaw_deg), 1),
                            },
                        )
                        all_samples.append(s)

        finally:
            env_adapter.close()

        # 保存全量数据集 manifest
        self.builder.save_dataset_manifest(all_samples)
        logger.info("Successfully generated %d v10 RGB-D samples!", len(all_samples))
        return all_samples
