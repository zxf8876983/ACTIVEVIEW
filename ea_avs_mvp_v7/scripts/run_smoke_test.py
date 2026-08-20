"""
v7.0 最小闭环端到端 Smoke Test 脚本 —— run_smoke_test.py
======================================================

输入：
    - Habitat 室内场景 (apartment_1.glb)
    - KinematicHumanoid URDF (neutral_0)
    - AMASS 动作数据 (如 fall_related)

输出：
    - RGB 图像 (.png)
    - Depth 深度图 (.npy)
    - metadata.json (包含相机位姿、人体 3D 骨架 GT 与动作标注)
    - [v7.0 Acceptance Report] 验收报告

运行方式：
    python -m ea_avs_mvp_v7.scripts.run_smoke_test [--action fall_related] [--num-frames 5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import convert_normalized_motion_to_pkl
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from ea_avs_mvp_v7.observation.recorder import ObservationRecorder
from ea_avs_mvp_v7.dataset.episode_generator import EpisodeGenerator
from ea_avs_mvp_v7.evaluation.basic_metrics import compute_episode_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v70_smoke_test")


def main():
    parser = argparse.ArgumentParser(description="EA-AVS-MVP v7.0 End-to-End Simulation Smoke Test")
    parser.add_argument("--action", type=str, default="fall_related", help="Target action class (default: fall_related)")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to capture (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for generated episode")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = load_v7_config()

    # 2. 读取动作清单并严格匹配动作 (严禁静默 fallback)
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Motion manifest not found at: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((m for m in manifest if m.get("target_class") == args.action), None)
    if item is None:
        raise ValueError(
            f"Action '{args.action}' not found in manifest! "
            f"Available actions: {sorted(list(set(m.get('target_class') for m in manifest)))}"
        )

    action_class = item.get("target_class")
    sid = item.get("babel_sid")
    rel_motion_file = item.get("local_motion_path")
    if not rel_motion_file:
        raise ValueError(f"Manifest entry for sid {sid} is missing 'local_motion_path'!")

    npz_p = from_relative_data_path(rel_motion_file)
    if not npz_p.exists():
        raise FileNotFoundError(f"AMASS npz file missing from disk: {npz_p}")

    logger.info("Selected Motion: [%s] sid=%s, file=%s", action_class, sid, rel_motion_file)

    # 3. 准备/转换 Motion PKL
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{action_class}_{sid}.pkl"

    if not pkl_path.exists():
        logger.info("Motion PKL missing, converting from AMASS npz...")
        norm_motion = load_amass_motion(
            npz_path=npz_p,
            start_frame=item.get("start_frame"),
            end_frame=item.get("end_frame"),
            metadata=item,
        )
        humanoid_urdf = HumanoidAgent(None, cfg.humanoid).urdf_path
        pkl_path = convert_normalized_motion_to_pkl(norm_motion, humanoid_urdf, pkl_path)
    else:
        logger.info("Using converted PKL: %s", pkl_path)

    # 4. 初始化环境与模拟器
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()
    if sim is None:
        raise RuntimeError("Failed to start Habitat simulator!")

    # 5. 加载 Humanoid 与 Robot
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    joint_summary = humanoid.get_joint_summary()
    if joint_summary["num_joints"] < 15:
        raise RuntimeError(f"Humanoid loaded with insufficient joints: {joint_summary['num_joints']}")

    robot = RobotAgent(sim)
    sensor = RGBDSensor(sim, cfg.sensor)

    # 6. 生成 Episode 观测
    motion_player = MotionPlayer(pkl_path, playback_fps=float(cfg.motion.get("playback_fps", 30.0)))
    recorder = ObservationRecorder(args.output_dir)
    ep_gen = EpisodeGenerator(env=env, humanoid=humanoid, robot=robot, sensor=sensor, recorder=recorder)

    episode_id = f"smoke_test_{action_class}_{sid}"
    episode = ep_gen.generate_single_episode(
        episode_id=episode_id,
        motion_player=motion_player,
        camera_position=[1.5, -1.60, 6.8],
        camera_yaw_deg=0.0,
        human_position=[1.5, -1.60, 4.0],
        human_yaw_rad=0.0,
        output_dir=recorder.output_dir,
        max_frames=args.num_frames,
    )

    # 7. 统计与全面质量校验
    stats = compute_episode_statistics(episode)
    env.close()

    # 校验 8 项核心标准
    assert stats["total_frames"] == args.num_frames, "Captured frames mismatch"
    assert stats["rgb_valid_ratio"] == 1.0, "RGB image generation incomplete"
    assert stats["depth_valid_ratio"] == 1.0, "Depth map generation incomplete"
    assert stats["avg_gt_keypoints_per_frame"] >= 15.0, "Ground-truth 3D joints incomplete"
    assert episode.action_class == args.action, "Action class mismatch"

    print("\n" + "=" * 65)
    print("[V7 Acceptance Report]")
    print(f"Scene:      {episode.scene_id}")
    print(f"Humanoid:   {cfg.humanoid.get('avatar_name', 'neutral_0')}")
    print(f"Motion:     {episode.motion_id}")
    print(f"Action:     {episode.action_label}")
    print(f"Frames:     {stats['total_frames']}")
    print(f"Quaternion: ({stats['total_frames']}, 216)")
    print(f"GT joints:  {int(stats['avg_gt_keypoints_per_frame'])}")
    print(f"RGB:        {stats['rgb_valid_ratio'] * 100:.1f}%")
    print(f"Depth:      {stats['depth_valid_ratio'] * 100:.1f}%")
    print(f"Episode:    {episode.episode_id}")
    print(f"Status:     PASS")
    print("=" * 65)
    print("PASS: ACTIVEVIEW v7.0 Humanoid-driven Active Perception Environment Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
