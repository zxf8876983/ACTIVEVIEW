"""
v7.0 最小闭环端到端 Smoke Test 脚本 —— run_smoke_test.py
======================================================

输入：
    - Habitat 室内场景 (apartment_1.glb)
    - KinematicHumanoid URDF (neutral_0)
    - AMASS 跌倒动作数据 (fall_related)

输出：
    - RGB 图像 (.png)
    - Depth 深度图 (.npy)
    - 相机位姿、人体 3D 骨架 GT 与动作标注 (.json)
    - Episode 汇总清单

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

    # 2. 读取动作清单并选取动作
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        logger.error("Motion manifest not found: %s", manifest_p)
        sys.exit(1)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((m for m in manifest if m.get("target_class") == args.action), manifest[0])
    action_class = item.get("target_class", "action")
    sid = item.get("babel_sid", "0")
    logger.info("Selected Motion: [%s] sid=%s, file=%s", action_class, sid, item.get("local_motion_path"))

    # 3. 准备/转换 Motion PKL
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{action_class}_{sid}.pkl"

    if not pkl_path.exists():
        logger.info("Motion PKL missing, converting from AMASS npz...")
        npz_p = from_relative_data_path(item["local_motion_path"])
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

    # 5. 加载 Humanoid 与 Robot
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()

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
        camera_position=[0.0, 0.1, 2.0],
        camera_yaw_deg=180.0,
        human_position=[0.0, 0.1, 0.0],
        human_yaw_rad=0.0,
        output_dir=recorder.output_dir,
        max_frames=args.num_frames,
    )

    # 7. 统计与验证
    stats = compute_episode_statistics(episode)
    env.close()

    print("\n" + "=" * 65)
    print("[v7.0 Smoke Test Acceptance Results]")
    print(f"  - Episode ID:         {episode.episode_id}")
    print(f"  - Scene ID:           {episode.scene_id}")
    print(f"  - Action Class:       {episode.action_class}")
    print(f"  - Action Label:       {episode.action_label}")
    print(f"  - Captured Frames:    {stats['total_frames']}")
    print(f"  - RGB Valid Ratio:    {stats['rgb_valid_ratio'] * 100:.1f}%")
    print(f"  - Depth Valid Ratio:  {stats['depth_valid_ratio'] * 100:.1f}%")
    print(f"  - Avg 3D GT Joints:   {stats['avg_gt_keypoints_per_frame']:.1f}")
    print("=" * 65)

    if stats["is_complete"]:
        print("[Status] PASS: v7.0 Simulation environment end-to-end verified!\n")
        sys.exit(0)
    else:
        print("[Status] FAIL: Incomplete episode data!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
