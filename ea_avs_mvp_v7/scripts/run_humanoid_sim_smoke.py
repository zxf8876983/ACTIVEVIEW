"""
v7.0 最小闭环端到端 Smoke Test 脚本 —— run_humanoid_sim_smoke.py
============================================================

功能：
    1. 初始化 Habitat 室内仿真场景 (apartment_1.glb)；
    2. 加载 KinematicHumanoid (neutral_0)；
    3. 加载并回放 AMASS 动作 (优先 fall_related 跌倒动作)；
    4. 控制机器人相机采集 RGB-D 观测并提取 3D 人体骨架 GT；
    5. 保存观测图片与 JSON 元数据至外部数据目录并输出验证结果。

运行方式：
    python -m ea_avs_mvp_v7.scripts.run_humanoid_sim_smoke [--num-frames 5] [--action fall_related]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from tools.motion_assets.data_paths import (
    get_data_root,
    from_relative_data_path,
    to_relative_data_path,
)
from ea_avs_mvp_v7.humanoid.humanoid_loader import resolve_humanoid_assets
from ea_avs_mvp_v7.humanoid.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.motion.motion_converter import convert_single_amass_motion
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.simulation.scene_loader import HabitatSceneLoader
from ea_avs_mvp_v7.simulation.robot_sensor import RobotSensorRig
from ea_avs_mvp_v7.simulation.observation_generator import ObservationGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v70_smoke_test")


def main():
    parser = argparse.ArgumentParser(description="EA-AVS-MVP v7.0 Minimum Humanoid Simulation Smoke Test")
    parser.add_argument("--config", type=str, default="ea_avs_mvp_v7/configs/v70_humanoid_sim.yaml", help="Config file path")
    parser.add_argument("--action", type=str, default="fall_related", help="Target action class (default: fall_related)")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to capture (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for observations")
    args = parser.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_root = get_data_root()
    manifest_path = data_root / cfg.get("motion", {}).get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_path.exists():
        logger.error("Motion manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. 选取目标动作 (优先 fall_related)
    selected_item = next((m for m in manifest if m.get("target_class") == args.action), None)
    if selected_item is None:
        selected_item = manifest[0]

    logger.info(
        "Selected Motion: [%s] sid=%s, label='%s', file=%s",
        selected_item.get("target_class"),
        selected_item.get("babel_sid"),
        selected_item.get("proc_label"),
        selected_item.get("local_motion_path"),
    )

    # 2. 准备/转换 Motion PKL
    humanoid_assets = resolve_humanoid_assets(cfg)
    target_class = selected_item.get("target_class", "action")
    sid = selected_item.get("babel_sid", "0")
    converted_dir = data_root / cfg.get("motion", {}).get("converted_dir", "assets/motions/converted")
    expected_pkl = converted_dir / f"{target_class}_{sid}.pkl"

    if not expected_pkl.exists():
        logger.info("Motion PKL not found on disk, converting on-the-fly...")
        expected_pkl = convert_single_amass_motion(
            manifest_item=selected_item,
            urdf_path=humanoid_assets.urdf_path,
            output_dir=converted_dir,
        )
    else:
        logger.info("Using existing converted motion: %s", expected_pkl)

    # 3. 初始化 Habitat 场景与模拟器
    logger.info("Initializing Habitat scene loader...")
    scene_loader = HabitatSceneLoader(cfg)
    sim = scene_loader.create_simulator()

    # 4. 加载 Humanoid Agent
    logger.info("Loading HumanoidAgent into simulation...")
    humanoid = HumanoidAgent(sim, config=cfg, assets=humanoid_assets)
    humanoid.load()

    # 放置 Humanoid 在场景安全区域
    humanoid_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    humanoid.set_base_pose(humanoid_pos, yaw_rad=0.0)

    # 5. 初始化机器人相机 Rig
    robot = RobotSensorRig(sim, config=cfg)
    # 放置机器人在人体前方观察 (Z+2.0m, 相机高度 1.2m, 面对人体 yaw 180 deg)
    camera_pos = np.array([0.0, 0.1, 2.0], dtype=np.float32)
    camera_yaw_deg = 180.0
    robot.set_pose(camera_pos, camera_yaw_deg)

    # 6. 初始化 Motion Player 与 Observation Generator
    player = MotionPlayer(expected_pkl, playback_fps=float(cfg.get("motion", {}).get("playback_fps", 30.0)))
    obs_generator = ObservationGenerator(
        scene_loader=scene_loader,
        humanoid_agent=humanoid,
        robot_sensor=robot,
        config=cfg,
    )

    # 7. 执行序列观测采集
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "runs" / "v70_smoke_test"
    records = obs_generator.run_sequence(
        motion_player=player,
        camera_pos=camera_pos,
        camera_yaw_deg=camera_yaw_deg,
        output_dir=output_dir,
        frame_step=1,
        max_frames=args.num_frames,
    )

    # 8. 校验产出
    sim.close()

    print("\n" + "=" * 65)
    print("[v7.0 Smoke Test Verification Results]")
    print(f"  - Action Class:       {player.action_class}")
    print(f"  - Action Label:       {player.action_label}")
    print(f"  - Captured Frames:    {len(records)}")
    print(f"  - Output Directory:   {output_dir}")
    if records:
        r0 = records[0]
        print(f"  - RGB Sample:         {r0.rgb_relative_path}")
        print(f"  - Depth Sample:       {r0.depth_relative_path}")
        print(f"  - GT Joint Count:     {len(r0.human_pose_gt_world)}")
        print(f"  - Camera Position:    {r0.camera_position}")
    print("=" * 65)

    if len(records) > 0:
        print("[Status] PASS: v7.0 Humanoid simulation & RGB-D observation minimum loop verified!\n")
        sys.exit(0)
    else:
        print("[Status] FAIL: No observations captured!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
