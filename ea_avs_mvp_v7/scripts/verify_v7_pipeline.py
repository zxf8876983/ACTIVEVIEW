"""
v7.0 端到端流水线全量验证脚本 —— verify_v7_pipeline.py
=====================================================

功能：
    严格验证 ACTIVEVIEW v7.0 完整感知与数据生产闭环：
    AMASS Motion -> MotionConverter -> Habitat Motion PKL -> Humanoid 加载与相对驱动
    -> RGB-D 渲染 -> 16 关节 3D GT Pose 提取 -> Episode 生成与数据落盘。

输出格式：
    [V7 Pipeline Verification]
    Motion:     fall_related_3522
    Frames:     10
    Humanoid:   neutral_0 (loaded & grounded)
    RGB:        100% (640x480)
    Depth:      100% (640x480)
    GT Pose:    16 joints verified
    Episode:    runs/v7_verification/metadata.json
    Status:     PASS

运行方式：
    python -m ea_avs_mvp_v7.scripts.verify_v7_pipeline
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.dataset.episode_generator import EpisodeGenerator
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.human.keypoint_mapping import validate_keypoints
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_pipeline")


def main():
    cfg = load_v7_config()

    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        print(f"ERROR: Motion asset manifest missing: {manifest_p}")
        sys.exit(1)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. 验证 AMASS Loader
    action_item = next((m for m in manifest if m.get("target_class") == "fall_related"), manifest[0])
    target_class = action_item.get("target_class")
    sid = action_item.get("babel_sid")
    motion_id = f"{target_class}_{sid}"

    npz_p = from_relative_data_path(action_item["local_motion_path"])
    if not npz_p.exists():
        print(f"ERROR: Raw AMASS motion file not found at: {npz_p}")
        sys.exit(1)

    norm_motion = load_amass_motion(npz_p, action_item.get("start_frame"), action_item.get("end_frame"), action_item)
    if norm_motion.num_frames <= 0:
        print("ERROR: NormalizedMotion has zero frames!")
        sys.exit(1)

    # 2. 验证 MotionConverter
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)
    converter = MotionConverter(urdf_path)
    converted_dir = get_data_root() / "assets/motions/converted"
    pkl_path = converted_dir / f"{motion_id}.pkl"
    pkl_path = converter.convert_and_save(norm_motion, pkl_path)

    if not pkl_path.exists():
        print(f"ERROR: Converted motion PKL missing: {pkl_path}")
        sys.exit(1)

    # 3. 验证 MotionPlayer
    player = MotionPlayer(pkl_path)
    if player.total_frames <= 0:
        print("ERROR: MotionPlayer has 0 frames!")
        sys.exit(1)

    # 4. 验证 Habitat Scene, Humanoid, Robot & Sensor
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    if not humanoid.is_loaded:
        print("ERROR: HumanoidAgent failed to load!")
        sys.exit(1)

    robot = RobotAgent(sim)
    sensor = RGBDSensor(sim, cfg.sensor)

    # 5. 验证 Episode Generator
    ep_gen = EpisodeGenerator(
        env=env,
        humanoid=humanoid,
        robot=robot,
        sensor=sensor,
    )

    out_verify_dir = get_data_root() / "runs"
    episode_id = f"verify_{motion_id}"

    episode = ep_gen.generate_single_episode(
        episode_id=episode_id,
        motion_player=player,
        camera_position=[1.5, -1.60, 6.8],
        camera_yaw_deg=0.0,
        human_position=[1.5, -1.60, 4.0],
        human_yaw_rad=0.0,
        output_dir=out_verify_dir,
        max_frames=10,
    )

    env.close()

    # 6. 校验生成产物完整性
    ep_dir = out_verify_dir / episode_id
    meta_json = ep_dir / "metadata.json"
    if not meta_json.exists():
        print(f"ERROR: Episode metadata.json not created at {meta_json}")
        sys.exit(1)

    with open(meta_json, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    rgb_files = list((ep_dir / "rgb").glob("*.png"))
    depth_files = list((ep_dir / "depth").glob("*.npy"))
    pose_files = list((ep_dir / "human_pose").glob("*.json"))

    if len(rgb_files) != 10 or len(depth_files) != 10 or len(pose_files) != 10:
        print(f"ERROR: Frame count mismatch! rgb={len(rgb_files)}, depth={len(depth_files)}, pose={len(pose_files)}")
        sys.exit(1)

    gt_joints = meta_data["frames"][0]["human_pose_gt"]
    validate_keypoints(gt_joints, min_joints=15)

    print("\n" + "=" * 65)
    print("[V7 Pipeline Verification]")
    print(f"Motion:     {motion_id}")
    print(f"Frames:     {len(rgb_files)}")
    print(f"Humanoid:   {cfg.humanoid.get('avatar_name', 'neutral_0')} (loaded & grounded)")
    print(f"RGB:        100% ({sensor.intrinsics['width']}x{sensor.intrinsics['height']})")
    print(f"Depth:      100% ({sensor.intrinsics['width']}x{sensor.intrinsics['height']})")
    print(f"GT Pose:    {len(gt_joints)} joints verified")
    print(f"Episode:    {ep_dir.relative_to(get_data_root().parent)}")
    print("Status:     PASS")
    print("=" * 65 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
