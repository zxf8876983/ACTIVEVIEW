"""
v7.0 综合演示主入口脚本 —— run_v7_demo.py
=========================================

功能：
    1. 一键运行 ACTIVEVIEW v7.0 完整端到端科研演示链路：
       AMASS Motion -> Habitat Motion PKL -> Humanoid 动作播放 -> 机器人 RGB-D 采集 -> 成果持久化；
    2. 加载 Habitat 室内场景 (apartment_1.glb) 与 neutral_0 Humanoid 实体；
    3. 播放 fall_related 动作并采集多帧 RGB 图像与 Depth 深度图；
    4. 生成并输出完整 metadata.json 与可视化成果至 data/ActiveView/visualizations/v7_demo/。

运行方式：
    python -m ea_avs_mvp_v7.scripts.run_v7_demo [--action fall_related] [--num-frames 15]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path, to_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.human.keypoint_mapping import validate_keypoints
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v7_demo")


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v7.0 Unified Demonstration Pipeline")
    parser.add_argument("--action", type=str, default="fall_related", help="Action class to demonstrate (default: fall_related)")
    parser.add_argument("--num-frames", type=int, default=15, help="Number of frames to render (default: 15)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    cfg = load_v7_config()

    # 1. 查找目标动作清单
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Motion manifest not found: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((m for m in manifest if m.get("target_class") == args.action), None)
    if item is None:
        raise ValueError(f"Action '{args.action}' not found in manifest! Available: {[m.get('target_class') for m in manifest]}")

    target_class = item.get("target_class")
    sid = item.get("babel_sid")
    motion_id = f"{target_class}_{sid}"
    raw_label = item.get("proc_label", item.get("raw_label", "action"))
    source_dataset = item.get("source_dataset", item.get("dataset_name", "AMASS"))

    # 2. 准备/转换 Motion PKL
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{motion_id}.pkl"
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)

    if not pkl_path.exists():
        npz_p = from_relative_data_path(item["local_motion_path"])
        norm_motion = load_amass_motion(npz_p, item.get("start_frame"), item.get("end_frame"), item)
        converter = MotionConverter(urdf_path)
        pkl_path = converter.convert_and_save(norm_motion, pkl_path)

    # 3. 输出目录初始化
    vis_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "visualizations" / "v7_demo"
    rgb_dir = vis_dir / "rgb"
    depth_dir = vis_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # 4. 初始化 Habitat 室内场景与传感器
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    humanoid.set_base_pose([0.0, 0.1, 0.0], yaw_rad=0.0)

    robot = RobotAgent(sim)
    camera_pos = [0.0, 0.1, 2.0]
    camera_yaw = 180.0
    robot.set_pose(camera_pos, camera_yaw)

    sensor = RGBDSensor(sim, cfg.sensor)
    player = MotionPlayer(pkl_path, playback_fps=float(cfg.motion.get("playback_fps", 30.0)))

    step_interval = max(1, player.total_frames // args.num_frames)
    frame_indices = list(range(0, player.total_frames, step_interval))[: args.num_frames]

    logger.info("Executing v7.0 Demonstration: %d frames -> %s", len(frame_indices), vis_dir)

    timestamps = []
    frames_meta = []
    pelvis_heights = []

    for idx, f_idx in enumerate(frame_indices):
        player.seek(f_idx)
        pose = player.get_current_pose()
        humanoid.apply_motion_frame(pose["joints_pose"], pose["root_transform"])

        obs = sensor.capture()
        rgb = obs["rgb"]
        depth = obs["depth"]
        gt_joints = humanoid.get_gt_joint_positions()
        validate_keypoints(gt_joints, min_joints=15)

        fname = f"frame_{idx:06d}"
        rgb_path = rgb_dir / f"{fname}.png"
        depth_path = depth_dir / f"{fname}.npy"

        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth.astype(np.float32))

        t = float(pose["timestamp"])
        timestamps.append(t)
        pelvis_pos = gt_joints.get("pelvis", [0.0, 0.0, 0.0])
        pelvis_heights.append(pelvis_pos[1])

        frames_meta.append({
            "frame_idx": idx,
            "motion_frame_idx": f_idx,
            "timestamp": t,
            "rgb_path": to_relative_data_path(rgb_path),
            "depth_path": to_relative_data_path(depth_path),
            "human_pose_gt": gt_joints,
        })

    cam_pose = sensor.get_camera_pose_matrix().tolist()
    cam_intrinsics = sensor.intrinsics
    final_gt_joints = humanoid.get_gt_joint_positions()
    env.close()

    # 5. 生成综合 metadata.json
    metadata = {
        "scene_id": cfg.habitat.get("scene_id", "apartment_1"),
        "episode_id": f"v7_demo_{motion_id}",
        "humanoid_id": cfg.humanoid.get("avatar_name", "neutral_0"),
        "motion_id": motion_id,
        "source_dataset": source_dataset,
        "action_class": target_class,
        "action_label": raw_label,
        "frame_count": len(frame_indices),
        "fps": player.fps,
        "timestamps": timestamps,
        "robot_pose": [float(x) for x in camera_pos] + [float(camera_yaw)],
        "camera_pose": cam_pose,
        "camera_intrinsics": cam_intrinsics,
        "human_pose_gt": final_gt_joints,
        "frames": frames_meta,
    }

    meta_path = vis_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    height_delta = max(pelvis_heights) - min(pelvis_heights)
    is_dynamic = height_delta > 0.05

    print("\n" + "=" * 65)
    print("[v7.0 Unified Demonstration Results]")
    print(f"  - scene_id:       {metadata['scene_id']}")
    print(f"  - humanoid_id:    {metadata['humanoid_id']}")
    print(f"  - motion_id:      {metadata['motion_id']}")
    print(f"  - source_dataset: {metadata['source_dataset']}")
    print(f"  - action_class:   {metadata['action_class']}")
    print(f"  - action_label:   {metadata['action_label']}")
    print(f"  - frame_count:    {metadata['frame_count']}")
    print(f"  - fps:            {metadata['fps']:.1f}")
    print(f"  - rgb_dir:        {rgb_dir}")
    print(f"  - depth_dir:      {depth_dir}")
    print(f"  - metadata_file:  {meta_path}")
    print(f"  - height_change:  {height_delta:.3f} m ({'Dynamic Motion CONFIRMED' if is_dynamic else 'Static'})")
    print("=" * 65)
    print("PASS: v7.0 Unified Demonstration Completed\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
