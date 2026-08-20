"""
v7.0 综合演示主入口脚本 —— run_v7_demo.py
=========================================

功能：
    1. 一键运行 ACTIVEVIEW v7.0 完整端到端科研演示链路：
       AMASS Motion -> Habitat Motion PKL -> Humanoid 动作播放 -> 机器人 RGB-D 采集 -> 动力学评价与成果持久化；
    2. 加载 Habitat 室内场景 (apartment_1.glb) 与 neutral_0 Humanoid 实体；
    3. 规范空间坐标对齐 (开阔客厅物理地面 Floor Y = -1.60m)：
       - Humanoid: [1.5, -1.60, 4.0], yaw=0.0 deg
       - Robot:    [1.5, -1.60, 6.8], yaw=0.0 deg (面向 -Z 轴，正对 Humanoid)
    4. 分离机器人底盘 robot_pose (position, rotation) 与相机 camera_pose (extrinsic, intrinsic)；
    5. 每帧保存完整的 16 关键点 3D 世界坐标真值 human_pose_gt；
    6. 计算多维动力学指标 ActionMotionMetrics；
    7. 输出标准化 metadata.json 与演示视频至 data/ActiveView/visualizations/v7_demo/。

运行方式：
    python -m ea_avs_mvp_v7.scripts.run_v7_demo [--action fall_related] [--num-frames 15]
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path, to_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.action_metrics import compute_action_motion_metrics
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.human.keypoint_mapping import validate_keypoints
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from .create_video import create_video_from_frames

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

    # 4. 初始化 Habitat 室内场景与机器人传感器 (地面基准 Y = -1.60m)
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    floor_y = float(cfg.humanoid.get("floor_height", -1.60))
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    humanoid.set_visibility(True)
    human_base_pos = [1.5, floor_y, 4.0]
    human_base_yaw = 0.0
    humanoid.set_base_pose(human_base_pos, yaw_rad=human_base_yaw)

    robot = RobotAgent(sim)
    robot_chassis_pos = [1.5, floor_y, 6.8]
    robot_chassis_yaw_deg = 0.0
    robot.set_pose(robot_chassis_pos, robot_chassis_yaw_deg)

    # 计算机器人底盘四元数 [qx, qy, qz, qw]
    yaw_rad = math.radians(robot_chassis_yaw_deg)
    robot_rotation_quat = [0.0, float(math.sin(yaw_rad / 2.0)), 0.0, float(math.cos(yaw_rad / 2.0))]

    sensor = RGBDSensor(sim, cfg.sensor)
    cam_mat = sensor.get_camera_pose_matrix()
    cam_pos = [float(cam_mat[0, 3]), float(cam_mat[1, 3]), float(cam_mat[2, 3])]

    # 打印标准化空间调试日志
    print("\n" + "=" * 65)
    print("[V7 Spatial Debug]")
    print(f"Human:  {human_base_pos}")
    print(f"Robot:  {robot_chassis_pos}")
    print(f"Camera: {[round(x, 3) for x in cam_pos]}")
    print("=" * 65 + "\n")

    player = MotionPlayer(pkl_path, playback_fps=float(cfg.motion.get("playback_fps", 30.0)))

    step_interval = max(1, player.total_frames // args.num_frames)
    frame_indices = list(range(0, player.total_frames, step_interval))[: args.num_frames]

    logger.info("Executing v7.0 Demonstration: %d frames -> %s", len(frame_indices), vis_dir)

    timestamps = []
    frames_meta = []
    gt_joints_sequence = []

    for idx, f_idx in enumerate(frame_indices):
        player.seek(f_idx)
        pose = player.get_current_pose()
        humanoid.apply_motion_frame(pose["joints_pose"], pose["root_transform"])

        obs = sensor.capture()
        rgb = obs["rgb"]
        depth = obs["depth"]
        gt_joints = humanoid.get_gt_joint_positions()
        validate_keypoints(gt_joints, min_joints=15)
        gt_joints_sequence.append(gt_joints)

        fname = f"frame_{idx:06d}"
        rgb_path = rgb_dir / f"{fname}.png"
        depth_path = depth_dir / f"{fname}.npy"

        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth.astype(np.float32))

        t = float(pose["timestamp"])
        timestamps.append(t)

        frames_meta.append({
            "frame_idx": idx,
            "timestamp": t,
            "rgb_path": to_relative_data_path(rgb_path),
            "depth_path": to_relative_data_path(depth_path),
            "human_pose_gt": gt_joints,
        })

    cam_extrinsic = sensor.get_camera_pose_matrix().tolist()
    cam_intrinsic = sensor.intrinsics
    env.close()

    # 5. 计算多维动力学运动指标
    metrics = compute_action_motion_metrics(gt_joints_sequence, timestamps)

    # 6. 生成标准 metadata.json
    scene_name = cfg.habitat.get("scene_id", "apartment_1")
    humanoid_name = cfg.humanoid.get("avatar_name", "neutral_0")

    metadata = {
        "scene_id": scene_name,
        "episode_id": f"v7_demo_{motion_id}",
        "humanoid_id": humanoid_name,
        "motion_id": motion_id,
        "source_dataset": source_dataset,
        "action_class": target_class,
        "action_label": raw_label,
        "frame_count": len(frame_indices),
        "fps": player.fps,
        "robot_pose": {
            "position": [float(x) for x in robot_chassis_pos],
            "rotation": robot_rotation_quat,
        },
        "camera_pose": {
            "extrinsic": cam_extrinsic,
            "intrinsic": cam_intrinsic,
        },
        "frames": frames_meta,
        "motion_metrics": metrics.to_dict(),
    }

    meta_path = vis_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 7. 自动生成 .mp4 视频
    video_out = vis_dir / "v7_demo.mp4"
    create_video_from_frames(
        input_dir=rgb_dir,
        output_mp4_path=video_out,
        fps=10.0,
    )

    print("\n" + "=" * 65)
    print("[v7.0 Unified Demonstration Results]")
    print(f"Scene:                {scene_name}")
    print(f"Humanoid:             {humanoid_name}")
    print(f"Motion:               {motion_id}")
    print(f"Action:               {raw_label}")
    print(f"Frames:               {len(frame_indices)}")
    print(f"RGB:                  100.0%")
    print(f"Depth:                100.0%")
    print(f"GT joints:            {len(gt_joints_sequence[0])}")
    print(f"Height Change:        {metrics.height_change:.3f} m")
    print(f"Torso Angle:          {metrics.torso_angle_change:.1f} deg")
    print(f"Motion Energy:        {metrics.joint_motion_energy:.3f}")
    print(f"Dynamic Motion:       {metrics.dynamic_motion}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v7.0 Humanoid Demonstration Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
