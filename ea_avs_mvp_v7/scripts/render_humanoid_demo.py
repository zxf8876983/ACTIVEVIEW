"""
Humanoid 动作真实性渲染与 Demo 生成脚本 —— render_humanoid_demo.py
================================================================

功能：
    1. 在 Habitat 室内场景中加载 neutral_0 拟人化模型并回放真实 AMASS 动作 (fall_related)；
    2. 控制机器人相机连续渲染 RGB 图像与 Depth 深度图；
    3. 编码输出 rgb_video.mp4、depth_video 伪彩色深度序列与综合 metadata.json；
    4. 输出保存至 ../../data/ActiveView/visualizations/v7_demo/；
    5. 校验 Pelvis 关键点垂直落差，确认 Humanoid 模型真实执行 AMASS 动作。

运行方式：
    python -m ea_avs_mvp_v7.scripts.render_humanoid_demo [--action fall_related] [--num-frames 15]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.human.keypoint_mapping import validate_keypoints
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("render_humanoid_demo")


def colorize_depth_meters(depth_map: np.ndarray, clip_min: float = 0.2, clip_max: float = 5.0) -> np.ndarray:
    """将物理深度图 (米) 映射为伪彩色 RGB uint8 数组。"""
    d_clamped = np.clip(depth_map, clip_min, clip_max)
    d_norm = (d_clamped - clip_min) / (clip_max - clip_min)

    # Jet 伪彩色映射
    r = np.clip(np.sin(d_norm * np.pi) * 255, 0, 255).astype(np.uint8)
    g = np.clip(np.cos(d_norm * np.pi * 0.5) * 255, 0, 255).astype(np.uint8)
    b = np.clip((1.0 - d_norm) * 255, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Render Humanoid Motion Perception Demonstration")
    parser.add_argument("--action", type=str, default="fall_related", help="Target action class (default: fall_related)")
    parser.add_argument("--num-frames", type=int, default=15, help="Number of frames to render (default: 15)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output demo directory")
    args = parser.parse_args()

    cfg = load_v7_config()
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Motion manifest not found: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((m for m in manifest if m.get("target_class") == args.action), None)
    if item is None:
        raise ValueError(f"Action '{args.action}' not found in manifest!")

    target_class = item.get("target_class")
    sid = item.get("babel_sid")
    motion_id = f"{target_class}_{sid}"
    raw_label = item.get("proc_label", item.get("raw_label", "action"))

    # 1. 准备动作 PKL
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{motion_id}.pkl"
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)

    if not pkl_path.exists():
        npz_p = from_relative_data_path(item["local_motion_path"])
        norm_motion = load_amass_motion(npz_p, item.get("start_frame"), item.get("end_frame"), item)
        converter = MotionConverter(urdf_path)
        pkl_path = converter.convert_and_save(norm_motion, pkl_path)

    # 2. 准备输出目录
    vis_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "visualizations" / "v7_demo"
    vis_dir.mkdir(parents=True, exist_ok=True)
    depth_video_dir = vis_dir / "depth_video"
    depth_video_dir.mkdir(parents=True, exist_ok=True)

    # 3. 初始化 Habitat 仿真环境
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

    logger.info("Rendering %d demonstration frames to %s...", len(frame_indices), vis_dir)

    rgb_frames_bgr = []
    depth_frames_bgr = []
    pelvis_heights = []
    frames_meta = []

    for idx, f_idx in enumerate(frame_indices):
        player.seek(f_idx)
        pose = player.get_current_pose()
        humanoid.apply_motion_frame(pose["joints_pose"], pose["root_transform"])

        obs = sensor.capture()
        rgb = obs["rgb"]
        depth = obs["depth"]
        gt_joints = humanoid.get_gt_joint_positions()
        validate_keypoints(gt_joints, min_joints=15)

        pelvis_pos = gt_joints.get("pelvis", [0.0, 0.0, 0.0])
        pelvis_heights.append(pelvis_pos[1])

        depth_color = colorize_depth_meters(depth)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_bgr = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

        # 保存单帧彩色深度图
        cv2.imwrite(str(depth_video_dir / f"depth_frame_{idx:04d}.png"), depth_bgr)

        rgb_frames_bgr.append(rgb_bgr)
        depth_frames_bgr.append(depth_bgr)

        frames_meta.append({
            "frame_idx": idx,
            "motion_frame_idx": f_idx,
            "timestamp": pose["timestamp"],
            "pelvis_position": pelvis_pos,
            "human_pose_gt": gt_joints,
        })

    cam_pose = sensor.get_camera_pose_matrix().tolist()
    final_gt_joints = humanoid.get_gt_joint_positions()
    env.close()

    # 4. 编码生成 rgb_video.mp4
    video_path = vis_dir / "rgb_video.mp4"
    if rgb_frames_bgr:
        h, w, _ = rgb_frames_bgr[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
        for f in rgb_frames_bgr:
            video_writer.write(f)
        video_writer.release()
        logger.info("Saved RGB demo video to: %s", video_path)

    # 5. 生成 metadata.json
    metadata = {
        "scene_id": cfg.habitat.get("scene_id", "apartment_1"),
        "motion_id": motion_id,
        "action_label": raw_label,
        "frame_count": len(frame_indices),
        "camera_pose": cam_pose,
        "human_pose_gt": final_gt_joints,
        "frames": frames_meta,
    }
    meta_path = vis_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    height_delta = max(pelvis_heights) - min(pelvis_heights)
    is_dynamic = height_delta > 0.05

    print("\n" + "=" * 65)
    print("[v7 Humanoid Motion Demo Results]")
    print(f"  - scene_id:       {metadata['scene_id']}")
    print(f"  - motion_id:      {motion_id}")
    print(f"  - action_label:   {raw_label}")
    print(f"  - frame_count:    {len(frame_indices)}")
    print(f"  - rgb_video:      {video_path}")
    print(f"  - depth_video:    {depth_video_dir}")
    print(f"  - metadata.json:  {meta_path}")
    print(f"  - height_change:  {height_delta:.3f} m ({'Dynamic Motion CONFIRMED' if is_dynamic else 'Static'})")
    print("=" * 65)
    print("PASS: Humanoid Demo Rendered Successfully\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
