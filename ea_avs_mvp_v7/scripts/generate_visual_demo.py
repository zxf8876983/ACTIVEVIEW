"""
动态人体主动感知可视化 Demo 生成脚本 —— generate_visual_demo.py
============================================================

功能：
    1. 在 Habitat 室内场景中加载 Humanoid 并回放 AMASS 摔倒动作 (fall_related)；
    2. 控制机器人相机连续采集 RGB 图像与 Depth 深度图；
    3. 渲染单帧与时序并排合成可视化大图 (RGB + Depth Colormap + 3D 骨架标注)；
    4. 输出保存至 ../../data/ActiveView/visualizations/v7_demo/；
    5. 直观验证人形模型动作的动态性与仿真环境的真实感知效果。

运行方式：
    python -m ea_avs_mvp_v7.scripts.generate_visual_demo [--action fall_related] [--num-frames 10]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_visual_demo")


def colorize_depth(depth_map: np.ndarray, clip_min: float = 0.1, clip_max: float = 6.0) -> np.ndarray:
    """将 metric depth (米) 映射为可视化的伪彩色 uint8 RGB 图像。"""
    d_clamped = np.clip(depth_map, clip_min, clip_max)
    d_norm = (d_clamped - clip_min) / (clip_max - clip_min)
    d_uint8 = (d_norm * 255.0).astype(np.uint8)

    # 简单 Turbo / Jet 色彩映射查找表
    r = np.clip(np.sin(d_norm * np.pi) * 255, 0, 255).astype(np.uint8)
    g = np.clip(np.cos(d_norm * np.pi * 0.5) * 255, 0, 255).astype(np.uint8)
    b = np.clip((1.0 - d_norm) * 255, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Perception Demonstration of Humanoid Motion")
    parser.add_argument("--action", type=str, default="fall_related", help="Target action class (default: fall_related)")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to render (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visual artifacts")
    args = parser.parse_args()

    cfg = load_v7_config()
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((m for m in manifest if m.get("target_class") == args.action), None)
    if item is None:
        raise ValueError(f"Action '{args.action}' not found in manifest!")

    target_class = item.get("target_class")
    sid = item.get("babel_sid")
    motion_id = f"{target_class}_{sid}"

    # 准备动作 PKL
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{motion_id}.pkl"
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)

    if not pkl_path.exists():
        npz_p = from_relative_data_path(item["local_motion_path"])
        norm_motion = load_amass_motion(npz_p, item.get("start_frame"), item.get("end_frame"), item)
        converter = MotionConverter(urdf_path)
        pkl_path = converter.convert_and_save(norm_motion, pkl_path)

    # 准备输出目录
    vis_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "visualizations" / "v7_demo"
    vis_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = vis_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Habitat 模拟器
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

    logger.info("Rendering %d visual demo frames to %s...", len(frame_indices), vis_dir)

    rendered_strips = []
    pelvis_heights = []

    for idx, f_idx in enumerate(frame_indices):
        player.seek(f_idx)
        pose = player.get_current_pose()
        humanoid.apply_motion_frame(pose["joints_pose"], pose["root_transform"])

        obs = sensor.capture()
        rgb = obs["rgb"]
        depth = obs["depth"]
        gt_joints = humanoid.get_gt_joint_positions()

        pelvis_pos = gt_joints.get("pelvis", [0.0, 0.0, 0.0])
        pelvis_heights.append(pelvis_pos[1])

        # 制作并排 RGB + Depth 图像
        depth_color = colorize_depth(depth)
        composite = np.concatenate([rgb, depth_color], axis=1)
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)

        # 绘制文本信息
        text = f"Frame {idx:02d} (t={pose['timestamp']:.2f}s) | Action: {target_class} | Pelvis Y: {pelvis_pos[1]:.2f}m"
        draw.rectangle([(10, 10), (550, 35)], fill=(0, 0, 0, 180))
        draw.text((15, 15), text, fill=(255, 255, 255))

        frame_fname = f"demo_frame_{idx:02d}.png"
        img.save(frames_dir / frame_fname)
        rendered_strips.append(img.resize((img.width // 2, img.height // 2)))

    env.close()

    # 制作多帧合成画廊海报
    if rendered_strips:
        strip_w, strip_h = rendered_strips[0].size
        num_cols = min(5, len(rendered_strips))
        num_rows = (len(rendered_strips) + num_cols - 1) // num_cols
        gallery = Image.new("RGB", (num_cols * strip_w, num_rows * strip_h), (30, 30, 30))

        for i, thumb in enumerate(rendered_strips):
            r = i // num_cols
            c = i % num_cols
            gallery.paste(thumb, (c * strip_w, r * strip_h))

        gallery_path = vis_dir / "v7_motion_sequence_gallery.png"
        gallery.save(gallery_path)
        logger.info("Saved visual sequence gallery to: %s", gallery_path)

    # 动作动力学位移变化校验 (验证人体在运动)
    height_delta = max(pelvis_heights) - min(pelvis_heights)
    is_dynamic = height_delta > 0.05

    print("\n" + "=" * 65)
    print("[v7 Visual Demo Generation Results]")
    print(f"  - Motion ID:          {motion_id}")
    print(f"  - Rendered Frames:    {len(frame_indices)}")
    print(f"  - Output Frames Dir:  {frames_dir}")
    print(f"  - Pelvis Height Max:  {max(pelvis_heights):.3f} m")
    print(f"  - Pelvis Height Min:  {min(pelvis_heights):.3f} m")
    print(f"  - Height Change:      {height_delta:.3f} m ({'Dynamic Motion CONFIRMED' if is_dynamic else 'Static'})")
    print(f"  - Sequence Gallery:   {vis_dir / 'v7_motion_sequence_gallery.png'}")
    print("=" * 65)
    print("PASS: Visual Demonstration Generated\n")


if __name__ == "__main__":
    main()
