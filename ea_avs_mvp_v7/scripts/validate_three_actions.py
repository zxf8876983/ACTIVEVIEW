"""
三类典型动作全量验证脚本 —— validate_three_actions.py
=====================================================

功能：
    1. 针对 3 类典型老人动作 (standing, sitting, fall_related) 逐一运行仿真闭环；
    2. 验证：Habitat 初始化 -> Humanoid 加载 -> 动作转换 -> 动作播放 -> RGB-D 生成 -> 3D GT 生成；
    3. 输出标准化 validation_report.json 至 data/ActiveView/runs/validation_report.json。

运行方式：
    python -m ea_avs_mvp_v7.scripts.validate_three_actions [--frames-per-action 5]
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
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.human.keypoint_mapping import validate_keypoints
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_three_actions")

TARGET_THREE_ACTIONS = ["standing", "sitting", "fall_related"]


def validate_single_action(
    action_class: str,
    manifest: list,
    cfg,
    urdf_path: Path,
    num_frames: int = 5,
) -> dict:
    """验证单个动作类别的端到端执行结果。"""
    item = next((m for m in manifest if m.get("target_class") == action_class), None)
    if item is None:
        raise ValueError(f"Action '{action_class}' not found in motion manifest!")

    sid = item.get("babel_sid")
    motion_id = f"{action_class}_{sid}"
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    pkl_path = converted_dir / f"{motion_id}.pkl"

    # 1. 动作转换
    if not pkl_path.exists():
        npz_p = from_relative_data_path(item["local_motion_path"])
        norm_motion = load_amass_motion(npz_p, item.get("start_frame"), item.get("end_frame"), item)
        converter = MotionConverter(urdf_path)
        pkl_path = converter.convert_and_save(norm_motion, pkl_path)

    # 2. 仿真环境实例化
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()
    humanoid_loaded = False
    motion_played = False
    rgb_generated = False
    depth_generated = False
    gt_pose_generated = False

    try:
        humanoid = HumanoidAgent(sim, cfg.humanoid)
        humanoid.load()
        humanoid.set_base_pose([0.0, 0.1, 0.0], 0.0)
        humanoid_loaded = True

        robot = RobotAgent(sim)
        robot.set_pose([0.0, 0.1, 2.0], 180.0)
        sensor = RGBDSensor(sim, cfg.sensor)

        player = MotionPlayer(pkl_path, playback_fps=float(cfg.motion.get("playback_fps", 30.0)))
        step_interval = max(1, player.total_frames // num_frames)
        frame_indices = list(range(0, player.total_frames, step_interval))[:num_frames]

        captured_count = 0
        for f_idx in frame_indices:
            player.seek(f_idx)
            pose = player.get_current_pose()
            humanoid.apply_motion_frame(pose["joints_pose"], pose["root_transform"])

            obs = sensor.capture()
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            gt_joints = humanoid.get_gt_joint_positions()
            validate_keypoints(gt_joints, min_joints=15)

            if rgb is not None and rgb.shape == (480, 640, 3):
                rgb_generated = True
            if depth is not None and depth.shape == (480, 640):
                depth_generated = True
            if len(gt_joints) >= 15:
                gt_pose_generated = True

            captured_count += 1

        if captured_count == num_frames:
            motion_played = True

    finally:
        env.close()

    status = "PASS" if (humanoid_loaded and motion_played and rgb_generated and depth_generated and gt_pose_generated) else "FAIL"

    return {
        "action": action_class,
        "motion_id": motion_id,
        "frames": num_frames,
        "humanoid_loaded": humanoid_loaded,
        "motion_played": motion_played,
        "rgb_generated": rgb_generated,
        "depth_generated": depth_generated,
        "gt_pose_generated": gt_pose_generated,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Three Core Humanoid Actions")
    parser.add_argument("--frames-per-action", type=int, default=5, help="Number of frames per action to test")
    args = parser.parse_args()

    cfg = load_v7_config()
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Motion manifest not found: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)
    report_list = []

    print("\n" + "=" * 65)
    print("[v7.0 Three-Action Validation Suite]")
    print(f"Target Actions: {TARGET_THREE_ACTIONS}")
    print("=" * 65)

    all_passed = True
    for action in TARGET_THREE_ACTIONS:
        logger.info("Testing action: '%s'...", action)
        res = validate_single_action(
            action_class=action,
            manifest=manifest,
            cfg=cfg,
            urdf_path=urdf_path,
            num_frames=args.frames_per_action,
        )
        report_list.append(res)
        print(f"  - [{action}] Motion ID: {res['motion_id']} | Frames: {res['frames']} | Status: {res['status']}")
        if res["status"] != "PASS":
            all_passed = False

    # 保存 validation_report.json
    out_dir = get_data_root() / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_list, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"Validation Report Saved: {report_path}")
    if all_passed:
        print("PASS: All 3 Target Actions Validated Successfully\n")
        sys.exit(0)
    else:
        print("FAIL: One or more actions failed validation\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
