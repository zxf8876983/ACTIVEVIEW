"""
深度采样与 3D 关节提升调试脚本 —— debug_depth_lifting.py
======================================================

功能：
    详细检查关键点局部深度采样、MAD 波动过滤、3D 世界坐标逆投影
    以及与 Humanoid GT 关节世界坐标的逐点对齐误差。
"""

import argparse
import os
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v6.human_state_estimator import HumanStateEstimator


def main():
    parser = argparse.ArgumentParser(description="Debug Depth Lifting & Joint 3D Error")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("[DebugLifting] Initializing HabitatRunner and HumanoidManager...")
    runner = HabitatRunner(config)
    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    human_pos = runner.sample_navigable_point()
    human_yaw = 0.0
    humanoid.set_base_pose(human_pos, human_yaw)
    humanoid.set_pose("standing")

    gt_info = get_humanoid_gt_skeleton(humanoid, strict=False)
    gt_skeleton = gt_info["skeleton"]

    robot_pos = runner.snap_point(human_pos + np.array([0.0, 0.0, 2.0], dtype=np.float32))
    robot_yaw = np.pi

    obs = runner.render_at(robot_pos, robot_yaw)
    camera_state = runner.get_camera_state(robot_pos, robot_yaw)

    estimator = HumanStateEstimator(config)
    state = estimator.estimate(obs["rgb"], obs["depth"], camera_state)

    print("\n" + "=" * 80)
    print(f"{'Keypoint':<16} | {'Status':<10} | {'Est 3D (X, Y, Z)':<24} | {'GT 3D (X, Y, Z)':<24} | {'Error (m)':<8}")
    print("=" * 80)

    for name in gt_skeleton.keys():
        gt_pt = gt_skeleton[name]
        joint = state.joints.get(name)
        if joint and joint.observable_3d and joint.position_world is not None:
            est_pt = joint.position_world
            err = float(np.linalg.norm(est_pt - gt_pt))
            status = "OBS_3D"
            est_str = f"[{est_pt[0]:.2f}, {est_pt[1]:.2f}, {est_pt[2]:.2f}]"
        elif name in state.proxy_full_skeleton:
            est_pt = state.proxy_full_skeleton[name]
            err = float(np.linalg.norm(est_pt - gt_pt))
            status = "PROXY"
            est_str = f"[{est_pt[0]:.2f}, {est_pt[1]:.2f}, {est_pt[2]:.2f}]"
        else:
            est_str = "None"
            err = -1.0
            status = "MISSING"

        gt_str = f"[{gt_pt[0]:.2f}, {gt_pt[1]:.2f}, {gt_pt[2]:.2f}]"
        err_str = f"{err:.3f}" if err >= 0 else "N/A"
        print(f"{name:<16} | {status:<10} | {est_str:<24} | {gt_str:<24} | {err_str:<8}")

    print("=" * 80)
    runner.close()


if __name__ == "__main__":
    main()
