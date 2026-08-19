"""
人体朝向估计校准脚本 —— calibrate_estimated_orientation.py
=========================================================

功能：
    遍历多个不同的人体 GT 朝向角（Yaw），测量 2D 检测与双侧 3D 解剖几何推导出的
    朝向角与 GT 朝向角的差异，输出 mean / median / max yaw error 及 valid rate。
"""

import argparse
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.geometry import normalize_angle


def main():
    parser = argparse.ArgumentParser(description="Calibrate Estimated Human Yaw")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("[CalibrateYaw] Initializing Simulator and Humanoid...")
    runner = HabitatRunner(config)
    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    estimator = HumanStateEstimator(config)

    human_pos = runner.sample_navigable_point()
    test_yaws_deg = [0, 45, 90, 135, 180, -135, -90, -45]

    print("\n" + "=" * 75)
    print(f"{'GT Yaw (deg)':<15} | {'Est Yaw (deg)':<15} | {'Error (deg)':<15} | {'Status':<20}")
    print("=" * 75)

    errors = []
    valid_count = 0

    for deg in test_yaws_deg:
        gt_yaw = np.deg2rad(deg)
        humanoid.set_base_pose(human_pos, gt_yaw)
        humanoid.set_pose("standing")

        # 放置机器人在人体正前方 (机器人观察朝向正对人体)
        fwd = np.array([np.sin(gt_yaw), 0.0, np.cos(gt_yaw)], dtype=np.float32)
        robot_pos = runner.snap_point(human_pos + fwd * 2.0)
        robot_yaw = normalize_angle(gt_yaw + np.pi)

        obs = runner.render_at(robot_pos, robot_yaw)
        cam_state = runner.get_camera_state(robot_pos, robot_yaw)

        state = estimator.estimate(obs["rgb"], obs["depth"], cam_state)

        if state.human_yaw is not None:
            valid_count += 1
            est_deg = float(np.rad2deg(state.human_yaw))
            diff_deg = float(np.rad2deg(abs(normalize_angle(state.human_yaw - gt_yaw))))
            errors.append(diff_deg)
            status = state.yaw_source
            print(f"{deg:<15.1f} | {est_deg:<15.1f} | {diff_deg:<15.1f} | {status:<20}")
        else:
            print(f"{deg:<15.1f} | {'None':<15} | {'N/A':<15} | {state.failure_reason:<20}")

    print("=" * 75)
    if errors:
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        max_err = float(np.max(errors))
        print(f"[CalibrateYaw] Valid Count: {valid_count}/{len(test_yaws_deg)} ({valid_count/len(test_yaws_deg)*100:.1f}%)")
        print(f"[CalibrateYaw] Mean Absolute Yaw Error:   {mean_err:.2f}°")
        print(f"[CalibrateYaw] Median Absolute Yaw Error: {median_err:.2f}°")
        print(f"[CalibrateYaw] Max Absolute Yaw Error:    {max_err:.2f}°")
    else:
        print("[CalibrateYaw] No valid yaw estimations produced.")

    runner.close()


if __name__ == "__main__":
    main()
