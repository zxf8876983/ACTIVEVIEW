"""
GT 与估计状态批量对比脚本 —— compare_gt_estimated_state.py
=========================================================

功能：
    在多随机场景 Episode 下批量测试并对比 GT 人体状态与纯视觉估计状态，
    统计位置误差、朝向误差、3D 关节 MPJPE 等指标。
"""

import argparse
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.metrics import compute_state_estimation_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare GT and Estimated Human State")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"[CompareState] Initializing for {args.episodes} episodes...")
    runner = HabitatRunner(config)
    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    estimator = HumanStateEstimator(config)

    print("\n" + "=" * 90)
    print(f"{'Ep':<4} | {'Valid':<6} | {'Pos Err (m)':<12} | {'Yaw Err (deg)':<14} | {'Obs MPJPE (m)':<14} | {'Proxy MPJPE (m)':<16} | {'3D Kpts':<8}")
    print("=" * 90)

    all_metrics = []
    for ep in range(args.episodes):
        human_pos = runner.sample_navigable_point()
        human_yaw = float(np.random.uniform(-np.pi, np.pi))
        humanoid.set_base_pose(human_pos, human_yaw)
        humanoid.set_pose("standing")

        gt_info = get_humanoid_gt_skeleton(humanoid, strict=False)
        gt_skeleton = gt_info["skeleton"]

        # 采样机器人起始位置 (1.5m ~ 3.5m 且测地可达)
        max_tries = config.get("episode", {}).get("max_sampling_tries", 100)
        robot_pos = None
        for _ in range(max_tries):
            pt = runner.sample_navigable_point()
            if not runner.is_navigable(pt):
                continue
            dist = float(np.linalg.norm(pt - human_pos))
            if not (1.5 <= dist <= 3.5):
                continue
            if runner.geodesic_distance(pt, human_pos) == float("inf"):
                continue
            robot_pos = pt
            break

        if robot_pos is None:
            # Fallback
            robot_pos = runner.snap_point(human_pos + np.array([1.5, 0.0, 0.0], dtype=np.float32))

        dx = human_pos[0] - robot_pos[0]
        dz = human_pos[2] - robot_pos[2]
        robot_yaw = float(np.arctan2(dx, dz))

        obs = runner.render_at(robot_pos, robot_yaw)
        cam_state = runner.get_camera_state(robot_pos, robot_yaw)

        state = estimator.estimate(obs["rgb"], obs["depth"], cam_state)

        m = compute_state_estimation_metrics(state, human_pos, human_yaw, gt_skeleton)
        all_metrics.append(m)

        pos_err_str = f"{m['pos_error_m']:.3f}" if m['pos_error_m'] is not None else "N/A"
        yaw_err_str = f"{m['yaw_error_deg']:.2f}" if m['yaw_error_deg'] is not None else "N/A"
        obs_mpjpe_str = f"{m['observable_joint_error_mean_m']:.3f}" if m['observable_joint_error_mean_m'] is not None else "N/A"
        proxy_mpjpe_str = f"{m['proxy_skeleton_mpjpe_m']:.3f}" if m['proxy_skeleton_mpjpe_m'] is not None else "N/A"
        kpt_cnt = f"{m['num_observable_3d_keypoints']}/15"

        print(f"{ep:<4} | {str(m['state_valid']):<6} | {pos_err_str:<12} | {yaw_err_str:<14} | {obs_mpjpe_str:<14} | {proxy_mpjpe_str:<16} | {kpt_cnt:<8}")

    print("=" * 90)
    runner.close()


if __name__ == "__main__":
    main()
