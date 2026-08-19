"""
当前视角 RGB-D 姿态检测调试脚本 —— debug_current_rgbd_pose.py
==========================================================

功能：
    在仿真环境中放置 Humanoid，渲染当前视角 RGB-D，运行 2D 姿态后端检测，
    生成带骨架与 BBox 叠加的可视化图像，并输出关键点置信度报告。
"""

import argparse
import os
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.visualization import save_rgb_image, save_depth_image, save_pose_overlay_image


def main():
    parser = argparse.ArgumentParser(description="Debug 2D Pose & RGB-D Observation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug_pose",
        help="Output directory for debug images",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[DebugPose] Initializing HabitatRunner and HumanoidManager...")
    runner = HabitatRunner(config)
    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    # 采样人体与机器人位置
    human_pos = runner.sample_navigable_point()
    human_yaw = 0.0
    humanoid.set_base_pose(human_pos, human_yaw)
    humanoid.set_pose("standing")

    # 放置机器人在人体前方约 2.0m
    robot_pos = human_pos + np.array([0.0, 0.0, 2.0], dtype=np.float32)
    robot_pos = runner.snap_point(robot_pos)
    robot_yaw = np.pi  # 看向人体 (+Z 反方向)

    print(f"[DebugPose] Human pos: {human_pos}, Robot pos: {robot_pos}")

    # 渲染当前视角
    obs = runner.render_at(robot_pos, robot_yaw)
    rgb = obs["rgb"]
    depth = obs["depth"]

    save_rgb_image(rgb, os.path.join(args.output_dir, "current_rgb.png"))
    save_depth_image(depth, os.path.join(args.output_dir, "current_depth.png"))

    camera_state = runner.get_camera_state(robot_pos, robot_yaw)

    # 运行估计器
    estimator = HumanStateEstimator(config)
    state = estimator.estimate(rgb, depth, camera_state)

    # 提取 2D 骨架并绘制叠加图
    if estimator.pose_backend.infer(rgb):
        top_det = estimator.pose_backend.infer(rgb)[0]
        overlay_path = os.path.join(args.output_dir, "pose_overlay.png")
        save_pose_overlay_image(rgb, top_det.keypoints, overlay_path, top_det.bbox_xyxy)
        print(f"[DebugPose] Saved pose overlay image to: {overlay_path}")

    print("\n" + "=" * 60)
    print("Estimated Human State Summary:")
    print("=" * 60)
    print(f"Valid: {state.valid} (Failure reason: {state.failure_reason})")
    print(f"Pose Detection Score: {state.pose_detection_score:.3f}")
    print(f"2D Visible Keypoints: {len(state.visible_2d_keypoints)}/15")
    print(f"3D Observable Keypoints: {len(state.observable_3d_keypoints)}/15")
    print(f"Estimated Position: {state.human_position_world} (Source: {state.human_position_source})")
    print(f"Estimated Yaw: {state.human_yaw} (Source: {state.yaw_source})")
    print(f"Body Scale: {state.body_scale}")
    print("=" * 60)

    runner.close()


if __name__ == "__main__":
    main()
