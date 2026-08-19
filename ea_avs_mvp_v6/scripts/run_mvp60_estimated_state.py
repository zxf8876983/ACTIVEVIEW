"""
主实验执行脚本 —— run_mvp60_estimated_state.py
=============================================

功能：
    运行 ACTIVEVIEW v6.0 基于当前 RGB-D 人体状态估计的主动视角选择完整实验。
    支持三支路严谨对比：
        1. EstimatedState-Ours（主方法：纯视觉估计状态驱动）
        2. GTState-Ours（特权基线：Humanoid GT 状态驱动）
        3. Oracle-NBV（离线上界：后验真实渲染得分最高位姿）
        4. Fixed / Random / Nearest 基线策略

核心科研约束：
    - 在线决策时 EstimatedState-Ours 严禁接收任何 GT 参数
    - 在选定目标位姿前严禁渲染候选点的未来观测
"""

import argparse
import copy
import os
import time
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v6.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.estimated_predictive_evaluator import EstimatedPredictiveEvaluator
from ea_avs_v6.predictive_evaluator import PredictiveEvaluator
from ea_avs_v6.true_evaluator import TrueEvaluator
from ea_avs_v6.policies import (
    EstimatedStateOursPolicy,
    GTStateOursPolicy,
    FixedPolicy,
    RandomPolicy,
    NearestPolicy,
)
from ea_avs_v6.oracle_policy import OraclePolicy, compute_oracle_gap
from ea_avs_v6.metrics import compute_state_estimation_metrics, MetricsWriter
from ea_avs_v6.geometry import compute_look_at_yaw, normalize_angle
from ea_avs_v6.visualization import (
    save_rgb_image,
    save_depth_image,
    save_pose_overlay_image,
    save_candidate_debug_json,
)


def run_experiment(config: dict, output_dir: str, num_episodes: int, save_artifacts: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    metrics_writer = MetricsWriter(output_dir)

    print(f"\n=======================================================")
    print(f" EA-AVS MVP v6.0 Experiment Runner")
    print(f" Episodes: {num_episodes} | Output Dir: {output_dir}")
    print(f"=======================================================\n")

    # 1. 初始化仿真器与 Humanoid
    runner = HabitatRunner(config)
    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    # 2. 初始化核心模块
    estimator = HumanStateEstimator(config)
    est_pred_evaluator = EstimatedPredictiveEvaluator(config)
    gt_pred_evaluator = PredictiveEvaluator(config)
    true_evaluator = TrueEvaluator(config)
    candidate_sampler = CandidateSampler(config)

    # 3. 初始化策略
    policy_est_ours = EstimatedStateOursPolicy()
    policy_gt_ours = GTStateOursPolicy()
    policy_fixed = FixedPolicy()
    policy_random = RandomPolicy(seed=config.get("project", {}).get("seed", 42))
    policy_nearest = NearestPolicy()
    policy_oracle = OraclePolicy(min_depth_coverage=config.get("oracle", {}).get("min_depth_coverage", 0.8))

    ep_cfg = config.get("episode", {})
    min_dist = ep_cfg.get("min_robot_human_distance", 1.5)
    max_dist = ep_cfg.get("max_robot_human_distance", 3.5)

    for ep_idx in range(num_episodes):
        ep_t0 = time.time()
        print(f"\n--- [Episode {ep_idx + 1}/{num_episodes}] ---")

        # -------------------------------------------------------------
        # A. 场景与人体放置
        # -------------------------------------------------------------
        human_pos = runner.sample_navigable_point()
        human_yaw = float(np.random.uniform(-np.pi, np.pi)) if ep_cfg.get("randomize_human_yaw", True) else 0.0
        humanoid.set_base_pose(human_pos, human_yaw)
        humanoid.set_pose("standing")

        # 读取 GT 人体骨架与 Object IDs
        gt_skeleton_data = get_humanoid_gt_skeleton(humanoid, strict=False)
        gt_skeleton = gt_skeleton_data["skeleton"]
        keypoint_meta = gt_skeleton_data["keypoint_meta"]
        humanoid_object_ids = humanoid.get_humanoid_object_ids()

        # 采样机器人初始位置
        max_tries = ep_cfg.get("max_sampling_tries", 100)
        robot_pos = None
        for _ in range(max_tries):
            pt = runner.sample_navigable_point()
            if not runner.is_navigable(pt):
                continue
            d = float(np.linalg.norm(pt - human_pos))
            if not (min_dist <= d <= max_dist):
                continue
            if runner.geodesic_distance(pt, human_pos) == float("inf"):
                continue
            robot_pos = pt
            break

        if robot_pos is None:
            robot_pos = runner.snap_point(human_pos + np.array([min_dist, 0.0, 0.0], dtype=np.float32))

        robot_yaw = compute_look_at_yaw(robot_pos, human_pos)

        # -------------------------------------------------------------
        # B. 当前视角观测与纯视觉人体状态估计
        # -------------------------------------------------------------
        obs_current = runner.render_at(robot_pos, robot_yaw)
        camera_state = runner.get_camera_state(robot_pos, robot_yaw)

        t_est_0 = time.time()
        estimated_state = estimator.estimate(obs_current["rgb"], obs_current["depth"], camera_state)
        t_est_ms = (time.time() - t_est_0) * 1000.0

        est_metrics = compute_state_estimation_metrics(estimated_state, human_pos, human_yaw, gt_skeleton)
        print(f"State Estimation: valid={estimated_state.valid}, pos_err={est_metrics['pos_error_m']}, yaw_err={est_metrics['yaw_error_deg']}, obs_3d={est_metrics['num_observable_3d_keypoints']}/15 ({t_est_ms:.1f}ms)")

        # -------------------------------------------------------------
        # C. 候选视角采样
        # -------------------------------------------------------------
        # 1) GT 采样空间（用于 GT 支路、Oracle 与客观候选空间）
        candidates_gt = candidate_sampler.sample(human_pos, robot_pos, runner)

        # 2) 估计采样空间（用于 EstimatedState 支路）
        if estimated_state.valid and estimated_state.human_position_world is not None:
            candidates_est = candidate_sampler.sample(estimated_state.human_position_world, robot_pos, runner)
        else:
            candidates_est = []

        current_view_gt = CandidateView(
            candidate_id=0, position=robot_pos.copy(), yaw=robot_yaw,
            geodesic_distance=0.0, euclidean_distance_to_human=float(np.linalg.norm(robot_pos - human_pos)),
            is_valid=True,
        )
        current_view_est = CandidateView(
            candidate_id=0, position=robot_pos.copy(), yaw=robot_yaw,
            geodesic_distance=0.0,
            euclidean_distance_to_human=float(np.linalg.norm(robot_pos - estimated_state.human_position_world)) if estimated_state.valid else float("inf"),
            is_valid=True,
        )

        # -------------------------------------------------------------
        # D. 移动前预测评分 (Zero Future Observation Leakage!)
        # -------------------------------------------------------------
        # 1) Estimated-State 预测评估
        current_view_est.pred_score = est_pred_evaluator.score_view_pred(
            runner, current_view_est.position, current_view_est.yaw, robot_pos, estimated_state, 0.0
        )
        for c in candidates_est:
            if c.is_valid:
                c.pred_score = est_pred_evaluator.score_view_pred(
                    runner, c.position, c.yaw, robot_pos, estimated_state, c.geodesic_distance
                )

        # 2) GT-State 预测评估
        current_view_gt.pred_score = gt_pred_evaluator.score_view_pred(
            runner, current_view_gt.position, current_view_gt.yaw, robot_pos,
            human_pos, human_yaw, "standing", gt_skeleton, 0.0,
            humanoid_object_ids=humanoid_object_ids, keypoint_meta=keypoint_meta,
        )
        for c in candidates_gt:
            if c.is_valid:
                c.pred_score = gt_pred_evaluator.score_view_pred(
                    runner, c.position, c.yaw, robot_pos,
                    human_pos, human_yaw, "standing", gt_skeleton, c.geodesic_distance,
                    humanoid_object_ids=humanoid_object_ids, keypoint_meta=keypoint_meta,
                )

        # -------------------------------------------------------------
        # E. 策略决策
        # -------------------------------------------------------------
        selected_views = {}
        # 1) EstimatedState-Ours (主方法)
        selected_views["EstimatedState-Ours"] = policy_est_ours.select(
            current_view_est, candidates_est, is_state_valid=estimated_state.valid
        )
        # 2) GTState-Ours (特权基线)
        selected_views["GTState-Ours"] = policy_gt_ours.select(
            current_view_gt, candidates_gt
        )
        # 3) Baselines (Fixed, Random, Nearest)
        selected_views["Fixed"] = policy_fixed.select(current_view_gt, candidates_gt)
        selected_views["Random"] = policy_random.select(current_view_gt, candidates_gt)
        selected_views["Nearest"] = policy_nearest.select(current_view_gt, candidates_gt)

        # -------------------------------------------------------------
        # F. 到达后真实渲染与指标计算 (Evaluation Phase)
        # -------------------------------------------------------------
        policy_results = {}
        rendered_observations = {}

        # 评估各策略选定位姿的真实观测得分
        for pol_name, sel_view in selected_views.items():
            obs_sel = runner.render_at(sel_view.position, sel_view.yaw)
            rendered_observations[pol_name] = obs_sel
            true_score = true_evaluator.score_view_true(
                runner=runner,
                obs=obs_sel,
                view_pos=sel_view.position,
                view_yaw=sel_view.yaw,
                human_base_pos=human_pos,
                human_yaw=human_yaw,
                pose_type="standing",
                human_skeleton=gt_skeleton,
                humanoid_object_ids=humanoid_object_ids,
                keypoint_meta=keypoint_meta,
            )
            policy_results[pol_name] = {
                "selected_candidate_id": sel_view.candidate_id,
                "position": sel_view.position.tolist(),
                "yaw": float(sel_view.yaw),
                "is_current": bool(sel_view is current_view_est or sel_view is current_view_gt or np.allclose(sel_view.position, robot_pos)),
                "pred_score": sel_view.pred_score,
                "true_score": true_score,
            }

        # 计算 Oracle-NBV 离线上界 (渲染所有 GT 候选点)
        current_view_gt.true_score = true_evaluator.score_view_true(
            runner, obs_current, current_view_gt.position, current_view_gt.yaw,
            human_pos, human_yaw, "standing", gt_skeleton,
            humanoid_object_ids=humanoid_object_ids, keypoint_meta=keypoint_meta,
        )
        for c in candidates_gt:
            if c.is_valid:
                obs_c = runner.render_at(c.position, c.yaw)
                c.true_score = true_evaluator.score_view_true(
                    runner, obs_c, c.position, c.yaw,
                    human_pos, human_yaw, "standing", gt_skeleton,
                    humanoid_object_ids=humanoid_object_ids, keypoint_meta=keypoint_meta,
                )

        oracle_view, oracle_detail = policy_oracle.select(current_view_gt, candidates_gt)
        if oracle_view:
            policy_results["Oracle"] = {
                "selected_candidate_id": oracle_view.candidate_id,
                "position": oracle_view.position.tolist(),
                "yaw": float(oracle_view.yaw),
                "is_current": bool(oracle_view is current_view_gt),
                "true_score": oracle_view.true_score,
                "detail": oracle_detail,
            }
        else:
            policy_results["Oracle"] = {"true_score": {"Q_true": None}, "detail": oracle_detail}

        # 计算综合比较指标
        est_view_res = selected_views["EstimatedState-Ours"]
        gt_view_res = selected_views["GTState-Ours"]

        gap_est = compute_oracle_gap(oracle_view, est_view_res, current_view_gt)
        gap_gt = compute_oracle_gap(oracle_view, gt_view_res, current_view_gt)

        q_true_est = policy_results["EstimatedState-Ours"]["true_score"]["Q_true"]
        q_true_gt = policy_results["GTState-Ours"]["true_score"]["Q_true"]
        estimation_gap = (q_true_gt - q_true_est) if (q_true_est is not None and q_true_gt is not None) else None

        agreement = (
            np.linalg.norm(np.array(policy_results["EstimatedState-Ours"]["position"]) - np.array(policy_results["GTState-Ours"]["position"])) < 0.3
            and abs(normalize_angle(policy_results["EstimatedState-Ours"]["yaw"] - policy_results["GTState-Ours"]["yaw"])) < np.deg2rad(30.0)
        )

        comparative_metrics = {
            "estimated_gt_agreement": bool(agreement),
            "oracle_gap_est": gap_est.get("oracle_gap"),
            "oracle_gap_gt": gap_gt.get("oracle_gap"),
            "estimation_gap": float(estimation_gap) if estimation_gap is not None else None,
        }

        print(f"Results Summary: Q_true(Est)={q_true_est:.3f}, Q_true(GT)={q_true_gt:.3f}, Agreement={agreement}, OracleGap(Est)={gap_est.get('oracle_gap')}")

        # -------------------------------------------------------------
        # G. 数据记录与可选可视化落盘
        # -------------------------------------------------------------
        ep_record = {
            "episode_index": ep_idx,
            "human_pos": human_pos.tolist(),
            "human_yaw": float(human_yaw),
            "robot_start_pos": robot_pos.tolist(),
            "robot_start_yaw": float(robot_yaw),
            "estimated_state": estimated_state.to_dict(),
            "estimation_metrics": est_metrics,
            "policy_results": policy_results,
            "comparative_metrics": comparative_metrics,
        }
        metrics_writer.log_episode(ep_record)

        if save_artifacts and ep_idx < 5:  # 保存前 5 个 episode 的图像
            ep_dir = os.path.join(output_dir, f"episode_{ep_idx:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            save_rgb_image(obs_current["rgb"], os.path.join(ep_dir, "current_rgb.png"))
            save_depth_image(obs_current["depth"], os.path.join(ep_dir, "current_depth.png"))

            if estimator.pose_backend.infer(obs_current["rgb"]):
                det0 = estimator.pose_backend.infer(obs_current["rgb"])[0]
                save_pose_overlay_image(obs_current["rgb"], det0.keypoints, os.path.join(ep_dir, "current_pose_overlay.png"), det0.bbox_xyxy)

            for pol, obs_p in rendered_observations.items():
                if obs_p and obs_p.get("rgb") is not None:
                    save_rgb_image(obs_p["rgb"], os.path.join(ep_dir, f"selected_rgb_{pol}.png"))

            save_candidate_debug_json(ep_record, os.path.join(ep_dir, "debug.json"))

    metrics_writer.close()
    runner.close()
    print(f"\n[RunExperiment] COMPLETED: Output saved to {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Run EA-AVS v6.0 Estimated-State Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mvp60_experiment",
        help="Directory to save experimental metrics and outputs",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes to run",
    )
    parser.add_argument(
        "--no_save_images",
        action="store_true",
        help="Disable saving observation images to save disk space",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    num_episodes = args.episodes or config.get("episode", {}).get("num_episodes", 20)
    save_artifacts = not args.no_save_images

    run_experiment(
        config=config,
        output_dir=args.output_dir,
        num_episodes=num_episodes,
        save_artifacts=save_artifacts,
    )


if __name__ == "__main__":
    main()
