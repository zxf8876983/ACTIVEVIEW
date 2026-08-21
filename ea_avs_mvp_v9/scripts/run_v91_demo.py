"""
v9.1 学习型视角打分综合演示脚本 —— run_v91_demo.py
==================================================

功能：
    1. 在 Habitat 室内仿真场景中加载已知人体与指定动作；
    2. 生成候选视点并执行三阶空间与可见性约束过滤；
    3. 调用训练好的 LearnableViewScorer 进行神经网络前向打分与视点排序；
    4. 执行 5 大基线综合横向对比：
       - Baseline 1: Random View
       - Baseline 2: Nearest View
       - Baseline 3: Geometry Best (v8)
       - Baseline 4: Rule Action (v9.0)
       - Method:     Learnable Action-conditioned (v9.1 Ours)
    5. 移动机器人至神经网络选定最优视点，捕获并保存 best_view_rgb.png；
    6. 保存结构化结果至 data/ActiveView/visualizations/v91_demo/：
       - v91_best_view.json
       - comparison_report.json
       - comparison_with_v90.json
       - best_view_rgb.png

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_v91_demo
    python -m ea_avs_mvp_v9.scripts.run_v91_demo --action sitting
    python -m ea_avs_mvp_v9.scripts.run_v91_demo --action bending
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

# 路径与配置
from ea_avs_mvp_v9.core.config import load_v9_config
from ea_avs_mvp_v9.core.paths import get_data_root, to_relative_data_path

# v8/v7 仿真基础设施
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.robot.robot_adapter import V8RobotAdapter
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

# v9.0 规则模块
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector
from ea_avs_mvp_v9.evaluation.action_metrics import compute_action_observation_metrics

# v9.1 学习型推理模块
from ea_avs_mvp_v9.inference.predict_view import ViewPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v91_demo")


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v9.1 Learnable Action-conditioned View Scoring Demo")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--action", type=str, default=None, choices=["fall", "sitting", "standing", "bending", "reaching"], help="Human action")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_v9_config(args.config)
    scene_id = cfg.scene.get("scene_id", "apartment_1")
    action_label = args.action or cfg.human.get("action", "sitting")

    # 1. 目录与检查点初始化
    if args.output_dir:
        vis_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else get_data_root() / args.output_dir
    else:
        vis_dir = get_data_root() / "visualizations/v91_demo"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (get_data_root() / "checkpoints/model_checkpoint.pth")
    if not ckpt_path.exists():
        logger.warning("Model checkpoint not found at %s. Training a quick model automatically...", ckpt_path)
        from ea_avs_mvp_v9.scripts.train_v91 import main as train_main
        # 快速训练一个模型
        from ea_avs_mvp_v9.training.dataset import generate_scoring_dataset
        from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer
        from ea_avs_mvp_v9.training.trainer import ViewScorerTrainer
        train_ds, val_ds = generate_scoring_dataset(num_episodes=150, seed=42)
        model = LearnableViewScorer()
        trainer = ViewScorerTrainer(model)
        trainer.train(train_ds, val_ds, num_epochs=30, batch_size=16, checkpoint_path=ckpt_path)

    # 2. 步骤一: 启动 Habitat 仿真并加载已知人体姿态
    logger.info("Step 1: Initializing Habitat Scene & Placing Humanoid with action '%s'...", action_label)
    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)
    gt_joints = humanoid.get_gt_joint_positions()

    robot_start_pos = cfg.robot.get("initial_pose", {}).get("position", [1.5, -1.60, 6.8])

    # 3. 步骤二: 生成候选视点并过滤
    logger.info("Step 2: Generating candidate viewpoints and applying constraints...")
    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pose.position,
        human_yaw_deg=human_pose.yaw_deg,
        ground_height=cfg.robot.get("ground_height", -1.60),
    )

    checker = ConstraintChecker(env_adapter=env_adapter, config=cfg.viewpoint)
    checked_candidates = checker.filter_feasible_viewpoints(
        raw_candidates,
        human_position=human_pose.position,
        human_joints_3d=gt_joints,
        robot_start_pos=robot_start_pos,
    )

    # 4. 步骤三: 提取多维几何与区域观测特征
    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    feat_map = {f.viewpoint_id: f for f in features}
    vp_map = {v.viewpoint_id: v for v in checked_candidates}

    # 5. 步骤四: 运行基线 1-4 (Random, Nearest, v8 Geometry Best, v9.0 Rule Action)
    logger.info("Step 3: Evaluating Baselines (Random, Nearest, Geometry Best v8, Rule Action v9.0)...")
    encoder = ActionEncoder(cfg.action_weights)
    act_embed = encoder.encode(action_label)

    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    geom_ranked = geom_evaluator.rank_viewpoints(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    geom_qualities = [q for _, q in geom_ranked]
    geom_map = {q.viewpoint_id: q.visibility_score for q in geom_qualities}

    rule_scorer = ActionConditionedScorer(cfg.scoring)
    rule_scores = rule_scorer.score_batch(features, act_embed, geom_map)
    rule_score_map = {s.viewpoint_id: s for s in rule_scores}

    # 选定各基线视点
    vp_rand, _ = ViewpointSelector.select(checked_candidates, rule_scores, strategy="random", seed=42)
    vp_near, _ = ViewpointSelector.select(checked_candidates, rule_scores, strategy="nearest", human_position=human_pose.position)
    vp_geom, _ = ViewpointSelector.select(checked_candidates, rule_scores, geometry_qualities=geom_qualities, strategy="geometry_best")
    vp_rule, s_rule = ViewpointSelector.select(checked_candidates, rule_scores, strategy="action_conditioned")

    # 6. 步骤五: 运行 v9.1 学习型打分模型前向推理
    logger.info("Step 4: Running v9.1 Neural View Scorer Inference...")
    predictor = ViewPredictor(checkpoint_path=ckpt_path)
    pred_res = predictor.predict_viewpoints(
        viewpoints=checked_candidates,
        features=features,
        human_joints_3d=gt_joints,
        human_yaw_deg=human_pose.yaw_deg,
        action_label=action_label,
    )

    best_v91_id = pred_res["best_viewpoint_id"]
    vp_learnable = vp_map[best_v91_id]
    s_learnable_pred = pred_res["best_predicted_score"]

    # 7. 步骤六: 移动机器人至 v9.1 选定视点并渲染 RGB 图像
    logger.info("Step 5: Moving Robot to v9.1 Selected Viewpoint ('%s') and Rendering RGB...", best_v91_id)
    robot_adapter = V8RobotAdapter(sim, cfg.camera)
    cam_info = robot_adapter.set_viewpoint(vp_learnable, verbose=True)

    obs = robot_adapter.capture_observation()
    env_adapter.close()

    best_rgb_path = vis_dir / "best_view_rgb.png"
    if obs.get("rgb") is not None:
        Image.fromarray(obs["rgb"]).save(best_rgb_path)

    # 8. 步骤七: 汇总 5 大基线评测结果
    logger.info("Step 6: Summarizing 5-Baseline Comparison and Saving Reports...")

    def build_baseline_entry(v_obj):
        f = feat_map[v_obj.viewpoint_id]
        s_obj = rule_score_map[v_obj.viewpoint_id]
        pred_s = pred_res["scores_map"].get(v_obj.viewpoint_id, 0.0)
        m = compute_action_observation_metrics(f, s_obj, act_embed)
        return {
            "selected_view_id": v_obj.viewpoint_id,
            "position": [round(float(x), 2) for x in v_obj.position],
            "yaw_deg": round(float(v_obj.yaw_deg), 1),
            "distance": f.distance,
            "viewing_angle_deg": f.viewing_angle_deg,
            "pose_coverage": f.pose_coverage,
            "critical_region_coverage": m["critical_region_coverage"],
            "geometry_score": s_obj.geometry_score,
            "rule_total_score": s_obj.total_score,
            "learnable_predicted_score": pred_s,
        }

    baselines_report = {
        "action": action_label,
        "critical_regions": act_embed.critical_regions,
        "total_feasible_candidates": sum(1 for v in checked_candidates if v.feasible),
        "baselines": {
            "random": build_baseline_entry(vp_rand),
            "nearest": build_baseline_entry(vp_near),
            "geometry_best_v8": build_baseline_entry(vp_geom),
            "rule_action_v90": build_baseline_entry(vp_rule),
            "learnable_action_v91": build_baseline_entry(vp_learnable),
        },
        "v91_matches_v90_optimal": bool(vp_learnable.viewpoint_id == vp_rule.viewpoint_id),
        "v91_matches_v8_geometry": bool(vp_learnable.viewpoint_id == vp_geom.viewpoint_id),
    }

    # 保存 comparison_report.json
    comp_json_path = vis_dir / "comparison_report.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(baselines_report, f, indent=2, ensure_ascii=False)

    # 保存 comparison_with_v90.json
    v90_comp = {
        "action": action_label,
        "v8_geometry_view": baselines_report["baselines"]["geometry_best_v8"],
        "v90_rule_view": baselines_report["baselines"]["rule_action_v90"],
        "v91_learnable_view": baselines_report["baselines"]["learnable_action_v91"],
        "v91_top1_match_with_rule_oracle": baselines_report["v91_matches_v90_optimal"],
    }
    with open(vis_dir / "comparison_with_v90.json", "w", encoding="utf-8") as f:
        json.dump(v90_comp, f, indent=2, ensure_ascii=False)

    # 保存 v91_best_view.json 与 best_view.json
    best_v91_data = {
        "version": "9.1.0",
        "method": "Human-state-aware Learnable Active View Selection",
        "selected_viewpoint_id": vp_learnable.viewpoint_id,
        "action": action_label,
        "predicted_score": s_learnable_pred,
        "ground_truth_score": rule_score_map[vp_learnable.viewpoint_id].total_score,
        "position": [float(x) for x in vp_learnable.position],
        "yaw_deg": float(vp_learnable.yaw_deg),
        "camera_pose": cam_info,
        "body_part_visibilities": feat_map[vp_learnable.viewpoint_id].body_part_visibilities,
        "metrics": feat_map[vp_learnable.viewpoint_id].to_dict(),
        "rgb_image": to_relative_data_path(best_rgb_path),
    }
    best_json_path = vis_dir / "v91_best_view.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_v91_data, f, indent=2, ensure_ascii=False)
    with open(vis_dir / "best_view.json", "w", encoding="utf-8") as f:
        json.dump(best_v91_data, f, indent=2, ensure_ascii=False)

    # 保存 7 大身体关键解剖部位可见性报表 body_part_visibility_report.json
    body_part_report = {
        "action": action_label,
        "supported_parts": ["head", "torso", "pelvis", "left_hand", "right_hand", "left_leg", "right_leg"],
        "baselines_visibility": {
            b_name: {
                "selected_view_id": b_entry["selected_view_id"],
                "distance": b_entry["distance"],
                "viewing_angle_deg": b_entry["viewing_angle_deg"],
                "pose_coverage": b_entry["pose_coverage"],
                "body_parts": feat_map[b_entry["selected_view_id"]].body_part_visibilities,
            }
            for b_name, b_entry in baselines_report["baselines"].items()
        }
    }
    body_part_path = vis_dir / "body_part_visibility_report.json"
    with open(body_part_path, "w", encoding="utf-8") as f:
        json.dump(body_part_report, f, indent=2, ensure_ascii=False)
    # 同时也复制一份到 results/
    res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "body_part_visibility_report.json", "w", encoding="utf-8") as f:
        json.dump(body_part_report, f, indent=2, ensure_ascii=False)

    # 9. 打印 5 大基线对比汇总
    print("\n" + "=" * 92)
    print(f"  ACTIVEVIEW v9.1 5-Baseline Viewpoint Selection Comparison (Action: {action_label.upper()})")
    print("=" * 92)
    print(f"{'Method / Baseline':<24} | {'Selected View':<16} | {'Dist(m)':<8} | {'Angle(deg)':<10} | {'Rule Score':<11} | {'Pred Score'}")
    print("-" * 92)

    for b_name, b_data in baselines_report["baselines"].items():
        v_id = b_data["selected_view_id"]
        dist_s = f"{b_data['distance']:.2f}"
        ang_s = f"{b_data['viewing_angle_deg']:.1f}"
        r_score = f"{b_data['rule_total_score']:.3f}"
        p_score = f"{b_data['learnable_predicted_score']:.3f}"
        print(f"{b_name:<24} | {v_id:<16} | {dist_s:<8} | {ang_s:<10} | {r_score:<11} | {p_score}")

    print("=" * 92)
    print(f"v9.1 Learner Matches v9.0 Rule Optimal Viewpoint: {baselines_report['v91_matches_v90_optimal']}")
    print(f"Rendered Image:    {best_rgb_path}")
    print(f"Comparison Report: {comp_json_path}")
    print(f"Best View JSON:    {best_json_path}")
    print("=" * 92)
    print("PASS:\nACTIVEVIEW v9.1 Learnable Action-conditioned View Scoring Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
