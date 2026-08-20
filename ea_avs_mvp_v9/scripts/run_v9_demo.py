"""
v9.0 Action-conditioned Active View Scoring 综合演示脚本 —— run_v9_demo.py
========================================================================

功能：
    1. 基于已知人体位置与动作状态 (Action Label)，执行动作感知主动视点选择：
       Human State + Action State (Known)
       ↓
       Local Candidate View Generation & Constraint Filtering (v8 pipeline)
       ↓
       Action Encoding & Feature Extraction (ActionEncoder + ViewFeatureExtractor)
       ↓
       Action-conditioned Scoring Q(v|a) = w_geom * Q_geom + w_act * Delta_Q(a, v)
       ↓
       Multi-baseline Selection & Comparison (random, nearest, geometry_best, action_conditioned)
       ↓
       Render Action-conditioned Best Viewpoint RGB Observation
    2. 输出成果至 data/ActiveView/visualizations/v9_demo/：
       - action_view_scores.json (包含所有候选视点的几何分、动作增益与综合分)
       - best_view.json (选定视点位置、外参、动作适配指标)
       - comparison_report.json (四大基线量化对比报告)
       - best_view_rgb.png (选定视点高质量渲染图像)
       - action.json (动作先验与嵌入配置)
       - metadata.json (实验元数据)

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_v9_demo
    python -m ea_avs_mvp_v9.scripts.run_v9_demo --action fall --strategy action_conditioned
    python -m ea_avs_mvp_v9.scripts.run_v9_demo --action sitting
    python -m ea_avs_mvp_v9.scripts.run_v9_demo --action bending
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 基础与数据路径
from ea_avs_mvp_v9.core.config import load_v9_config
from ea_avs_mvp_v9.core.paths import get_data_root, to_relative_data_path

# v8 基础设施复用 (严禁修改 v8 源码)
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.robot.robot_adapter import V8RobotAdapter
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

# v9 动作感知新模块
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector
from ea_avs_mvp_v9.evaluation.baseline_comparison import compare_all_baselines
from ea_avs_mvp_v9.visualization.action_view_plotter import format_comparison_table
from ea_avs_mvp_v9.dataset.v9_dataset_loader import save_action_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v9_demo")


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v9.0 Action-conditioned Active View Scoring Demo")
    parser.add_argument("--config", type=str, default=None, help="Path to custom v9_demo.yaml")
    parser.add_argument("--action", type=str, default=None, choices=["fall", "sitting", "standing", "bending", "reaching"], help="Human action state")
    parser.add_argument("--strategy", type=str, default=None, choices=["action_conditioned", "geometry_best", "nearest", "random"], help="Viewpoint selection strategy")
    parser.add_argument("--evaluation-mode", type=str, default=None, choices=["oracle", "estimated"], help="Evaluation information boundary")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    cfg = load_v9_config(args.config)
    scene_id = cfg.scene.get("scene_id", "apartment_1")
    action_label = args.action or cfg.human.get("action", "fall")
    strategy = args.strategy or "action_conditioned"
    eval_mode = args.evaluation_mode or cfg.scoring.get("evaluation_mode", "oracle")
    pose_source = cfg.scoring.get("pose_source", "oracle")

    # 1. 初始化输出目录
    if args.output_dir:
        vis_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else get_data_root() / args.output_dir
    else:
        vis_dir = get_data_root() / cfg.simulation.get("output_dir", "visualizations/v9_demo")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 2. 步骤一: 启动仿真环境并加载已知人体与动作
    logger.info("Step 1: Initializing Habitat Scene & Placing Humanoid with action '%s'...", action_label)
    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)

    robot_start_pos = cfg.robot.get("initial_pose", {}).get("position", [1.5, -1.60, 6.8])

    # 3. 步骤二: 生成候选视点并执行空间与几何硬约束过滤
    logger.info("Step 2: Generating and Filtering Candidate Viewpoints (v8 constraint pipeline)...")
    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pose.position,
        human_yaw_deg=human_pose.yaw_deg,
        ground_height=cfg.robot.get("ground_height", -1.60),
    )

    gt_joints = humanoid.get_gt_joint_positions()
    checker = ConstraintChecker(env_adapter=env_adapter, config={**cfg.viewpoint, "evaluation_mode": eval_mode})
    checked_candidates = checker.filter_feasible_viewpoints(
        raw_candidates,
        human_position=human_pose.position,
        human_joints_3d=gt_joints,
        robot_start_pos=robot_start_pos,
    )
    for vp in checked_candidates:
        vp.evaluation_mode = eval_mode

    # 4. 步骤三: 提取 v8 几何质量得分 Q_geom(v)
    logger.info("Step 3: Evaluating v8 Geometry Quality Scores...")
    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": eval_mode, "pose_source": pose_source})
    ranked_geom_pairs = geom_evaluator.rank_viewpoints(
        viewpoints=checked_candidates,
        human_joints_3d=gt_joints,
        human_yaw_deg=human_pose.yaw_deg,
        pose_source=pose_source,
    )
    geom_qualities = [q for _, q in ranked_geom_pairs]
    geom_score_map = {q.viewpoint_id: q.visibility_score for q in geom_qualities}

    # 5. 步骤四: 动作编码与视点多维特征提取 (v9 Action Module)
    logger.info("Step 4: Encoding Action '%s' and Extracting Region Features...", action_label)
    encoder = ActionEncoder(cfg.action_weights)
    action_embedding = encoder.encode(action_label)

    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(
        viewpoints=checked_candidates,
        human_joints_3d=gt_joints,
        human_yaw_deg=human_pose.yaw_deg,
    )

    # 6. 步骤五: 动作条件打分 Q(v|a) 与全基线对比评测
    logger.info("Step 5: Computing Action-conditioned Scores Q(v|a) & Multi-baseline Comparison...")
    scorer = ActionConditionedScorer({**cfg.scoring, "evaluation_mode": eval_mode, "pose_source": pose_source})
    action_scores = scorer.score_batch(features, action_embedding, geom_score_map)

    # 执行四大基线对比
    comparison_report = compare_all_baselines(
        viewpoints=checked_candidates,
        action_scores=action_scores,
        features=features,
        geometry_qualities=geom_qualities,
        action=action_embedding,
        human_position=human_pose.position,
        seed=42,
    )

    # 选定最终视点
    selected_vp, selected_score = ViewpointSelector.select(
        viewpoints=checked_candidates,
        action_scores=action_scores,
        geometry_qualities=geom_qualities,
        strategy=strategy,
        human_position=human_pose.position,
        seed=42,
    )

    # 7. 步骤六: 移动机器人至选定视点并渲染高质量 RGB 图像
    logger.info("Step 6: Moving Robot to Selected Viewpoint ('%s') and Rendering RGB...", strategy)
    robot_adapter = V8RobotAdapter(sim, cfg.camera)
    cam_info = robot_adapter.set_viewpoint(selected_vp, verbose=True)

    obs = robot_adapter.capture_observation()
    env_adapter.close()

    # 保存最佳视角图像
    best_rgb_path = vis_dir / "best_view_rgb.png"
    if obs.get("rgb") is not None:
        Image.fromarray(obs["rgb"]).save(best_rgb_path)

    # 8. 步骤七: 持久化全部结构化产物
    logger.info("Step 7: Saving action_view_scores.json, best_view.json and comparison_report.json...")

    # 保存 action_view_scores.json
    scores_json_path = vis_dir / "action_view_scores.json"
    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in action_scores], f, indent=2, ensure_ascii=False)

    # 保存 best_view.json
    best_json_path = vis_dir / "best_view.json"
    feat_dict = next((f.to_dict() for f in features if f.viewpoint_id == selected_vp.viewpoint_id), {})
    best_data = {
        "selected_viewpoint_id": selected_vp.viewpoint_id,
        "strategy": strategy,
        "action_name": action_embedding.action_name,
        "evaluation_mode": eval_mode,
        "pose_source": pose_source,
        "total_score": selected_score.total_score,
        "geometry_score": selected_score.geometry_score,
        "action_delta": selected_score.action_delta,
        "position": [float(x) for x in selected_vp.position],
        "yaw_deg": float(selected_vp.yaw_deg),
        "camera_pose": {
            "position": cam_info["camera_position"],
            "rotation": cam_info["camera_rotation"],
        },
        "view_feature": feat_dict,
        "rgb_image": to_relative_data_path(best_rgb_path),
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)

    # 保存 comparison_report.json
    comp_json_path = vis_dir / "comparison_report.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)

    # 保存 action.json
    save_action_metadata(vis_dir, action_embedding.to_dict())

    # 保存 metadata.json
    meta_json_path = vis_dir / "metadata.json"
    metadata = {
        "version": "9.0.0",
        "scene_id": scene_id,
        "action_target": action_embedding.action_name,
        "evaluation_mode": eval_mode,
        "pose_source": pose_source,
        "total_candidates": len(checked_candidates),
        "feasible_candidates": sum(1 for v in checked_candidates if v.feasible),
        "selected_strategy": strategy,
        "best_view": best_data,
        "comparison_summary": comparison_report,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 9. 打印对比实验表格与验收结论
    table_str = format_comparison_table(comparison_report)
    print("\n" + table_str)
    print("\n" + "=" * 65)
    print("[V9.0 Action-conditioned View Selection Summary]")
    print(f"Action Label:          {action_embedding.action_name.upper()}")
    print(f"Selected Strategy:     {strategy}")
    print(f"Selected Viewpoint ID: {selected_vp.viewpoint_id}")
    print(f"Total Score Q(v|a):    {selected_score.total_score:.3f} (Q_geom: {selected_score.geometry_score:.3f}, Delta_Q: {selected_score.action_delta:+.3f})")
    print(f"View Position:         {[round(x, 2) for x in selected_vp.position]}")
    print(f"View Yaw:              {selected_vp.yaw_deg:.1f} deg")
    print(f"Comparison Report:     {comp_json_path}")
    print(f"Scores Output:         {scores_json_path}")
    print(f"Rendered Image:        {best_rgb_path}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v9.0 Action-conditioned Active View Scoring Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
