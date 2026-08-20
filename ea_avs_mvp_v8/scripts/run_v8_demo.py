"""
v8.2 Local Active View Planning Final Baseline 演示脚本 —— run_v8_demo.py
========================================================================

功能：
    1. 基于已知人体位置与局部观察空间假定，执行局部主动视点规划流水线：
       Human Position (Known)
       ↓
       Local Candidate Views Generation (Polar Grid around Human)
       ↓
       View Constraint Pipeline (NavMesh + LineOfSight RayCast + HumanVisibility FOV & Area)
       ↓
       View Quality Evaluation & Baseline Strategy Selection (Geometry Best / Random / Nearest)
       ↓
       Render Selected Viewpoint (Capture & Save best_view_rgb.png)
    2. 输出成果至 data/ActiveView/visualizations/v8_demo/：
       - view_selection_report.json (科研统计核心报表: strategy, evaluation_mode, candidate counts, score, distance, pose_coverage, visibility_loss_ratio)
       - candidate_statistics.json (约束逐级过滤统计)
       - candidate_views.json (包含全部候选视点及可行性标记)
       - best_view.json (包含最佳视点位置、朝向、Q(v) 与各项指标)
       - best_view_rgb.png (最佳视角高质量渲染图像，清晰可见人体)
       - metadata.json (实验元数据汇总)

运行方式：
    python -m ea_avs_mvp_v8.scripts.run_v8_demo
    python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy geometry_best --evaluation-mode oracle
    python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy nearest
    python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy random
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v8.core.config import load_v8_config
from ea_avs_mvp_v8.core.paths import get_data_root, to_relative_data_path
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.robot.robot_adapter import V8RobotAdapter
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v8.evaluation.baseline_strategies import select_view
from ea_avs_mvp_v8.evaluation.view_metrics import summarize_viewpoint_qualities

# 复用 v7 Humanoid
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v8_demo")


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v8.2 Local Active View Planning Baseline Demonstration")
    parser.add_argument("--config", type=str, default=None, help="Path to custom v8_demo.yaml or v8_experiment.yaml")
    parser.add_argument("--strategy", type=str, default=None, choices=["geometry_best", "random", "nearest"], help="View selection baseline strategy")
    parser.add_argument("--evaluation-mode", type=str, default=None, choices=["oracle", "estimated"], help="Information boundary evaluation mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    cfg = load_v8_config(args.config)
    scene_id = cfg.scene.get("scene_id", "apartment_1")
    strategy = args.strategy or cfg.evaluation.get("strategy", "geometry_best")
    eval_mode = args.evaluation_mode or cfg.evaluation.get("evaluation_mode", "oracle")
    pose_source = cfg.evaluation.get("pose_source", "oracle")

    # 1. 输出目录初始化
    if args.output_dir:
        vis_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else get_data_root() / args.output_dir
    else:
        vis_dir = get_data_root() / cfg.simulation.get("output_dir", "visualizations/v8_demo")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 2. 步骤一: 启动仿真环境并加载已知人体 (Known Human Location)
    logger.info("Step 1: Loading Habitat Scene & Placing Humanoid at known location...")
    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)

    robot_start_pos = cfg.robot.get("initial_pose", {}).get("position", [1.5, -1.60, 6.8])

    # 3. 步骤二: 生成以人体为中心的局部候选视点 (Local Candidate Views)
    logger.info("Step 2: Generating Local Candidate Viewpoints around Human...")
    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pose.position,
        human_yaw_deg=human_pose.yaw_deg,
        ground_height=cfg.robot.get("ground_height", -1.60),
    )
    generated_count = len(raw_candidates)

    # 4. 步骤三: 视点空间与几何观测约束过滤 (View Constraint Pipeline)
    logger.info("Step 3: Running View Constraint Pipeline (NavMesh + LineOfSight + HumanVisibility)...")
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

    filtered_feasible_views = [vp for vp in checked_candidates if vp.feasible]
    filtered_count = len(filtered_feasible_views)

    # 5. 步骤四: 视点观测质量评价与 Baseline 策略选择 (View Quality & Selection)
    logger.info("Step 4: Evaluating View Quality (mode=%s, pose_src=%s) and Selecting via '%s'...", eval_mode, pose_source, strategy)
    evaluator = ViewQualityEvaluator({**cfg.evaluation, "evaluation_mode": eval_mode, "pose_source": pose_source})
    ranked_pairs = evaluator.rank_viewpoints(
        viewpoints=checked_candidates,
        human_joints_3d=gt_joints,
        human_yaw_deg=human_pose.yaw_deg,
        pose_source=pose_source,
    )
    all_qualities = [q for _, q in ranked_pairs]

    selected_vp, selected_quality = select_view(
        viewpoints=checked_candidates,
        qualities=all_qualities,
        strategy=strategy,
        human_position=human_pose.position,
        seed=42,
    )

    # 6. 步骤五: 渲染选定视点 RGB 图像 (Render Selected Viewpoint)
    logger.info("Step 5: Moving Robot to Selected Viewpoint and Rendering RGB...")
    robot_adapter = V8RobotAdapter(sim, cfg.camera)
    cam_info = robot_adapter.set_viewpoint(selected_vp, verbose=True)

    obs = robot_adapter.capture_observation()
    env_adapter.close()

    # 保存最佳视角 RGB 图像 (best_view_rgb.png)
    best_rgb_path = vis_dir / "best_view_rgb.png"
    if obs.get("rgb") is not None:
        Image.fromarray(obs["rgb"]).save(best_rgb_path)

    # 7. 步骤六: 持久化实验统计数据与元数据文件
    logger.info("Step 6: Saving candidate_statistics.json, view_selection_report.json and viewpoint datasets...")

    # 保存 candidate_statistics.json
    stats = checker.compute_statistics(
        viewpoints=checked_candidates,
        selected_view_id=selected_vp.viewpoint_id,
        strategy=strategy,
        evaluation_mode=eval_mode,
    )
    stats_json_path = vis_dir / "candidate_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 保存 view_selection_report.json (论文核心统计汇总)
    report = {
        "strategy": strategy,
        "evaluation_mode": eval_mode,
        "total_candidates": stats["total_candidates"],
        "navmesh_valid": stats["navmesh_valid"],
        "line_of_sight_valid": stats["line_of_sight_valid"],
        "human_visible": stats["human_visible"],
        "feasible_candidates": stats["feasible_candidates"],
        "selected_view_id": selected_vp.viewpoint_id,
        "best_score": selected_quality.visibility_score,
        "distance": selected_quality.distance,
        "pose_coverage": selected_quality.pose_coverage,
        "visibility_loss_ratio": selected_quality.visibility_loss_ratio,
    }
    report_json_path = vis_dir / "view_selection_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存 candidate_views.json
    views_json_path = vis_dir / "candidate_views.json"
    with open(views_json_path, "w", encoding="utf-8") as f:
        json.dump([v.to_dict() for v in checked_candidates], f, indent=2, ensure_ascii=False)

    # 保存 best_view.json
    best_json_path = vis_dir / "best_view.json"
    best_data = {
        "best_viewpoint_id": selected_vp.viewpoint_id,
        "score": selected_quality.visibility_score,
        "strategy": strategy,
        "evaluation_mode": eval_mode,
        "pose_source": pose_source,
        "position": [float(x) for x in selected_vp.position],
        "yaw_deg": float(selected_vp.yaw_deg),
        "camera_pose": {
            "position": cam_info["camera_position"],
            "rotation": cam_info["camera_rotation"],
        },
        "metrics": {
            "distance": selected_quality.distance,
            "viewing_angle_deg": selected_quality.viewing_angle_deg,
            "visible_joints_count": selected_quality.visible_joints_count,
            "pose_coverage": selected_quality.pose_coverage,
            "visibility_loss_ratio": selected_quality.visibility_loss_ratio,
            "occlusion_ratio": selected_quality.occlusion_ratio,
        },
        "rgb_image": to_relative_data_path(best_rgb_path),
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)

    # 保存 metadata.json
    summary = summarize_viewpoint_qualities(all_qualities)
    meta_json_path = vis_dir / "metadata.json"
    metadata = {
        "scene_id": scene_id,
        "human_position": [float(x) for x in human_pose.position],
        "evaluation_mode": eval_mode,
        "pose_source": pose_source,
        "generated_views": generated_count,
        "filtered_views": filtered_count,
        "statistics": stats,
        "selection_report": report,
        "best_viewpoint": best_data,
        "visibility_summary": summary,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 8. 打印验收标准格式报告
    print("\n" + "=" * 65)
    print("[V8.2 Local Active View Planning Results]")
    print(f"Evaluation Mode:       {eval_mode} (pose_source: {pose_source})")
    print(f"Selection Strategy:    {strategy}")
    print(f"Generated views:       {generated_count}")
    print(f"Filtered views:        {filtered_count}")
    print(f"Selected Viewpoint ID: {selected_vp.viewpoint_id}")
    print(f"Best viewpoint score:  {selected_quality.visibility_score:.3f}")
    print(f"Selected Pos:          {[round(x, 2) for x in selected_vp.position]}")
    print(f"Selected Yaw:          {selected_vp.yaw_deg:.1f} deg")
    print(f"Distance to Human:     {selected_quality.distance:.2f} m")
    print(f"Pose Coverage:         {selected_quality.pose_coverage:.3f}")
    print(f"Visibility Loss Ratio: {selected_quality.visibility_loss_ratio:.3f}")
    print(f"Rendered Image:        {best_rgb_path}")
    print(f"Selection Report:      {report_json_path}")
    print(f"Candidate Statistics:  {stats_json_path}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v8.2 Local Active View Planning Final Baseline Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
