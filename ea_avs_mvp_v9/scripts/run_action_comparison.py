"""
v9.0 核心实验脚本 —— run_action_comparison.py
=============================================

功能：
    证明科学命题：Q(v|A) != Q(v)，即不同动作状态驱动主动观察视角的自适应迁移。

实验设定：
    1. 同一室内仿真场景 (apartment_1)；
    2. 同一老人初始位置 ([1.5, -1.60, 4.0])；
    3. 同一候选视点池 (32 局部极坐标候选视点，经 NavMesh / LineOfSight / FOV 严格过滤)；
    4. 评测 5 种典型动作：fall, sitting, standing, bending, reaching；
    5. 对比每个动作下 Geometry-best (v8) 与 Action-conditioned (v9) 的选点、坐标、偏角与打分；
    6. 输出结果至 data/ActiveView/results/v9_action_comparison.json。

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_action_comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

from ea_avs_mvp_v9.action.action_encoder import ALL_ACTION_CLASSES, ActionEncoder
from ea_avs_mvp_v9.core.config import load_v9_config
from ea_avs_mvp_v9.core.paths import get_data_root
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector
from ea_avs_mvp_v9.evaluation.baseline_comparison import compare_all_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_action_comparison")


def run_action_comparison_experiment(
    config_path: str = None,
    output_dir: str = None,
    headless_habitat: bool = True,
) -> Dict[str, Any]:
    """执行动作对比核心实验。"""
    cfg = load_v9_config(config_path)
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    # 1. 结果输出目录
    if output_dir:
        res_dir = Path(output_dir) if Path(output_dir).is_absolute() else get_data_root() / output_dir
    else:
        res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 2. 仿真环境与人体加载
    logger.info("Initializing Scene and extracting ground truth human posture...")
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

    # 3. 候选视点采样与约束过滤
    logger.info("Generating candidate viewpoints and applying spatial constraints...")
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
    env_adapter.close()

    # 4. 纯几何打分 Q_geom (v8 baseline)
    logger.info("Evaluating v8 baseline geometry scores...")
    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    ranked_geom_pairs = geom_evaluator.rank_viewpoints(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    geom_qualities = [q for _, q in ranked_geom_pairs]
    geom_map = {q.viewpoint_id: q.visibility_score for q in geom_qualities}

    # v8 选出的几何最优视点
    v8_best_vp, _ = ViewpointSelector.select(
        viewpoints=checked_candidates,
        action_scores=[],
        geometry_qualities=geom_qualities,
        strategy="geometry_best",
    )
    v8_best_q = next(q for q in geom_qualities if q.viewpoint_id == v8_best_vp.viewpoint_id)

    # 5. 特征提取
    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    feat_map = {f.viewpoint_id: f for f in features}

    # 6. 动作感知打分与对比
    logger.info("Evaluating action-conditioned scores across all 5 action categories...")
    encoder = ActionEncoder(cfg.action_weights)
    scorer = ActionConditionedScorer(cfg.scoring)

    actions_results: Dict[str, Any] = {}
    comparison_reports: Dict[str, Any] = {}

    target_actions = [a.value for a in ALL_ACTION_CLASSES]

    for act_name in target_actions:
        act_embed = encoder.encode(act_name)
        scores = scorer.score_batch(features, act_embed, geom_map)

        # 4 基线对比报告
        comp_rep = compare_all_baselines(
            viewpoints=checked_candidates,
            action_scores=scores,
            features=features,
            geometry_qualities=geom_qualities,
            action=act_embed,
            human_position=human_pose.position,
        )
        comparison_reports[act_name] = comp_rep

        # v9 选出的动作感知最优视点
        v9_best_vp, v9_best_s = ViewpointSelector.select(
            viewpoints=checked_candidates,
            action_scores=scores,
            geometry_qualities=geom_qualities,
            strategy="action_conditioned",
        )
        v9_best_feat = feat_map[v9_best_vp.viewpoint_id]

        # 计算 v8 视点在当前动作下的得分
        v8_s_in_curr_act = next(s for s in scores if s.viewpoint_id == v8_best_vp.viewpoint_id)
        v8_feat = feat_map[v8_best_vp.viewpoint_id]

        actions_results[act_name] = {
            "action": act_name,
            "critical_regions": act_embed.critical_regions,
            "preferred_angle_range": act_embed.preferred_angle_range,
            "optimal_distance": act_embed.optimal_distance,
            "geometry_best_viewpoint": {
                "viewpoint_id": v8_best_vp.viewpoint_id,
                "position": [round(float(x), 2) for x in v8_best_vp.position],
                "yaw_deg": round(float(v8_best_vp.yaw_deg), 1),
                "viewing_angle_deg": round(float(v8_feat.viewing_angle_deg), 1),
                "distance": round(float(v8_feat.distance), 2),
                "geometry_score": v8_best_q.visibility_score,
                "action_total_score": v8_s_in_curr_act.total_score,
            },
            "action_conditioned_viewpoint": {
                "viewpoint_id": v9_best_vp.viewpoint_id,
                "position": [round(float(x), 2) for x in v9_best_vp.position],
                "yaw_deg": round(float(v9_best_vp.yaw_deg), 1),
                "viewing_angle_deg": round(float(v9_best_feat.viewing_angle_deg), 1),
                "distance": round(float(v9_best_feat.distance), 2),
                "geometry_score": v9_best_s.geometry_score,
                "action_delta": v9_best_s.action_delta,
                "action_total_score": v9_best_s.total_score,
            },
            "viewpoint_shifted": bool(v9_best_vp.viewpoint_id != v8_best_vp.viewpoint_id),
            "score_gain": round(v9_best_s.total_score - v8_s_in_curr_act.total_score, 3),
        }

    # 7. 汇总实验报告
    experiment_summary = {
        "experiment_name": "v9_action_conditioned_view_comparison",
        "scene_id": scene_id,
        "human_position": [float(x) for x in human_pose.position],
        "total_feasible_candidates": sum(1 for v in checked_candidates if v.feasible),
        "v8_geometry_best_view_id": v8_best_vp.viewpoint_id,
        "actions": actions_results,
        "detailed_comparison_reports": comparison_reports,
    }

    # 持久化输出
    out_json_path = res_dir / "v9_action_comparison.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2, ensure_ascii=False)

    # 复制一份到 visualizations/v9_demo/ 作为直观展示
    vis_dir = get_data_root() / "visualizations" / "v9_demo"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with open(vis_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(comparison_reports["fall"], f, indent=2, ensure_ascii=False)

    # 生成可视化对比图
    from ea_avs_mvp_v9.visualization.action_comparison_plotter import plot_action_comparison_figure
    plot_action_comparison_figure(experiment_summary, vis_dir / "action_comparison.png")
    plot_action_comparison_figure(experiment_summary, res_dir / "action_comparison.png")

    # 8. 打印结构化对比表格
    print("\n" + "=" * 90)
    print("  ACTIVEVIEW v9.0 Multi-Action Viewpoint Comparison Experiment (Q(v|A) vs Q_geom(v))")
    print("=" * 90)
    print(f"{'Action':<12} | {'v8 Geometry Best':<18} | {'v9 Action-Conditioned':<22} | {'Angle Shift':<12} | {'Gain':<8} | {'Shifted?'}")
    print("-" * 90)

    for act_name, res in actions_results.items():
        v8_info = f"{res['geometry_best_viewpoint']['viewpoint_id']} ({res['geometry_best_viewpoint']['action_total_score']:.3f})"
        v9_info = f"{res['action_conditioned_viewpoint']['viewpoint_id']} ({res['action_conditioned_viewpoint']['action_total_score']:.3f})"
        ang_shift = f"{res['geometry_best_viewpoint']['viewing_angle_deg']:.0f}° -> {res['action_conditioned_viewpoint']['viewing_angle_deg']:.0f}°"
        gain_str = f"{res['score_gain']:+.3f}"
        shift_str = "YES" if res["viewpoint_shifted"] else "NO"
        print(f"{act_name.upper():<12} | {v8_info:<18} | {v9_info:<22} | {ang_shift:<12} | {gain_str:<8} | {shift_str}")

    print("=" * 90)
    print(f"Results successfully saved to: {out_json_path}\n")

    return experiment_summary


def main():
    parser = argparse.ArgumentParser(description="Run v9.0 Multi-Action Viewpoint Comparison Experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    run_action_comparison_experiment(config_path=args.config, output_dir=args.output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
