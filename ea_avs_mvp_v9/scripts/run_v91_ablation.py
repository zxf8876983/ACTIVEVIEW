"""
v9.1 学习型模型消融实验脚本 —— run_v91_ablation.py
=================================================

功能：
    执行 v9.1 模型的三项核心消融实验：
    1. Ablation 1 (Without Action Feature): 仅输入人体姿态 + 候选视角，消融动作 One-hot 特征；
    2. Ablation 2 (Without Human Pose): 仅输入动作类别 + 候选视角，消融 16 骨骼关键点坐标；
    3. Ablation 3 (Rule vs Learning): 规则打分 (v9.0) vs 学习型打分 (v9.1 Full Model) 对比。

输出：
    data/ActiveView/results/v91_ablation_report.json

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_v91_ablation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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
from ea_avs_mvp_v9.inference.predict_view import ViewPredictor
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v91_ablation")


def run_v91_ablation_experiment(
    config_path: str = None,
    checkpoint_path: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """执行 v9.1 完整消融实验。"""
    cfg = load_v9_config(config_path)
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    if output_dir:
        res_dir = Path(output_dir) if Path(output_dir).is_absolute() else get_data_root() / output_dir
    else:
        res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    ckpt_p = Path(checkpoint_path) if checkpoint_path else (get_data_root() / "checkpoints/model_checkpoint.pth")
    if not ckpt_p.exists():
        logger.info("Training initial checkpoint for ablation...")
        from ea_avs_mvp_v9.training.dataset import generate_scoring_dataset
        from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer
        from ea_avs_mvp_v9.training.trainer import ViewScorerTrainer
        t_ds, v_ds = generate_scoring_dataset(num_episodes=150, seed=42)
        m = LearnableViewScorer()
        trainer = ViewScorerTrainer(m)
        trainer.train(t_ds, v_ds, num_epochs=30, batch_size=16, checkpoint_path=ckpt_p)

    predictor = ViewPredictor(checkpoint_path=ckpt_p)

    # 1. 加载仿真环境与提取姿态
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

    # 2. 提取基础几何与特征
    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    feat_map = {f.viewpoint_id: f for f in features}

    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    geom_ranked = geom_evaluator.rank_viewpoints(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    geom_map = {q.viewpoint_id: q.visibility_score for _, q in geom_ranked}

    encoder = ActionEncoder(cfg.action_weights)
    rule_scorer = ActionConditionedScorer(cfg.scoring)

    ablation_summary: Dict[str, Any] = {}
    target_actions = [a.value for a in ALL_ACTION_CLASSES]

    for act_name in target_actions:
        act_embed = encoder.encode(act_name)
        rule_scores = rule_scorer.score_batch(features, act_embed, geom_map)
        vp_rule, s_rule = ViewpointSelector.select(checked_candidates, rule_scores, strategy="action_conditioned")

        # Full Model (Pose + Action + View)
        pred_full = predictor.predict_viewpoints(
            checked_candidates, features, gt_joints, human_yaw_deg=human_pose.yaw_deg, action_label=act_name
        )

        # Ablation 1: w/o Action (ablate_action=True)
        pred_no_action = predictor.predict_viewpoints(
            checked_candidates, features, gt_joints, human_yaw_deg=human_pose.yaw_deg, action_label=act_name, ablate_action=True
        )

        # Ablation 2: w/o Pose (ablate_pose=True)
        pred_no_pose = predictor.predict_viewpoints(
            checked_candidates, features, gt_joints, human_yaw_deg=human_pose.yaw_deg, action_label=act_name, ablate_pose=True
        )

        ablation_summary[act_name] = {
            "action": act_name,
            "v90_rule_optimal": {
                "viewpoint_id": vp_rule.viewpoint_id,
                "score": s_rule.total_score,
                "viewing_angle_deg": feat_map[vp_rule.viewpoint_id].viewing_angle_deg,
            },
            "v91_full_model": {
                "viewpoint_id": pred_full["best_viewpoint_id"],
                "predicted_score": pred_full["best_predicted_score"],
                "viewing_angle_deg": feat_map[pred_full["best_viewpoint_id"]].viewing_angle_deg,
                "matches_rule_oracle": bool(pred_full["best_viewpoint_id"] == vp_rule.viewpoint_id),
            },
            "ablation_without_action": {
                "viewpoint_id": pred_no_action["best_viewpoint_id"],
                "predicted_score": pred_no_action["best_predicted_score"],
                "viewing_angle_deg": feat_map[pred_no_action["best_viewpoint_id"]].viewing_angle_deg,
            },
            "ablation_without_pose": {
                "viewpoint_id": pred_no_pose["best_viewpoint_id"],
                "predicted_score": pred_no_pose["best_predicted_score"],
                "viewing_angle_deg": feat_map[pred_no_pose["best_viewpoint_id"]].viewing_angle_deg,
            },
        }

    report = {
        "experiment_name": "v91_feature_ablation_and_rule_comparison",
        "scene_id": scene_id,
        "actions": ablation_summary,
    }

    out_file = res_dir / "v91_ablation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印消融对比表格
    print("\n" + "=" * 96)
    print("  ACTIVEVIEW v9.1 Feature Ablation Study & Rule vs Learning Comparison")
    print("=" * 96)
    print(f"{'Action':<12} | {'v9.0 Rule':<18} | {'v9.1 Full Model':<20} | {'w/o Action':<18} | {'w/o Pose'}")
    print("-" * 96)

    for act_name, d in ablation_summary.items():
        r_str = f"{d['v90_rule_optimal']['viewpoint_id']} ({d['v90_rule_optimal']['viewing_angle_deg']:.0f}°)"
        f_str = f"{d['v91_full_model']['viewpoint_id']} ({d['v91_full_model']['viewing_angle_deg']:.0f}°)"
        na_str = f"{d['ablation_without_action']['viewpoint_id']} ({d['ablation_without_action']['viewing_angle_deg']:.0f}°)"
        np_str = f"{d['ablation_without_pose']['viewpoint_id']} ({d['ablation_without_pose']['viewing_angle_deg']:.0f}°)"
        print(f"{act_name.upper():<12} | {r_str:<18} | {f_str:<20} | {na_str:<18} | {np_str}")

    print("=" * 96)
    print(f"Ablation report saved to: {out_file}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run v9.1 Feature Ablation Experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_v91_ablation_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
