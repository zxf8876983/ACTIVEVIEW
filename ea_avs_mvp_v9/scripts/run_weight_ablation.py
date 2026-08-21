"""
v9.0 权重敏感性与消融实验脚本 —— run_weight_ablation.py
=====================================================

功能：
    验证动作感知增益权重 w_action 对视点选择的影响：
    测试权重组合 (w_geometry / w_action):
        - 0.8 / 0.2 (几何主导)
        - 0.6 / 0.4 (基准配置)
        - 0.4 / 0.6 (动作感知主导)
        - 0.2 / 0.8 (强动作驱动)

输出：
    data/ActiveView/results/weight_ablation_report.json

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_weight_ablation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_weight_ablation")

WEIGHT_PAIRS: List[Tuple[float, float]] = [
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
]


def run_weight_ablation_experiment(
    config_path: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """执行权重敏感性消融实验。"""
    cfg = load_v9_config(config_path)
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    if output_dir:
        res_dir = Path(output_dir) if Path(output_dir).is_absolute() else get_data_root() / output_dir
    else:
        res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. 仿真环境与视点生成
    logger.info("Initializing Scene and extracting human joints for weight ablation...")
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

    # 2. 基础几何与特征提取
    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    ranked_geom_pairs = geom_evaluator.rank_viewpoints(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    geom_qualities = [q for _, q in ranked_geom_pairs]
    geom_map = {q.viewpoint_id: q.visibility_score for q in geom_qualities}

    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
    feat_map = {f.viewpoint_id: f for f in features}

    encoder = ActionEncoder(cfg.action_weights)
    target_actions = [a.value for a in ALL_ACTION_CLASSES]

    # 3. 权重消融循环
    logger.info("Executing weight ablation across %d weight configurations...", len(WEIGHT_PAIRS))
    ablation_results: Dict[str, Any] = {}

    for w_geom, w_act in WEIGHT_PAIRS:
        weight_key = f"{w_geom:.1f}/{w_act:.1f}"
        scorer = ActionConditionedScorer({"w_geometry": w_geom, "w_action": w_act})

        act_dict: Dict[str, Any] = {}
        for act_name in target_actions:
            act_embed = encoder.encode(act_name)
            scores = scorer.score_batch(features, act_embed, geom_map)

            best_vp, best_score = ViewpointSelector.select(
                viewpoints=checked_candidates,
                action_scores=scores,
                geometry_qualities=geom_qualities,
                strategy="action_conditioned",
            )
            feat = feat_map[best_vp.viewpoint_id]

            act_dict[act_name] = {
                "selected_view_id": best_vp.viewpoint_id,
                "position": [round(float(x), 2) for x in best_vp.position],
                "viewing_angle_deg": round(float(feat.viewing_angle_deg), 1),
                "distance": round(float(feat.distance), 2),
                "total_score": best_score.total_score,
                "geometry_score": best_score.geometry_score,
                "action_delta": best_score.action_delta,
            }

        ablation_results[weight_key] = {
            "w_geometry": w_geom,
            "w_action": w_act,
            "actions": act_dict,
        }

    # 4. 持久化输出
    report = {
        "experiment_name": "v9_weight_sensitivity_ablation",
        "scene_id": scene_id,
        "human_position": [float(x) for x in human_pose.position],
        "weight_configurations": ablation_results,
    }

    out_file = res_dir / "weight_ablation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 5. 打印汇总表格
    print("\n" + "=" * 80)
    print("  ACTIVEVIEW v9.0 Weight Sensitivity Ablation (w_geom / w_act)")
    print("=" * 80)
    header = f"{'Action':<12} | " + " | ".join([f"{k} (View/Score)" for k in ablation_results.keys()])
    print(header)
    print("-" * 80)

    for act_name in target_actions:
        row_items = [f"{act_name.upper():<12}"]
        for w_key in ablation_results.keys():
            item = ablation_results[w_key]["actions"][act_name]
            v_str = f"{item['selected_view_id']} ({item['total_score']:.3f})"
            row_items.append(f"{v_str:<18}")
        print(" | ".join(row_items))

    print("=" * 80)
    print(f"Weight ablation report saved to: {out_file}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run v9.0 Weight Ablation Experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_weight_ablation_experiment(config_path=args.config, output_dir=args.output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
