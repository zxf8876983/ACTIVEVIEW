"""
v9.1 遮挡与空间布局鲁棒性实验脚本 —— run_v91_occlusion_experiment.py
==================================================================

功能：
    在 Habitat 室内环境及家具几何结构下，通过改变机器人初始站位与人体周边空间遮挡条件，
    评测 LearnableViewScorer (Q(v | H)) 是否能自适应规避遮挡视线、优选关键部位高可见性的观察视角。

输出：
    data/ActiveView/results/v91_occlusion_experiment_report.json

运行方式：
    python -m ea_avs_mvp_v9.scripts.run_v91_occlusion_experiment
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

from ea_avs_mvp_v9.core.config import load_v9_config
from ea_avs_mvp_v9.core.paths import get_data_root
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.inference.predict_view import ViewPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v91_occlusion")


def run_occlusion_experiments(
    config_path: str = None,
    checkpoint_path: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """执行空间布局与遮挡鲁棒性评测。"""
    cfg = load_v9_config(config_path)
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    if output_dir:
        res_dir = Path(output_dir) if Path(output_dir).is_absolute() else get_data_root() / output_dir
    else:
        res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    ckpt_p = Path(checkpoint_path) if checkpoint_path else (get_data_root() / "checkpoints/model_checkpoint.pth")
    predictor = ViewPredictor(checkpoint_path=ckpt_p)

    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    # 测试 3 种不同的空间初始位置配置
    test_robot_start_positions = [
        {"name": "standard_front", "pos": [1.5, -1.60, 6.8]},
        {"name": "side_corner", "pos": [3.2, -1.60, 4.0]},
        {"name": "narrow_corridor", "pos": [0.3, -1.60, 3.5]},
    ]

    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)
    gt_joints = humanoid.get_gt_joint_positions()

    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pose.position,
        human_yaw_deg=human_pose.yaw_deg,
        ground_height=cfg.robot.get("ground_height", -1.60),
    )

    checker = ConstraintChecker(env_adapter=env_adapter, config=cfg.viewpoint)
    feat_extractor = ViewFeatureExtractor(cfg.camera)

    experiment_cases = []

    for r_case in test_robot_start_positions:
        c_name = r_case["name"]
        r_pos = r_case["pos"]

        checked_candidates = checker.filter_feasible_viewpoints(
            raw_candidates,
            human_position=human_pose.position,
            human_joints_3d=gt_joints,
            robot_start_pos=r_pos,
        )

        features = feat_extractor.extract_batch(checked_candidates, gt_joints, human_yaw_deg=human_pose.yaw_deg)
        feat_map = {f.viewpoint_id: f for f in features}

        pred_res = predictor.predict_viewpoints(
            viewpoints=checked_candidates,
            features=features,
            human_joints_3d=gt_joints,
            human_yaw_deg=human_pose.yaw_deg,
            action_metadata="sitting",
        )

        best_id = pred_res["best_viewpoint_id"]
        best_f = feat_map[best_id]

        feasible_count = sum(1 for v in checked_candidates if v.feasible)
        mean_vis_loss = float(np.mean([f.visibility_loss_ratio for f in features if f.feasible])) if feasible_count > 0 else 1.0

        experiment_cases.append({
            "case_name": c_name,
            "robot_start_position": r_pos,
            "feasible_viewpoints": feasible_count,
            "total_candidates": len(checked_candidates),
            "mean_pool_visibility_loss": round(mean_vis_loss, 3),
            "selected_best_view": {
                "viewpoint_id": best_id,
                "predicted_score": pred_res["best_predicted_score"],
                "distance": best_f.distance,
                "viewing_angle_deg": best_f.viewing_angle_deg,
                "pose_coverage": best_f.pose_coverage,
                "body_part_visibilities": best_f.body_part_visibilities,
            },
        })

    env_adapter.close()

    summary_report = {
        "experiment_name": "v91_occlusion_and_spatial_robustness_study",
        "scene_id": scene_id,
        "human_position": [float(x) for x in human_pose.position],
        "cases": experiment_cases,
    }

    out_file = res_dir / "v91_occlusion_experiment_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("  ACTIVEVIEW v9.1 Occlusion & Spatial Geometry Robustness Study")
    print("=" * 90)
    print(f"{'Case Name':<18} | {'Robot Start Pos':<20} | {'Feasible':<10} | {'Selected View':<16} | {'Coverage'}")
    print("-" * 90)

    for c in experiment_cases:
        p_str = f"[{c['robot_start_position'][0]:.1f}, {c['robot_start_position'][2]:.1f}]"
        f_str = f"{c['feasible_viewpoints']}/{c['total_candidates']}"
        v_str = c["selected_best_view"]["viewpoint_id"]
        cov_str = f"{c['selected_best_view']['pose_coverage']*100:.1f}%"
        print(f"{c['case_name']:<18} | {p_str:<20} | {f_str:<10} | {v_str:<16} | {cov_str}")

    print("=" * 90)
    print(f"Occlusion report saved to: {out_file}\n")
    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Run v9.1 Occlusion & Spatial Robustness Experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_occlusion_experiments(config_path=args.config, checkpoint_path=args.checkpoint, output_dir=args.output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
