#!/usr/bin/env python3
"""
Comprehensive Real Perception Experiments Suite —— run_real_perception_experiments.py (v11.4.1)
=============================================================================================

职责：
    1. Experiment 1: GT Skeleton vs Estimated Skeleton 对比实验 (证明 GT 仅为理论上界)；
    2. Experiment 2: Viewpoint Robustness 视角旋转鲁棒性评测 (8 个离散角度 0°~315°)；
    3. Experiment 3: Occlusion Robustness 物理遮挡鲁棒性评测 (Easy, Medium, Hard 三档)；
    4. Experiment 4: Closed-Loop Active Perception 主动感知闭环评测汇总；
    5. 保存结构化实验结果到 results/v11_4_1_experiments.json。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry, DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.perception_pipeline import HabitatPerceptionPipeline, get_perception_pipeline
from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
from ea_avs_mvp_v11.core.paths import get_data_root
from tools.dataset_generation.generate_16class_amass_dataset import synthesize_canonical_motion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("real_perception_experiments")


def run_all_experiments() -> Dict[str, Any]:
    data_root = get_data_root()
    stgcn_ckpt = data_root / "checkpoints" / "v11_st_gcn" / "best_st_gcn_model.pth"
    classifier = ActionClassifier(checkpoint_path=stgcn_ckpt if stgcn_ckpt.exists() else None)
    aligner = CanonicalSkeletonAligner()
    pipeline = get_perception_pipeline()

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 实验一: GT Skeleton vs Estimated Skeleton
    # =========================================================================
    logger.info("Running Experiment 1: GT Skeleton vs Estimated Skeleton...")
    exp1_records = []
    num_samples_per_class = 20

    for act_id, act_name in enumerate(DEFAULT_ACTION_CATEGORIES):
        gt_correct, est_correct = 0, 0
        gt_confs, est_confs = [], []
        gt_ents, est_ents = [], []

        for i in range(num_samples_per_class):
            base_m = synthesize_canonical_motion(act_name, seed=act_id * 5000 + i)
            # GT 推理 (无噪声、无遮挡)
            gt_canon = aligner.align(base_m)
            gt_pred = classifier.predict_sequence(gt_canon, is_normalized=True, apply_canonical=False)
            if gt_pred.predicted_class == act_id:
                gt_correct += 1
            gt_confs.append(gt_pred.top1_confidence)
            gt_ents.append(gt_pred.entropy)

            # Estimated 推理 (由传感器与 Pose 估计器提取，包含距离噪声与轻度遮挡)
            obs = pipeline.observe(
                scene_id="apartment_1",
                human_state={"position": [0, 0, 0], "placement_difficulty": 0.5},
                robot_viewpoint={"angle": 30.0, "distance": 2.5, "position": [1.0, 1.25, 2.0]},
                base_motion_seq=base_m,
            )
            est_canon = aligner.align(obs["skeleton"])
            est_pred = classifier.predict_sequence(est_canon, is_normalized=True, apply_canonical=False)
            if est_pred.predicted_class == act_id:
                est_correct += 1
            est_confs.append(est_pred.top1_confidence)
            est_ents.append(est_pred.entropy)

        exp1_records.append({
            "action_id": act_id,
            "action_label": act_name,
            "gt_accuracy": round(gt_correct / num_samples_per_class * 100, 2),
            "estimated_accuracy": round(est_correct / num_samples_per_class * 100, 2),
            "gt_mean_confidence": round(float(np.mean(gt_confs)), 4),
            "estimated_mean_confidence": round(float(np.mean(est_confs)), 4),
            "gt_mean_entropy": round(float(np.mean(gt_ents)), 4),
            "estimated_mean_entropy": round(float(np.mean(est_ents)), 4),
        })

    # =========================================================================
    # 实验二: Viewpoint Robustness (0° ~ 315°)
    # =========================================================================
    logger.info("Running Experiment 2: Viewpoint Robustness across 8 Discrete Angles...")
    test_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    exp2_records = []

    for ang in test_angles:
        corr_before, corr_after = 0, 0
        confs, ents = [], []
        tot = 0

        for act_id, act_name in enumerate(DEFAULT_ACTION_CATEGORIES):
            for i in range(10):
                base_m = synthesize_canonical_motion(act_name, seed=act_id * 3000 + i + 999)
                obs = pipeline.observe(
                    scene_id="apartment_1",
                    human_state={"position": [0, 0, 0], "placement_difficulty": 0.2},
                    robot_viewpoint={"angle": ang, "distance": 2.0, "position": [0, 1.25, 2.0]},
                    base_motion_seq=base_m,
                )
                est_skel = obs["skeleton"]

                # 未经过 Canonical 对齐
                pred_raw = classifier.predict_sequence(est_skel, is_normalized=True, apply_canonical=False)
                if pred_raw.predicted_class == act_id:
                    corr_before += 1

                # 经过 Canonical 对齐
                canon_skel = aligner.align(est_skel)
                pred_canon = classifier.predict_sequence(canon_skel, is_normalized=True, apply_canonical=False)
                if pred_canon.predicted_class == act_id:
                    corr_after += 1
                confs.append(pred_canon.top1_confidence)
                ents.append(pred_canon.entropy)
                tot += 1

        exp2_records.append({
            "angle_deg": ang,
            "accuracy_without_canonical": round(corr_before / tot * 100, 2),
            "accuracy_with_canonical": round(corr_after / tot * 100, 2),
            "mean_confidence": round(float(np.mean(confs)), 4),
            "mean_entropy": round(float(np.mean(ents)), 4),
        })

    # =========================================================================
    # 实验三: Occlusion Robustness (Easy, Medium, Hard)
    # =========================================================================
    logger.info("Running Experiment 3: Physical Occlusion Robustness...")
    occlusion_tiers = [
        {"tier": "Easy (Vis > 0.80)", "placement_diff": 0.10, "dist": 1.8},
        {"tier": "Medium (Vis 0.50~0.80)", "placement_diff": 0.45, "dist": 2.5},
        {"tier": "Hard (Vis < 0.50)", "placement_diff": 0.85, "dist": 3.2},
    ]

    exp3_records = []

    for tier_cfg in occlusion_tiers:
        tot, corr = 0, 0
        confs, ents, vis_ratios = [], [] ,[]

        for act_id, act_name in enumerate(DEFAULT_ACTION_CATEGORIES):
            for i in range(15):
                base_m = synthesize_canonical_motion(act_name, seed=act_id * 7000 + i + 123)
                human_st = {"position": [0, 0, 0], "placement_difficulty": tier_cfg["placement_diff"]}
                robot_vp = {"angle": 45.0, "distance": tier_cfg["dist"], "position": [1.0, 1.2, tier_cfg["dist"]]}
                obs = pipeline.observe(
                    scene_id="apartment_1",
                    human_state=human_st,
                    robot_viewpoint=robot_vp,
                    base_motion_seq=base_m,
                )
                canon_skel = aligner.align(obs["skeleton"])
                pred = classifier.predict_sequence(canon_skel, is_normalized=True, apply_canonical=False, skeleton_source="estimated")
                if pred.predicted_class == act_id:
                    corr += 1
                confs.append(pred.top1_confidence)
                ents.append(pred.entropy)
                vis_ratios.append(obs["visible_ratio"])
                tot += 1

        exp3_records.append({
            "tier": tier_cfg["tier"],
            "placement_difficulty": tier_cfg["placement_diff"],
            "mean_visible_ratio": round(float(np.mean(vis_ratios)), 4),
            "accuracy": round(corr / tot * 100, 2),
            "mean_confidence": round(float(np.mean(confs)), 4),
            "mean_entropy": round(float(np.mean(ents)), 4),
        })

    # 整合所有实验数据
    all_experiments = {
        "experiment_1_gt_vs_estimated": {
            "title": "GT Skeleton vs Estimated Skeleton",
            "overall_gt_accuracy": round(float(np.mean([r["gt_accuracy"] for r in exp1_records])), 2),
            "overall_estimated_accuracy": round(float(np.mean([r["estimated_accuracy"] for r in exp1_records])), 2),
            "per_class_results": exp1_records,
        },
        "experiment_2_viewpoint_robustness": {
            "title": "Viewpoint Robustness across Angles",
            "mean_acc_without_canonical": round(float(np.mean([r["accuracy_without_canonical"] for r in exp2_records])), 2),
            "mean_acc_with_canonical": round(float(np.mean([r["accuracy_with_canonical"] for r in exp2_records])), 2),
            "angle_results": exp2_records,
        },
        "experiment_3_occlusion_robustness": {
            "title": "Physical Occlusion Robustness",
            "tier_results": exp3_records,
        },
    }

    out_file = results_dir / "v11_4_2_experiments.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)

    logger.info("Successfully executed all experiments and saved results to: %s", out_file)
    return all_experiments



if __name__ == "__main__":
    run_all_experiments()
