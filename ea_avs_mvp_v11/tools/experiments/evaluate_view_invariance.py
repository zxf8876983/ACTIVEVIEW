#!/usr/bin/env python3
"""
View Rotation Robustness & Invariance Test —— evaluate_view_invariance.py
========================================================================

职责：
    1. 在 8 个离散观察视角 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) 下，
       对全量动作实例评估 ST-GCN 动作分类准确率与不确定度；
    2. 严格对比：
       - [Before]: 未加入 Canonical Alignment (直接输入 Camera Frame 骨架)
       - [After]:  加入 Canonical Skeleton Alignment (自动正规化至 Human Canonical Frame)
    3. 导出结构化分析数据至 `results/view_invariance_analysis.json`。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry, DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_view_invariance")


def evaluate_view_invariance() -> Dict[str, Any]:
    data_root = get_data_root()
    stgcn_ckpt = data_root / "checkpoints" / "v11_st_gcn" / "best_st_gcn_model.pth"
    classifier = ActionClassifier(checkpoint_path=stgcn_ckpt)
    registry = ActionRegistry(data_root=data_root, exclude_locomotion=True)
    aligner = CanonicalSkeletonAligner()

    angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    num_actions = len(registry.categories)

    results_before = {}
    results_after = {}

    logger.info("Evaluating view invariance across %d angles for %d categories...", len(angles), num_actions)

    for ang in angles:
        rad = math.radians(ang)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array(
            [
                [cos_a, 0.0, -sin_a],
                [0.0, 1.0, 0.0],
                [sin_a, 0.0, cos_a],
            ],
            dtype=np.float32,
        )

        corr_before, corr_after = 0, 0
        total = 0
        ent_before_list, ent_after_list = [], []
        conf_before_list, conf_after_list = [], []

        for act_id in range(num_actions):
            # 每类评估 20 个实例
            for inst_idx in range(20):
                base_skel = registry.get_skeleton_sequence(action_id=act_id, instance_idx=inst_idx, split="train")

                # 模拟相机视角旋转
                rot_skel = np.zeros_like(base_skel)
                for t in range(base_skel.shape[0]):
                    rot_skel[t] = (R @ base_skel[t].T).T

                # 1. Before: 无 Canonical Alignment
                pred_before = classifier.predict_sequence(rot_skel, is_normalized=True, apply_canonical=False)
                if pred_before.predicted_class == act_id:
                    corr_before += 1
                ent_before_list.append(pred_before.entropy)
                conf_before_list.append(pred_before.top1_confidence)

                # 2. After: 启用 Canonical Alignment
                pred_after = classifier.predict_sequence(rot_skel, is_normalized=True, apply_canonical=True)
                if pred_after.predicted_class == act_id:
                    corr_after += 1
                ent_after_list.append(pred_after.entropy)
                conf_after_list.append(pred_after.top1_confidence)

                total += 1

        acc_b = round(corr_before / total * 100, 2)
        acc_a = round(corr_after / total * 100, 2)

        results_before[f"{int(ang):03d}deg"] = {
            "angle_deg": ang,
            "accuracy_pct": acc_b,
            "mean_entropy": round(float(np.mean(ent_before_list)), 4),
            "mean_confidence": round(float(np.mean(conf_before_list)), 4),
            "num_samples": total,
        }
        results_after[f"{int(ang):03d}deg"] = {
            "angle_deg": ang,
            "accuracy_pct": acc_a,
            "mean_entropy": round(float(np.mean(ent_after_list)), 4),
            "mean_confidence": round(float(np.mean(conf_after_list)), 4),
            "num_samples": total,
        }

        logger.info("Angle: %5.1f deg | Before Acc: %6.2f%% | After Acc: %6.2f%% | Gain: %+6.2f%%",
                    ang, acc_b, acc_a, acc_a - acc_b)

    report_data = {
        "title": "ACTIVEVIEW v11.4.1 View Rotation Robustness & Invariance Analysis",
        "categories": registry.categories,
        "angles_evaluated": angles,
        "results_before_canonical_alignment": results_before,
        "results_after_canonical_alignment": results_after,
        "summary": {
            "mean_accuracy_before_pct": round(float(np.mean([v["accuracy_pct"] for v in results_before.values()])), 2),
            "mean_accuracy_after_pct": round(float(np.mean([v["accuracy_pct"] for v in results_after.values()])), 2),
            "variance_before": round(float(np.var([v["accuracy_pct"] for v in results_before.values()])), 2),
            "variance_after": round(float(np.var([v["accuracy_pct"] for v in results_after.values()])), 2),
        },
    }

    out_p = Path("results/view_invariance_analysis.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info("=================================================================")
    logger.info("  VIEW ROTATION ROBUSTNESS TEST COMPLETE!                       ")
    logger.info("  Mean Accuracy: Before=%.2f%% -> After=%.2f%%                   ",
                report_data["summary"]["mean_accuracy_before_pct"],
                report_data["summary"]["mean_accuracy_after_pct"])
    logger.info("  Variance:      Before=%.2f -> After=%.2f                       ",
                report_data["summary"]["variance_before"],
                report_data["summary"]["variance_after"])
    logger.info("  Saved results to: %s", out_p.resolve())
    logger.info("=================================================================")

    return report_data


if __name__ == "__main__":
    evaluate_view_invariance()
