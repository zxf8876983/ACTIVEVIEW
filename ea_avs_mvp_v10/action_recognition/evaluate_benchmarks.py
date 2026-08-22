"""
ST-GCN 动作识别多实验基准评测引擎 (Phase 3.1 科学一致性版本) —— evaluate_benchmarks.py
================================================================================

职责：
    1. 运行三大标准对比实验：
       - [实验一: Clean Perception Oracle (理想感知上界)]: 干净无遮挡环境下的动作识别表现 (Accuracy / Precision / Recall / F1 / Entropy)；
       - [实验二: Habitat Perception Baseline (室内仿真观测)]: 室内多视角与遮挡环境下的动作识别表现与性能退化；
       - [实验三: Active Viewpoint Uncertainty Analysis (视点不确定度)]: 统计 16 个网格视角 (r, theta) 下的准确率、F1 与归一化信息熵；
    2. 严格明确：Clean Perception Oracle 代表“理想感知条件”下的动作识别性能，绝非 GT 骨骼；
    3. 输出结构化 `phase3_benchmark_results.json` 报表。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn.functional as F

from ea_avs_mvp_v7.core.paths import get_data_root
from ea_avs_mvp_v10.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v10.action_recognition.action_dataset import create_action_dataloader
from ea_avs_mvp_v10.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v10.action_recognition.trainer import STGCNTrainer
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_benchmarks")


def run_all_benchmarks(
    checkpoint_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    skel_def = get_skeleton_definition()
    action_data_root = dataset_dir or (get_data_root() / "datasets" / "action")

    test_clean_data_p = action_data_root / "test" / "clean_perception" / "data.npy"
    test_clean_label_p = action_data_root / "test" / "clean_perception" / "labels.npy"

    test_hab_data_p = action_data_root / "test" / "habitat_perception" / "data.npy"
    test_hab_label_p = action_data_root / "test" / "habitat_perception" / "labels.npy"
    test_hab_vp_p = action_data_root / "test" / "habitat_perception" / "viewpoints.json"

    ckpt_p = checkpoint_path or (get_data_root() / "checkpoints" / "v10_st_gcn" / "best_st_gcn_model.pth")

    action_classes = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]
    classifier = ActionClassifier(checkpoint_path=ckpt_p, action_classes=action_classes, skel_def=skel_def)
    trainer = STGCNTrainer(model=classifier.model, num_classes=len(action_classes), skel_def=skel_def)

    logger.info(">>> 1. Running Experiment 1: Clean Perception Oracle (Ideal Perception Condition)...")
    clean_loader = create_action_dataloader(test_clean_data_p, test_clean_label_p, batch_size=32, shuffle=False)
    exp1_clean_res = trainer.evaluate(clean_loader)

    logger.info(">>> 2. Running Experiment 2: Habitat Perception Baseline (Indoor Multi-View)...")
    hab_loader = create_action_dataloader(test_hab_data_p, test_hab_label_p, batch_size=32, shuffle=False)
    exp2_hab_res = trainer.evaluate(hab_loader)

    logger.info(">>> 3. Running Experiment 3: Active Viewpoint Uncertainty Analysis...")
    hab_data_np = np.load(test_hab_data_p)
    hab_labels_np = np.load(test_hab_label_p)
    with open(test_hab_vp_p, "r", encoding="utf-8") as f:
        vp_meta = json.load(f)

    predictions = classifier.predict_batch(hab_data_np)
    viewpoint_stats: Dict[str, Dict[str, Any]] = {}

    for idx, (pred, item_vp) in enumerate(zip(predictions, vp_meta)):
        vp_info = item_vp["viewpoint"]
        radius = float(vp_info.get("radius", 2.0))
        angle_deg = float(vp_info.get("angle_deg", 0.0))
        key = f"r{radius:.1f}_a{int(angle_deg):03d}"

        label = int(hab_labels_np[idx])
        pred_cls = pred.predicted_class
        is_correct = (pred_cls == label)

        if key not in viewpoint_stats:
            viewpoint_stats[key] = {
                "radius": radius,
                "angle_deg": angle_deg,
                "total": 0,
                "correct": 0,
                "preds": [],
                "labels": [],
                "entropies": [],
                "norm_entropies": [],
                "top1_confs": [],
            }

        viewpoint_stats[key]["total"] += 1
        viewpoint_stats[key]["correct"] += (1 if is_correct else 0)
        viewpoint_stats[key]["preds"].append(pred_cls)
        viewpoint_stats[key]["labels"].append(label)
        viewpoint_stats[key]["entropies"].append(pred.entropy)
        viewpoint_stats[key]["norm_entropies"].append(pred.normalized_entropy)
        viewpoint_stats[key]["top1_confs"].append(pred.top1_confidence)

    vp_analysis = {}
    for k, v in viewpoint_stats.items():
        acc = v["correct"] / max(v["total"], 1)
        mean_ent = float(np.mean(v["entropies"]))
        mean_norm_ent = float(np.mean(v["norm_entropies"]))
        mean_conf = float(np.mean(v["top1_confs"]))

        # 计算该视角下的 Macro F1
        preds_arr = np.array(v["preds"])
        labels_arr = np.array(v["labels"])
        f1_list = []
        for c in range(len(action_classes)):
            tp = int(np.sum((preds_arr == c) & (labels_arr == c)))
            fp = int(np.sum((preds_arr == c) & (labels_arr != c)))
            fn = int(np.sum((preds_arr != c) & (labels_arr == c)))
            p = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
            r = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / max(p + r, 1e-8) if (p + r) > 0 else 0.0
            f1_list.append(f1)
        macro_f1 = float(np.mean(f1_list))

        vp_analysis[k] = {
            "radius": v["radius"],
            "angle_deg": v["angle_deg"],
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_normalized_uncertainty": round(mean_norm_ent, 4),
            "mean_top1_confidence": round(mean_conf, 4),
            "num_samples": v["total"],
        }

    # 汇总完整报表
    results = {
        "experiment_1_clean_perception_oracle": {
            "definition": "Ideal perception condition with unobstructed RGB view and standard distance; upper bound of downstream action classifier.",
            "accuracy": exp1_clean_res["accuracy"],
            "precision": exp1_clean_res["precision"],
            "recall": exp1_clean_res["recall"],
            "f1_score": exp1_clean_res["f1_score"],
            "loss": exp1_clean_res["loss"],
            "mean_entropy": exp1_clean_res["mean_entropy"],
            "mean_normalized_uncertainty": exp1_clean_res["mean_normalized_uncertainty"],
            "per_class_metrics": {action_classes[k]: v for k, v in exp1_clean_res["per_class_metrics"].items()},
            "num_samples": exp1_clean_res["num_samples"],
        },
        "experiment_2_habitat_perception_baseline": {
            "definition": "Realistic multi-view perception in indoor scene with distance variation and viewpoint obstacles.",
            "accuracy": exp2_hab_res["accuracy"],
            "precision": exp2_hab_res["precision"],
            "recall": exp2_hab_res["recall"],
            "f1_score": exp2_hab_res["f1_score"],
            "loss": exp2_hab_res["loss"],
            "mean_entropy": exp2_hab_res["mean_entropy"],
            "mean_normalized_uncertainty": exp2_hab_res["mean_normalized_uncertainty"],
            "per_class_metrics": {action_classes[k]: v for k, v in exp2_hab_res["per_class_metrics"].items()},
            "num_samples": exp2_hab_res["num_samples"],
            "accuracy_drop_vs_oracle": round(exp1_clean_res["accuracy"] - exp2_hab_res["accuracy"], 4),
            "f1_drop_vs_oracle": round(exp1_clean_res["f1_score"] - exp2_hab_res["f1_score"], 4),
            "entropy_increase_vs_oracle": round(exp2_hab_res["mean_entropy"] - exp1_clean_res["mean_entropy"], 4),
        },
        "experiment_3_viewpoint_uncertainty_analysis": {
            "num_viewpoints_evaluated": len(vp_analysis),
            "viewpoints": vp_analysis,
        },
    }

    out_p = output_report_path or (action_data_root / "phase3_benchmark_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=================================================================")
    logger.info("PHASE 3.1 BENCHMARK RESULTS SUMMARY:")
    logger.info("  [Exp 1: Clean Perception Oracle] Acc: %.2f%% | F1: %.4f | Entropy: %.4f | NormUncertainty: %.4f",
                results["experiment_1_clean_perception_oracle"]["accuracy"] * 100,
                results["experiment_1_clean_perception_oracle"]["f1_score"],
                results["experiment_1_clean_perception_oracle"]["mean_entropy"],
                results["experiment_1_clean_perception_oracle"]["mean_normalized_uncertainty"])
    logger.info("  [Exp 2: Habitat Perception Baseline] Acc: %.2f%% | F1: %.4f | Entropy: %.4f | NormUncertainty: %.4f | Acc Drop: -%.2f%%",
                results["experiment_2_habitat_perception_baseline"]["accuracy"] * 100,
                results["experiment_2_habitat_perception_baseline"]["f1_score"],
                results["experiment_2_habitat_perception_baseline"]["mean_entropy"],
                results["experiment_2_habitat_perception_baseline"]["mean_normalized_uncertainty"],
                results["experiment_2_habitat_perception_baseline"]["accuracy_drop_vs_oracle"] * 100)
    logger.info("Saved complete benchmark results to: %s", out_p)
    logger.info("=================================================================")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN Benchmarks (Phase 3.1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint) if args.checkpoint else None
    data_d = Path(args.dataset_dir) if args.dataset_dir else None
    out_p = Path(args.output) if args.output else None

    run_all_benchmarks(checkpoint_path=ckpt_p, dataset_dir=data_d, output_report_path=out_p)


if __name__ == "__main__":
    main()
