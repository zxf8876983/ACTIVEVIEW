"""
ST-GCN 动作识别多实验基准评测引擎 —— evaluate_benchmarks.py
======================================================

职责：
    1. 运行三大标准对比实验：
       - [实验一: Clean Perception Baseline (Oracle 上界)]: 干净无遮挡环境下的动作识别准确率；
       - [实验二: Habitat Perception Baseline (室内仿真观测)]: 室内复杂视角与阴影下的动作识别表现；
       - [实验三: Active Viewpoint Uncertainty Analysis (视点不确定度)]: 统计不同视角 (r, theta) 下的准确率与信息熵；
    2. 生成定量对比表格与指标分析；
    3. 输出 `phase3_benchmark_results.json`。
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

    logger.info(">>> 1. Running Experiment 1: Clean Perception Baseline (Oracle)...")
    clean_loader = create_action_dataloader(test_clean_data_p, test_clean_label_p, batch_size=16, shuffle=False)
    exp1_clean_res = trainer.evaluate(clean_loader)

    logger.info(">>> 2. Running Experiment 2: Habitat Perception Baseline...")
    hab_loader = create_action_dataloader(test_hab_data_p, test_hab_label_p, batch_size=16, shuffle=False)
    exp2_hab_res = trainer.evaluate(hab_loader)

    logger.info(">>> 3. Running Experiment 3: Active Viewpoint Uncertainty Analysis...")
    hab_data_np = np.load(test_hab_data_p)
    hab_labels_np = np.load(test_hab_label_p)
    with open(test_hab_vp_p, "r", encoding="utf-8") as f:
        vp_meta = json.load(f)

    # 按视角分组统计准确率与信息熵
    predictions = classifier.predict_batch(hab_data_np)
    viewpoint_stats: Dict[str, Dict[str, Any]] = {}

    for idx, (pred, item_vp) in enumerate(zip(predictions, vp_meta)):
        vp_info = item_vp["viewpoint"]
        radius = float(vp_info.get("radius", 2.0))
        angle_deg = float(vp_info.get("angle_deg", 0.0))
        key = f"r{radius:.1f}_a{int(angle_deg):03d}"

        is_correct = (pred.predicted_class == int(hab_labels_np[idx]))

        if key not in viewpoint_stats:
            viewpoint_stats[key] = {
                "radius": radius,
                "angle_deg": angle_deg,
                "total": 0,
                "correct": 0,
                "entropies": [],
                "top1_confs": [],
            }

        viewpoint_stats[key]["total"] += 1
        viewpoint_stats[key]["correct"] += (1 if is_correct else 0)
        viewpoint_stats[key]["entropies"].append(pred.normalized_entropy)
        viewpoint_stats[key]["top1_confs"].append(pred.top1_confidence)

    vp_analysis = {}
    for k, v in viewpoint_stats.items():
        acc = v["correct"] / max(v["total"], 1)
        mean_ent = float(np.mean(v["entropies"]))
        mean_conf = float(np.mean(v["top1_confs"]))
        vp_analysis[k] = {
            "radius": v["radius"],
            "angle_deg": v["angle_deg"],
            "accuracy": round(acc, 4),
            "mean_normalized_entropy": round(mean_ent, 4),
            "mean_top1_confidence": round(mean_conf, 4),
            "num_samples": v["total"],
        }

    # 汇总报表
    results = {
        "experiment_1_clean_oracle": {
            "accuracy": exp1_clean_res["accuracy"],
            "loss": exp1_clean_res["loss"],
            "mean_entropy": exp1_clean_res["mean_entropy"],
            "per_class_accuracy": {action_classes[k]: round(v, 4) for k, v in exp1_clean_res["per_class_accuracy"].items()},
            "num_samples": exp1_clean_res["num_samples"],
        },
        "experiment_2_habitat_baseline": {
            "accuracy": exp2_hab_res["accuracy"],
            "loss": exp2_hab_res["loss"],
            "mean_entropy": exp2_hab_res["mean_entropy"],
            "per_class_accuracy": {action_classes[k]: round(v, 4) for k, v in exp2_hab_res["per_class_accuracy"].items()},
            "num_samples": exp2_hab_res["num_samples"],
            "accuracy_drop_vs_oracle": round(exp1_clean_res["accuracy"] - exp2_hab_res["accuracy"], 4),
            "entropy_increase_vs_oracle": round(exp2_hab_res["mean_entropy"] - exp1_clean_res["mean_entropy"], 4),
        },
        "experiment_3_viewpoint_uncertainty": {
            "num_viewpoints_evaluated": len(vp_analysis),
            "viewpoints": vp_analysis,
        },
    }

    out_p = output_report_path or (action_data_root / "phase3_benchmark_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=================================================================")
    logger.info("PHASE 3 BENCHMARK RESULTS SUMMARY:")
    logger.info("  [Exp 1: Clean Oracle]   Accuracy: %.2f%% | Mean Entropy: %.4f",
                results["experiment_1_clean_oracle"]["accuracy"] * 100, results["experiment_1_clean_oracle"]["mean_entropy"])
    logger.info("  [Exp 2: Habitat Baseline] Accuracy: %.2f%% | Mean Entropy: %.4f | Drop: -%.2f%%",
                results["experiment_2_habitat_baseline"]["accuracy"] * 100, results["experiment_2_habitat_baseline"]["mean_entropy"],
                results["experiment_2_habitat_baseline"]["accuracy_drop_vs_oracle"] * 100)
    logger.info("Saved complete benchmark results to: %s", out_p)
    logger.info("=================================================================")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN Benchmarks")
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
