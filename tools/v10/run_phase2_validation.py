#!/usr/bin/env python3
"""
ACTIVEVIEW v10.0 Phase 2.1 科研验证集生成与 GT 精度评测流水线 —— run_phase2_validation.py
=============================================================================

职责：
    1. 从 Phase 1 数据集中抽取 10 个代表性样本 (包含 standing 与 walking 动作)；
    2. 对每个样本运行 RGB-D 3D 骨架估计；
    3. 调用 `SkeletonEvaluator` 与 Habitat GT 骨骼进行空间对齐与 MPJPE / PA-MPJPE / PCK 评测；
    4. 输出结构化的 `examples/v10_phase2_validation/sample_xxx/` 评测包：
       - rgb.png
       - estimated_skeleton_3d.png
       - gt_skeleton_3d.png
       - overlay_comparison.png
       - metrics.json
       - metadata.json
    5. 生成综合定量评测汇总表与失败案例分析报表。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_repo_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.evaluation.skeleton_alignment import transform_gt_to_camera_frame
from ea_avs_mvp_v10.evaluation.skeleton_evaluator import EvaluationMetrics, SkeletonEvaluator
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v10.visualization.skeleton_visualizer import SkeletonVisualizer
from tools.v10.skeleton_compare_visualizer import plot_skeleton_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase2_validation")


def run_phase2_1_validation(num_samples: int = 10) -> Dict[str, Any]:
    repo_root = get_repo_root()
    dataset_root = get_v10_dataset_root()
    skel_def = get_skeleton_definition()
    pipeline = V10PerceptionPipeline(dataset_root=dataset_root)
    evaluator = SkeletonEvaluator(skel_def=skel_def)
    visualizer = SkeletonVisualizer(skel_def=skel_def)

    val_dir = repo_root / "ea_avs_mvp_v10" / "examples" / "v10_phase2_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_root / "metadata" / "samples.json", "r", encoding="utf-8") as f:
        all_samples = json.load(f)["samples"]

    # 选取涵盖 standing 和 walking 的代表性多视角样本
    selected = []
    standing_samples = [s for s in all_samples if s["action_label"] == "standing"][:5]
    walking_samples = [s for s in all_samples if s["action_label"] == "walking"][:5]
    selected = standing_samples + walking_samples

    if len(selected) < num_samples:
        selected = all_samples[:num_samples]

    logger.info(">>> Running Phase 2.1 Scientific Validation on %d samples...", len(selected))

    eval_batch_data = []

    for idx, s_dict in enumerate(selected):
        sid = s_dict["sample_id"]
        action = s_dict["action_label"]
        sample_dir = val_dir / f"sample_{idx:03d}_{action}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_obj = V10Sample.from_dict(s_dict)
        rgb_img = np.array(Image.open(dataset_root / sample_obj.rgb_path))
        depth_map = np.load(dataset_root / sample_obj.depth_path)

        # 1. 估计 3D 骨架 (来自 RGB-D，严格无 GT 泄漏)
        skel, _ = pipeline.process_sample(sample_obj, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

        # 2. 读取 Habitat GT 骨骼
        gt_file = dataset_root / "ground_truth" / "skeleton" / f"{sid}.json"
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_joints_dict = json.load(f)["joints_3d"]

        c2w = np.array(sample_obj.camera_pose.matrix_4x4, dtype=np.float32)

        # 3. 评测指标
        metrics = evaluator.evaluate_sample(
            estimated_skeleton=skel,
            gt_joints_dict=gt_joints_dict,
            camera_matrix_4x4=c2w,
            sample_id=sid,
            action_label=action,
        )

        # 4. 保存单个评测文件: rgb.png
        Image.fromarray(rgb_img).save(sample_dir / "rgb.png")

        # 5. 保存 estimated_skeleton_3d.png
        fig_est = plt.figure(figsize=(6, 5.0))
        ax_est = fig_est.add_subplot(1, 1, 1, projection="3d")
        visualizer.draw_3d_limbs(
            ax_est,
            skel.joints_3d_camera,
            skel.perception_confidence,
            f"Estimated 3D Pose ({action.upper()})",
            is_normalized=False,
            view_mode="front",
        )
        plt.tight_layout()
        plt.savefig(sample_dir / "estimated_skeleton_3d.png", dpi=130)
        plt.close(fig_est)

        # 6. 保存 gt_skeleton_3d.png
        fig_gt = plt.figure(figsize=(6, 5.0))
        ax_gt = fig_gt.add_subplot(1, 1, 1, projection="3d")
        gt_cam_dict = transform_gt_to_camera_frame(gt_joints_dict, c2w)
        for name, p in gt_cam_dict.items():
            ax_gt.scatter(p[0], p[2], p[1], color="#E74C3C", s=25)
        gt_edges = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ("neck", "head"), ("pelvis", "spine3"), ("spine3", "neck"),
        ]
        for j1_n, j2_n in gt_edges:
            if j1_n in gt_cam_dict and j2_n in gt_cam_dict:
                p1 = gt_cam_dict[j1_n]
                p2 = gt_cam_dict[j2_n]
                ax_gt.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color="#E74C3C", linewidth=2.2)

        gt_pts = np.array(list(gt_cam_dict.values()))
        gt_span = max(np.ptp(gt_pts[:, 0]), np.ptp(gt_pts[:, 1]), np.ptp(gt_pts[:, 2]), 0.8)
        gt_mid = np.mean(gt_pts, axis=0)
        ax_gt.set_xlim(gt_mid[0] - gt_span/2, gt_mid[0] + gt_span/2)
        ax_gt.set_ylim(gt_mid[2] - gt_span/2, gt_mid[2] + gt_span/2)
        ax_gt.set_zlim(gt_mid[1] - gt_span/2, gt_mid[1] + gt_span/2)
        ax_gt.set_title(f"Habitat Ground Truth 3D ({action.upper()})", fontsize=10, fontweight="bold")
        ax_gt.set_xlabel("X (m)")
        ax_gt.set_ylabel("Z (m)")
        ax_gt.set_zlabel("Y (m)")
        ax_gt.view_init(elev=0, azim=-90)
        plt.tight_layout()
        plt.savefig(sample_dir / "gt_skeleton_3d.png", dpi=130)
        plt.close(fig_gt)

        # 7. 保存 overlay_comparison.png
        plot_skeleton_comparison(
            sample_id=sid,
            rgb_img=rgb_img,
            estimated_skeleton=skel,
            gt_joints_dict=gt_joints_dict,
            camera_matrix_4x4=c2w,
            metrics=metrics,
            output_path=sample_dir / "overlay_comparison.png",
            skel_def=skel_def,
        )

        # 8. 保存 metrics.json
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # 9. 保存 metadata.json (根据规范增加完整元数据)
        metadata_payload = {
            "sample_id": sid,
            "action_label": action,
            "view_id": s_dict.get("view_id", "vp_000"),
            "backend": "mediapipe_33",
            "joint_definition": "configs/skeleton_definition.json",
            "coordinate_system": "camera_frame_right_hand",
            "unit": "meter",
            "has_gt": True,
            "evaluation_available": True,
            "metrics_summary": {
                "mpjpe_mm": metrics.mpjpe_mm,
                "pa_mpjpe_mm": metrics.pa_mpjpe_mm,
                "pck_10cm": metrics.pck_10cm,
            },
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        eval_batch_data.append({
            "estimated_skeleton": skel,
            "gt_joints_dict": gt_joints_dict,
            "camera_matrix_4x4": c2w,
            "sample_id": sid,
            "action_label": action,
        })

    # 生成批量汇总指标
    summary = evaluator.evaluate_batch(eval_batch_data)
    summary_file = val_dir / "validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(">>> Validation Completed.")
    logger.info("  [Primary] Absolute MPJPE: %.2f mm | Relative MPJPE: %.2f mm",
                summary["primary_results"]["mean_abs_mpjpe_mm"], summary["primary_results"]["mean_mpjpe_mm"])
    logger.info("  [Secondary] PCK@5cm: %.1f%% | PCK@10cm: %.1f%% | PCK@15cm: %.1f%%",
                summary["secondary_results"]["mean_pck_5cm"] * 100,
                summary["secondary_results"]["mean_pck_10cm"] * 100,
                summary["secondary_results"]["mean_pck_15cm"] * 100)
    logger.info("  [Supplementary] PA-MPJPE: %.2f mm", summary["supplementary_results"]["mean_pa_mpjpe_mm"])
    logger.info("Saved summary report to: %s", summary_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v10.0 Phase 2.1 Validation Runner")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to validate")
    args = parser.parse_args()

    run_phase2_1_validation(args.num_samples)


if __name__ == "__main__":
    main()
