"""
离线 GT 人体 3D 骨架评测器 —— skeleton_evaluator.py
===================================================

职责：
    1. 评估估计 3D 骨架与 Habitat 仿真真值 (Ground Truth) 之间的空间重构误差；
    2. 计算标准三维人体姿态指标（严格分层优先级）：
       - [主要指标 Primary Metrics]:
         * Absolute MPJPE (绝对空间关节误差, 单位: 米 / 毫米)
         * MPJPE (Root-Relative MPJPE, 根节点对齐关节相对误差, 单位: 米 / 毫米)
       - [辅助指标 Secondary Metrics]:
         * PCK@threshold (Percentage of Correct Keypoints @ 5cm, 10cm, 15cm)
       - [补充指标 Supplementary Metrics]:
         * PA-MPJPE (Procrustes-Aligned MPJPE, 刚体旋转与尺度对齐后净结构误差)
    3. 科学表述原则：
       "3D reconstruction is geometrically consistent with observed 2D keypoints."
       (重建的 3D 骨架保证与二维视觉观测的几何一致性，但不等同于真实人体 3D 完全绝对准确)
    4. 严格隔离：本模块仅用于离线科研评测与上界对比，禁止任何在线前向计算调用。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v10.evaluation.skeleton_alignment import (
    compute_procrustes_alignment,
    extract_aligned_joint_pairs,
    transform_gt_to_camera_frame,
)
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    sample_id: str
    action_label: str
    num_evaluated_joints: int
    # 主要指标 (Primary Metrics)
    abs_mpjpe_meters: float             # 绝对相机空间 MPJPE (米)
    abs_mpjpe_mm: float                 # 绝对相机空间 MPJPE (毫米)
    mpjpe_meters: float                 # 根节点相对 MPJPE (米)
    mpjpe_mm: float                     # 根节点相对 MPJPE (毫米)
    # 辅助指标 (Secondary Metrics)
    pck_5cm: float                      # PCK@0.05m (5cm / 50mm)
    pck_10cm: float                     # PCK@0.10m (10cm / 100mm)
    pck_15cm: float                     # PCK@0.15m (15cm / 150mm)
    # 补充指标 (Supplementary Metrics)
    pa_mpjpe_meters: float              # Procrustes 对齐 MPJPE (米)
    pa_mpjpe_mm: float                  # Procrustes 对齐 MPJPE (毫米)
    # 细粒度分项
    per_joint_errors_m: Dict[str, float] = field(default_factory=dict)
    per_part_errors_m: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "action_label": self.action_label,
            "num_evaluated_joints": self.num_evaluated_joints,
            "primary_metrics": {
                "abs_mpjpe_meters": round(self.abs_mpjpe_meters, 4),
                "abs_mpjpe_mm": round(self.abs_mpjpe_mm, 2),
                "mpjpe_meters": round(self.mpjpe_meters, 4),
                "mpjpe_mm": round(self.mpjpe_mm, 2),
            },
            "secondary_metrics": {
                "pck_5cm": round(self.pck_5cm, 4),
                "pck_10cm": round(self.pck_10cm, 4),
                "pck_15cm": round(self.pck_15cm, 4),
            },
            "supplementary_metrics": {
                "pa_mpjpe_meters": round(self.pa_mpjpe_meters, 4),
                "pa_mpjpe_mm": round(self.pa_mpjpe_mm, 2),
            },
            # 扁平化兼容字段
            "abs_mpjpe_meters": round(self.abs_mpjpe_meters, 4),
            "abs_mpjpe_mm": round(self.abs_mpjpe_mm, 2),
            "mpjpe_meters": round(self.mpjpe_meters, 4),
            "mpjpe_mm": round(self.mpjpe_mm, 2),
            "pck_5cm": round(self.pck_5cm, 4),
            "pck_10cm": round(self.pck_10cm, 4),
            "pck_15cm": round(self.pck_15cm, 4),
            "pa_mpjpe_meters": round(self.pa_mpjpe_meters, 4),
            "pa_mpjpe_mm": round(self.pa_mpjpe_mm, 2),
            "per_joint_errors_m": {k: round(v, 4) for k, v in self.per_joint_errors_m.items()},
            "per_part_errors_m": {k: round(v, 4) for k, v in self.per_part_errors_m.items()},
        }


class SkeletonEvaluator:
    """标准 3D 骨架重建精度评测器。"""

    def __init__(self, skel_def: Optional[SkeletonDefinition] = None):
        self.skel_def = skel_def or get_skeleton_definition()

    def evaluate_sample(
        self,
        estimated_skeleton: EstimatedSkeleton3D,
        gt_joints_dict: Dict[str, List[float]],
        camera_matrix_4x4: np.ndarray,
        sample_id: str = "sample",
        action_label: str = "unknown",
    ) -> EvaluationMetrics:
        """
        评估单个样本估计骨架与 GT 骨架之间的误差指标。
        """
        P_est, P_gt, joint_names = extract_aligned_joint_pairs(
            estimated_skeleton=estimated_skeleton,
            gt_joints_dict=gt_joints_dict,
            camera_matrix_4x4=camera_matrix_4x4,
            skel_def=self.skel_def,
        )

        assert len(joint_names) > 0, "No matching joints found between estimated skeleton and GT skeleton"

        # 1. 绝对距离误差 (Absolute MPJPE - Primary Metric)
        abs_errors = np.linalg.norm(P_est - P_gt, axis=1)
        abs_mpjpe_m = float(np.mean(abs_errors))
        abs_mpjpe_mm = abs_mpjpe_m * 1000.0

        # 2. 根节点相对误差 (Root-Relative MPJPE - Primary Metric)
        # 寻找骨盆根节点索引
        if "pelvis" in joint_names:
            root_idx = joint_names.index("pelvis")
            p_est_root = P_est[root_idx]
            p_gt_root = P_gt[root_idx]
        else:
            p_est_root = np.mean(P_est, axis=0)
            p_gt_root = np.mean(P_gt, axis=0)

        P_est_rel = P_est - p_est_root
        P_gt_rel = P_gt - p_gt_root

        rel_errors = np.linalg.norm(P_est_rel - P_gt_rel, axis=1)
        mpjpe_m = float(np.mean(rel_errors))
        mpjpe_mm = mpjpe_m * 1000.0

        # 3. 辅助指标 (PCK@thresholds - Secondary Metric)
        pck_5 = float(np.mean(rel_errors <= 0.05))
        pck_10 = float(np.mean(rel_errors <= 0.10))
        pck_15 = float(np.mean(rel_errors <= 0.15))

        # 4. Procrustes 对齐误差 (PA-MPJPE - Supplementary Metric)
        P_aligned, R, s, t = compute_procrustes_alignment(P_est, P_gt)
        pa_errors = np.linalg.norm(P_aligned - P_gt, axis=1)
        pa_mpjpe_m = float(np.mean(pa_errors))
        pa_mpjpe_mm = pa_mpjpe_m * 1000.0

        # 5. 逐关节误差与分部位误差
        per_joint: Dict[str, float] = {}
        for idx, name in enumerate(joint_names):
            per_joint[name] = float(rel_errors[idx])

        part_mapping = {
            "head": ["head"],
            "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "pelvis"],
            "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            "legs": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
        }
        per_part: Dict[str, float] = {}
        for part_name, sub_joints in part_mapping.items():
            valid_errs = [per_joint[j] for j in sub_joints if j in per_joint]
            if len(valid_errs) > 0:
                per_part[part_name] = float(np.mean(valid_errs))

        return EvaluationMetrics(
            sample_id=sample_id,
            action_label=action_label,
            num_evaluated_joints=len(joint_names),
            abs_mpjpe_meters=abs_mpjpe_m,
            abs_mpjpe_mm=abs_mpjpe_mm,
            mpjpe_meters=mpjpe_m,
            mpjpe_mm=mpjpe_mm,
            pck_5cm=pck_5,
            pck_10cm=pck_10,
            pck_15cm=pck_15,
            pa_mpjpe_meters=pa_mpjpe_m,
            pa_mpjpe_mm=pa_mpjpe_mm,
            per_joint_errors_m=per_joint,
            per_part_errors_m=per_part,
        )

    def evaluate_batch(
        self,
        samples_eval_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量评估多个样本并生成汇总统计。
        """
        metrics_list: List[EvaluationMetrics] = []
        for item in samples_eval_data:
            m = self.evaluate_sample(
                estimated_skeleton=item["estimated_skeleton"],
                gt_joints_dict=item["gt_joints_dict"],
                camera_matrix_4x4=item["camera_matrix_4x4"],
                sample_id=item.get("sample_id", "sample"),
                action_label=item.get("action_label", "unknown"),
            )
            metrics_list.append(m)

        all_abs_mpjpe = [m.abs_mpjpe_meters for m in metrics_list]
        all_mpjpe = [m.mpjpe_meters for m in metrics_list]
        all_pa_mpjpe = [m.pa_mpjpe_meters for m in metrics_list]
        all_pck5 = [m.pck_5cm for m in metrics_list]
        all_pck10 = [m.pck_10cm for m in metrics_list]
        all_pck15 = [m.pck_15cm for m in metrics_list]

        summary = {
            "evaluation_principle": "3D reconstruction is geometrically consistent with observed 2D keypoints",
            "metric_hierarchy": {
                "primary_metrics": ["Absolute MPJPE", "MPJPE (Root-relative)"],
                "secondary_metrics": ["PCK@5cm", "PCK@10cm", "PCK@15cm"],
                "supplementary_metrics": ["PA-MPJPE (Procrustes-Aligned)"],
                "rationale": "ACTIVEVIEW targets robot active perception where absolute camera-frame spatial position and scale are vital and cannot be arbitrarily eliminated.",
            },
            "num_evaluated_samples": len(metrics_list),
            "primary_results": {
                "mean_abs_mpjpe_meters": round(float(np.mean(all_abs_mpjpe)), 4),
                "mean_abs_mpjpe_mm": round(float(np.mean(all_abs_mpjpe) * 1000.0), 2),
                "mean_mpjpe_meters": round(float(np.mean(all_mpjpe)), 4),
                "mean_mpjpe_mm": round(float(np.mean(all_mpjpe) * 1000.0), 2),
                "median_mpjpe_mm": round(float(np.median(all_mpjpe) * 1000.0), 2),
            },
            "secondary_results": {
                "mean_pck_5cm": round(float(np.mean(all_pck5)), 4),
                "mean_pck_10cm": round(float(np.mean(all_pck10)), 4),
                "mean_pck_15cm": round(float(np.mean(all_pck15)), 4),
            },
            "supplementary_results": {
                "mean_pa_mpjpe_meters": round(float(np.mean(all_pa_mpjpe)), 4),
                "mean_pa_mpjpe_mm": round(float(np.mean(all_pa_mpjpe) * 1000.0), 2),
            },
            "samples": [m.to_dict() for m in metrics_list],
        }
        return summary
