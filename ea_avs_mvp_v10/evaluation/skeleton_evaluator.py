"""
离线 GT 人体 3D 骨架评测器 —— skeleton_evaluator.py
===================================================

职责：
    1. 评估估计 3D 骨架与 Habitat 仿真真值 (Ground Truth) 之间的空间重构误差；
    2. 计算标准三维人体姿态指标：
       - MPJPE (Mean Per-Joint Position Error, 根节点相对与绝对误差, 单位: 米 / 毫米)
       - PA-MPJPE (Procrustes-Aligned MPJPE, 刚体旋转尺度对齐后误差)
       - PCK@threshold (Percentage of Correct Keypoints @ 0.05m, 0.10m, 0.15m)
    3. 输出细粒度分身体部位与逐关节误差统计报表；
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
    mpjpe_meters: float
    mpjpe_mm: float
    pa_mpjpe_meters: float
    pa_mpjpe_mm: float
    abs_mpjpe_meters: float
    pck_05cm: float                     # PCK@0.05m (50mm)
    pck_10cm: float                     # PCK@0.10m (100mm)
    pck_15cm: float                     # PCK@0.15m (150mm)
    per_joint_errors_m: Dict[str, float] = field(default_factory=dict)
    per_part_errors_m: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "action_label": self.action_label,
            "num_evaluated_joints": self.num_evaluated_joints,
            "mpjpe_meters": round(self.mpjpe_meters, 4),
            "mpjpe_mm": round(self.mpjpe_mm, 2),
            "pa_mpjpe_meters": round(self.pa_mpjpe_meters, 4),
            "pa_mpjpe_mm": round(self.pa_mpjpe_mm, 2),
            "abs_mpjpe_meters": round(self.abs_mpjpe_meters, 4),
            "pck_05cm": round(self.pck_05cm, 4),
            "pck_10cm": round(self.pck_10cm, 4),
            "pck_15cm": round(self.pck_15cm, 4),
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

        # 1. 绝对距离误差 (Absolute MPJPE)
        abs_errors = np.linalg.norm(P_est - P_gt, axis=1)
        abs_mpjpe = float(np.mean(abs_errors))

        # 2. 根节点相对误差 (Root-Relative MPJPE)
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

        # 3. Procrustes 对齐误差 (PA-MPJPE)
        P_aligned, R, s, t = compute_procrustes_alignment(P_est, P_gt)
        pa_errors = np.linalg.norm(P_aligned - P_gt, axis=1)
        pa_mpjpe_m = float(np.mean(pa_errors))
        pa_mpjpe_mm = pa_mpjpe_m * 1000.0

        # 4. PCK@thresholds
        pck_05 = float(np.mean(rel_errors <= 0.05))
        pck_10 = float(np.mean(rel_errors <= 0.10))
        pck_15 = float(np.mean(rel_errors <= 0.15))

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
            mpjpe_meters=mpjpe_m,
            mpjpe_mm=mpjpe_mm,
            pa_mpjpe_meters=pa_mpjpe_m,
            pa_mpjpe_mm=pa_mpjpe_mm,
            abs_mpjpe_meters=abs_mpjpe,
            pck_05cm=pck_05,
            pck_10cm=pck_10,
            pck_15cm=pck_15,
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

        all_mpjpe = [m.mpjpe_meters for m in metrics_list]
        all_pa_mpjpe = [m.pa_mpjpe_meters for m in metrics_list]
        all_pck05 = [m.pck_05cm for m in metrics_list]
        all_pck10 = [m.pck_10cm for m in metrics_list]
        all_pck15 = [m.pck_15cm for m in metrics_list]

        summary = {
            "num_evaluated_samples": len(metrics_list),
            "mean_mpjpe_meters": round(float(np.mean(all_mpjpe)), 4),
            "mean_mpjpe_mm": round(float(np.mean(all_mpjpe) * 1000.0), 2),
            "median_mpjpe_mm": round(float(np.median(all_mpjpe) * 1000.0), 2),
            "mean_pa_mpjpe_mm": round(float(np.mean(all_pa_mpjpe) * 1000.0), 2),
            "mean_pck_05cm": round(float(np.mean(all_pck05)), 4),
            "mean_pck_10cm": round(float(np.mean(all_pck10)), 4),
            "mean_pck_15cm": round(float(np.mean(all_pck15)), 4),
            "samples": [m.to_dict() for m in metrics_list],
        }
        return summary
