"""
Evaluation module for v10.0.
Provides offline Ground Truth comparison, MPJPE/PA-MPJPE calculation, and PCK metrics.
GT data is strictly reserved for evaluation/benchmarking and never enters online models.
"""

from .skeleton_alignment import (
    compute_procrustes_alignment,
    extract_aligned_joint_pairs,
    transform_gt_to_camera_frame,
)
from .skeleton_evaluator import EvaluationMetrics, SkeletonEvaluator

__all__ = [
    "SkeletonEvaluator",
    "EvaluationMetrics",
    "transform_gt_to_camera_frame",
    "extract_aligned_joint_pairs",
    "compute_procrustes_alignment",
]
