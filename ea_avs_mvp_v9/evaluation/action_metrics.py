"""
动作相关观测质量指标计算 —— action_metrics.py
=========================================
"""

from typing import Any, Dict, List
import numpy as np

from ea_avs_mvp_v9.core.types import ActionEmbedding, ActionViewpointScore, ViewFeature


def compute_action_observation_metrics(
    selected_feature: ViewFeature,
    selected_score: ActionViewpointScore,
    action: ActionEmbedding,
) -> Dict[str, Any]:
    """计算动作任务相关的核心评测指标。"""
    # 1. 关键区域覆盖率
    crit_covs = [selected_feature.region_coverages.get(r, 0.0) for r in action.critical_regions]
    critical_region_coverage = float(np.mean(crit_covs)) if crit_covs else selected_feature.pose_coverage

    # 2. 视角偏向与偏好区间重合度
    min_a, max_a = action.preferred_angle_range
    angle = selected_feature.viewing_angle_deg
    aspect_alignment = bool(min_a <= angle <= max_a)

    return {
        "action_name": action.action_name,
        "action_class": action.action_class.value,
        "critical_region_coverage": round(critical_region_coverage, 3),
        "aspect_alignment": aspect_alignment,
        "distance_to_optimal": round(abs(selected_feature.distance - action.optimal_distance), 3),
        "total_score": selected_score.total_score,
        "geometry_score": selected_score.geometry_score,
        "action_delta": selected_score.action_delta,
        "pose_coverage": selected_feature.pose_coverage,
    }
