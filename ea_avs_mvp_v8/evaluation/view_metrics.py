"""
视点质量评价指标与汇总统计 —— view_metrics.py
=============================================

职责：
    1. 汇总视点可见性、平均距离、姿态覆盖率与有效视点比例；
    2. 生成标准格式的 visibility.json。
"""

from typing import Any, Dict, List
import numpy as np

from ea_avs_mvp_v8.core.types import ViewpointQuality


def summarize_viewpoint_qualities(
    qualities: List[ViewpointQuality],
) -> Dict[str, Any]:
    """汇总一组候选视点的统计特征。"""
    if not qualities:
        return {
            "total_viewpoints": 0,
            "valid_viewpoints": 0,
            "mean_visibility_score": 0.0,
            "mean_distance": 0.0,
            "mean_pose_coverage": 0.0,
        }

    scores = [q.visibility_score for q in qualities]
    dists = [q.distance for q in qualities]
    covs = [q.pose_coverage for q in qualities]
    valids = [1 for q in qualities if q.is_valid]

    return {
        "total_viewpoints": len(qualities),
        "valid_viewpoints": sum(valids),
        "valid_ratio": round(float(sum(valids) / len(qualities)), 3),
        "mean_visibility_score": round(float(np.mean(scores)), 3),
        "max_visibility_score": round(float(np.max(scores)), 3),
        "min_visibility_score": round(float(np.min(scores)), 3),
        "mean_distance": round(float(np.mean(dists)), 3),
        "mean_pose_coverage": round(float(np.mean(covs)), 3),
    }
