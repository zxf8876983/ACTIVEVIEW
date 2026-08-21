"""
人体物理状态感知视点质量评价器 —— human_state_scorer.py
======================================================

职责：
    1. 基于人体物理状态 (16 关键点解剖姿态) 与视点描述子计算科学综合效用 Q*(v | H)；
    2. 计算公式：
       Q*(v | H) = w1 * global_visibility + w2 * pose_coverage + w3 * body_part_visibility - w4 * distance_penalty
    3. 支持提取 7 大关键部位可见性贡献。
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v9.core.types import ViewFeature

logger = logging.getLogger(__name__)


class HumanStateAwareViewScorer:
    """人体物理状态驱动的视点质量综合评价器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.w1 = float(self.config.get("w_global_visibility", self.config.get("w1", 0.35)))
        self.w2 = float(self.config.get("w_pose_coverage", self.config.get("w2", 0.35)))
        self.w3 = float(self.config.get("w_body_part_visibility", self.config.get("w3", 0.20)))
        self.w4 = float(self.config.get("w_distance_penalty", self.config.get("w4", 0.10)))
        self.optimal_distance = float(self.config.get("optimal_distance", 2.0))
        self.max_distance = float(self.config.get("max_distance", 4.5))

    def score(
        self,
        feature: ViewFeature,
        geom_visibility: float = 1.0,
    ) -> Dict[str, Any]:
        """计算单个候选视点在当前人体姿态下的质量效用。"""
        if not feature.feasible:
            return {
                "viewpoint_id": feature.viewpoint_id,
                "total_score": 0.0,
                "global_visibility": 0.0,
                "pose_coverage": 0.0,
                "body_part_visibility": 0.0,
                "distance_penalty": 1.0,
                "feasible": False,
            }

        glob_vis = float(geom_visibility)
        pose_cov = float(feature.pose_coverage)

        # 7 大身体关键解剖部位平均可见性
        parts = feature.body_part_visibilities
        if parts:
            part_vis = float(np.mean(list(parts.values())))
        else:
            part_vis = pose_cov

        # 距离偏离惩罚
        dist_err = abs(feature.distance - self.optimal_distance)
        dist_pen = min(1.0, dist_err / 2.0)

        total_q = (
            self.w1 * glob_vis
            + self.w2 * pose_cov
            + self.w3 * part_vis
            - self.w4 * dist_pen
        )
        total_q = float(np.clip(total_q, 0.0, 1.0))

        return {
            "viewpoint_id": feature.viewpoint_id,
            "total_score": round(total_q, 3),
            "global_visibility": round(glob_vis, 3),
            "pose_coverage": round(pose_cov, 3),
            "body_part_visibility": round(part_vis, 3),
            "distance_penalty": round(dist_pen, 3),
            "feasible": True,
        }

    def score_batch(
        self,
        features: List[ViewFeature],
        geom_visibility_map: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """批量计算所有候选视点的效用打分。"""
        geom_map = geom_visibility_map or {}
        return [
            self.score(f, geom_visibility=geom_map.get(f.viewpoint_id, 0.8))
            for f in features
        ]
