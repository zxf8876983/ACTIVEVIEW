"""
动作感知视点综合打分器 —— action_scorer.py
=========================================

职责：
    1. 结合几何视点特征 (ViewFeature) 与动作先验表示 (ActionEmbedding)；
    2. 计算多维动作感知观测收益 Delta_Q(a, v):
       - 动作关键身体部位匹配收益 (Region Matching)
       - 动作推荐观察朝向偏好收益 (Aspect Alignment)
       - 动作推荐观测距离适配收益 (Distance Affinity)
    3. 输出标准化综合评分 Q(v|a) = w_geom * Q_geom(v) + w_act * Delta_Q(a, v)。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ea_avs_mvp_v9.core.types import ActionEmbedding, ActionViewpointScore, ViewFeature

logger = logging.getLogger(__name__)


class ActionConditionedScorer:
    """动作条件化视点质量打分器。"""

    def __init__(self, scoring_config: Optional[Dict[str, Any]] = None):
        self.config = scoring_config or {}
        self.w_geom = float(self.config.get("w_geometry", 0.60))
        self.w_action = float(self.config.get("w_action", 0.40))
        self.evaluation_mode = str(self.config.get("evaluation_mode", "oracle"))
        self.pose_source = str(self.config.get("pose_source", "oracle"))

    def score_single(
        self,
        feature: ViewFeature,
        action: ActionEmbedding,
        geometry_score: float,
    ) -> ActionViewpointScore:
        """评估单个视点在指定动作条件下的观测质量。"""
        if not feature.feasible:
            return ActionViewpointScore(
                viewpoint_id=feature.viewpoint_id,
                action_name=action.action_name,
                geometry_score=0.0,
                action_delta=0.0,
                total_score=0.0,
                region_score=0.0,
                aspect_score=0.0,
                distance_score=0.0,
                evaluation_mode=self.evaluation_mode,
                pose_source=self.pose_source,
                metadata={"reason": "infeasible_candidate"},
            )

        # 1. 计算关键身体区域匹配得分 [0.0, 1.0]
        region_score = 0.0
        total_reg_w = sum(action.region_weights.values()) if action.region_weights else 1.0
        for reg_name, reg_cov in feature.region_coverages.items():
            w = action.region_weights.get(reg_name, 0.0)
            region_score += w * reg_cov
        region_score = float(region_score / max(1e-4, total_reg_w))

        # 2. 计算视角偏向与动作朝向偏好吻合度 [0.0, 1.0]
        min_a, max_a = action.preferred_angle_range
        angle = feature.viewing_angle_deg
        if min_a <= angle <= max_a:
            aspect_score = 1.0
        elif angle < min_a:
            aspect_score = max(0.0, math.cos(math.radians(min_a - angle)))
        else:
            aspect_score = max(0.0, math.cos(math.radians(angle - max_a)))

        # 3. 计算动作专属距离适配度 [0.0, 1.0]
        opt_dist = action.optimal_distance
        dist_diff = abs(feature.distance - opt_dist)
        dist_score = max(0.0, 1.0 - (dist_diff / max(1e-4, opt_dist)))

        # 4. 计算动作增益 Delta_Q(a, v)
        action_delta = (
            action.aspect_weight * aspect_score
            + (1.0 - action.aspect_weight - action.distance_weight) * region_score
            + action.distance_weight * dist_score
        )
        action_delta = float(np.clip(action_delta, 0.0, 1.0))

        # 5. 综合总评分 Q(v|a)
        total_score = self.w_geom * geometry_score + self.w_action * action_delta
        total_score = float(np.clip(total_score, 0.0, 1.0))

        return ActionViewpointScore(
            viewpoint_id=feature.viewpoint_id,
            action_name=action.action_name,
            geometry_score=round(geometry_score, 3),
            action_delta=round(action_delta, 3),
            total_score=round(total_score, 3),
            region_score=round(region_score, 3),
            aspect_score=round(aspect_score, 3),
            distance_score=round(dist_score, 3),
            evaluation_mode=self.evaluation_mode,
            pose_source=self.pose_source,
            metadata={
                "w_geom": self.w_geom,
                "w_action": self.w_action,
                "action_class": action.action_class.value,
            },
        )

    def score_batch(
        self,
        features: List[ViewFeature],
        action: ActionEmbedding,
        geometry_scores: Dict[str, float],
    ) -> List[ActionViewpointScore]:
        """批量评估候选视点打分。"""
        scores = []
        for feat in features:
            g_score = geometry_scores.get(feat.viewpoint_id, 0.0)
            scores.append(self.score_single(feat, action, g_score))
        return scores
