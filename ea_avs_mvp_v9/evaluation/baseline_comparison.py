"""
多基线综合对比评测器 —— baseline_comparison.py
=============================================

职责：
    1. 在相同候选视点池与动作条件下，执行四大基线策略横向对比：
       - random (随机可行视点)
       - nearest (最近可行视点)
       - geometry_best (v8 几何最优)
       - action_conditioned (v9 动作感知最优)
    2. 计算各基线在 Q(v|a)、Q_geom、Critical Region Coverage、Distance 等维度的定量差异；
    3. 输出标准化 comparison_report.json 数据。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality
from ea_avs_mvp_v9.core.types import ActionEmbedding, ActionViewpointScore, ViewFeature
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector
from .action_metrics import compute_action_observation_metrics

logger = logging.getLogger(__name__)


def compare_all_baselines(
    viewpoints: List[CandidateViewpoint],
    action_scores: List[ActionViewpointScore],
    features: List[ViewFeature],
    geometry_qualities: List[ViewpointQuality],
    action: ActionEmbedding,
    human_position: Optional[Union[List[float], np.ndarray]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """执行全部 4 种基线策略的对比评测。"""
    feat_map = {f.viewpoint_id: f for f in features}
    score_map = {s.viewpoint_id: s for s in action_scores}

    strategies = ["random", "nearest", "geometry_best", "action_conditioned"]
    results_by_strategy: Dict[str, Any] = {}

    for strat in strategies:
        sel_vp, sel_score = ViewpointSelector.select(
            viewpoints=viewpoints,
            action_scores=action_scores,
            geometry_qualities=geometry_qualities,
            strategy=strat,
            human_position=human_position,
            seed=seed,
        )

        feat = feat_map[sel_vp.viewpoint_id]
        score_obj = sel_score if sel_score is not None else score_map[sel_vp.viewpoint_id]
        metrics = compute_action_observation_metrics(feat, score_obj, action)

        results_by_strategy[strat] = {
            "selected_view_id": sel_vp.viewpoint_id,
            "position": [float(x) for x in sel_vp.position],
            "yaw_deg": float(sel_vp.yaw_deg),
            "distance": feat.distance,
            "viewing_angle_deg": feat.viewing_angle_deg,
            "action_total_score": score_obj.total_score,
            "geometry_score": score_obj.geometry_score,
            "action_delta": score_obj.action_delta,
            "pose_coverage": feat.pose_coverage,
            "critical_region_coverage": metrics["critical_region_coverage"],
            "aspect_alignment": metrics["aspect_alignment"],
        }

    # 计算动作感知相比 v8 纯几何基线的增益
    geom_score = results_by_strategy["geometry_best"]["action_total_score"]
    act_score = results_by_strategy["action_conditioned"]["action_total_score"]
    score_gain = round(act_score - geom_score, 3)

    return {
        "action_name": action.action_name,
        "action_class": action.action_class.value,
        "critical_regions": action.critical_regions,
        "total_feasible_candidates": sum(1 for v in viewpoints if v.feasible),
        "strategies": results_by_strategy,
        "action_conditioned_gain_over_v8": score_gain,
        "preferred_viewpoint_shifted": bool(
            results_by_strategy["geometry_best"]["selected_view_id"]
            != results_by_strategy["action_conditioned"]["selected_view_id"]
        ),
    }
