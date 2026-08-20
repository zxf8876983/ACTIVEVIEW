"""
视点选择 Baseline 实验策略接口 —— baseline_strategies.py
======================================================

职责：
    1. 提供标准化的主动视角选择基线策略 (Baseline View Selection Strategies)：
       - Random View: 在所有空间与几何可行的候选视点中随机采样；
       - Nearest View: 选择距离目标人体最近的有效候选视点；
       - Geometry Best View: 基于 v8 多维质量评价评分 Q(v) 选择最优视角；
    2. 为后续 v9 Action-aware Active View Selection 与动作感知算法提供标准对比基准接口。
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality

logger = logging.getLogger(__name__)


def select_random_view(
    viewpoints: List[CandidateViewpoint],
    qualities: Optional[List[ViewpointQuality]] = None,
    seed: Optional[int] = None,
) -> Tuple[CandidateViewpoint, Optional[ViewpointQuality]]:
    """策略 1: 随机选择一个合法的候选视点 (Random View)。"""
    feasible_vps = [vp for vp in viewpoints if vp.feasible]
    if not feasible_vps:
        feasible_vps = viewpoints

    rng = random.Random(seed)
    selected_vp = rng.choice(feasible_vps)

    selected_q = None
    if qualities:
        for q in qualities:
            if q.viewpoint_id == selected_vp.viewpoint_id:
                selected_q = q
                break

    return selected_vp, selected_q


def select_nearest_view(
    viewpoints: List[CandidateViewpoint],
    qualities: Optional[List[ViewpointQuality]] = None,
    human_position: Optional[Union[List[float], np.ndarray]] = None,
) -> Tuple[CandidateViewpoint, Optional[ViewpointQuality]]:
    """策略 2: 选择距离人体最近的合法候选视点 (Nearest View)。"""
    feasible_vps = [vp for vp in viewpoints if vp.feasible]
    if not feasible_vps:
        feasible_vps = viewpoints

    if human_position is not None:
        hx, hy, hz = [float(x) for x in human_position]
        h_pos = np.array([hx, hz], dtype=np.float32)

        def dist_fn(vp: CandidateViewpoint) -> float:
            return float(np.linalg.norm(np.array([vp.position[0], vp.position[2]]) - h_pos))

        selected_vp = min(feasible_vps, key=dist_fn)
    else:
        selected_vp = min(feasible_vps, key=lambda v: v.radius)

    selected_q = None
    if qualities:
        for q in qualities:
            if q.viewpoint_id == selected_vp.viewpoint_id:
                selected_q = q
                break

    return selected_vp, selected_q


def select_geometry_best_view(
    viewpoints: List[CandidateViewpoint],
    qualities: List[ViewpointQuality],
) -> Tuple[CandidateViewpoint, ViewpointQuality]:
    """策略 3: 基于几何与遮挡综合质量评分 Q(v) 选择最优视点 (Geometry Best View)。"""
    valid_pairs = []
    for vp in viewpoints:
        for q in qualities:
            if vp.viewpoint_id == q.viewpoint_id:
                valid_pairs.append((vp, q))
                break

    feasible_pairs = [p for p in valid_pairs if p[0].feasible]
    if not feasible_pairs:
        feasible_pairs = valid_pairs

    # 按 visibility_score (Q(v)) 降序，其次按 viewing_angle_deg 升序 (正面优先)
    sorted_pairs = sorted(
        feasible_pairs,
        key=lambda item: (item[1].is_valid, item[1].visibility_score, -item[1].viewing_angle_deg),
        reverse=True,
    )
    return sorted_pairs[0]


def select_view(
    viewpoints: List[CandidateViewpoint],
    qualities: List[ViewpointQuality],
    strategy: str = "geometry_best",
    human_position: Optional[Union[List[float], np.ndarray]] = None,
    seed: Optional[int] = None,
) -> Tuple[CandidateViewpoint, ViewpointQuality]:
    """统一视点选择策略调度接口。

    参数：
        viewpoints: 候选视点列表
        qualities: 对应质量评估列表
        strategy: 策略名称 ("geometry_best", "nearest", "random")
        human_position: 人体世界坐标 [hx, hy, hz]
        seed: 随机种子 (仅对 random 策略生效)

    返回：
        (selected_viewpoint, selected_quality)
    """
    strat_lower = strategy.lower().strip()
    if strat_lower == "random":
        vp, q = select_random_view(viewpoints, qualities, seed=seed)
        if q is None and qualities:
            q = qualities[0]
        return vp, q
    elif strat_lower == "nearest":
        vp, q = select_nearest_view(viewpoints, qualities, human_position=human_position)
        if q is None and qualities:
            q = qualities[0]
        return vp, q
    elif strat_lower in ["geometry_best", "best", "ours_geometry"]:
        return select_geometry_best_view(viewpoints, qualities)
    else:
        logger.warning("Unknown strategy '%s', defaulting to 'geometry_best'", strategy)
        return select_geometry_best_view(viewpoints, qualities)
