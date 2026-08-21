"""
动作感知视点选择器与基线调度器 —— viewpoint_selector.py
======================================================

职责：
    1. 提供四种统一的标准视点选择策略：
       - random: 随机选择一个可行视点
       - nearest: 选择距离人体最近的可行视点
       - geometry_best: 基于 v8 几何质量评价 Q_geom(v) 选择最优视点
       - action_conditioned: 基于 v9 动作条件评分 Q(v|a) 选择最优视点
    2. 输出统一的选定视点对象、得分与评测指标。
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality
from ea_avs_mvp_v9.core.types import ActionViewpointScore

logger = logging.getLogger(__name__)


class ViewpointSelector:
    """视角选择策略调度器。"""

    @staticmethod
    def select_random(
        viewpoints: List[CandidateViewpoint],
        action_scores: Optional[List[ActionViewpointScore]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[CandidateViewpoint, Optional[ActionViewpointScore]]:
        """策略 1: 随机选择一个合法视点。"""
        feas = [v for v in viewpoints if v.feasible]
        pool = feas if feas else viewpoints
        rng = random.Random(seed)
        sel_vp = rng.choice(pool)

        sel_score = None
        if action_scores:
            for s in action_scores:
                if s.viewpoint_id == sel_vp.viewpoint_id:
                    sel_score = s
                    break
        return sel_vp, sel_score

    @staticmethod
    def select_nearest(
        viewpoints: List[CandidateViewpoint],
        action_scores: Optional[List[ActionViewpointScore]] = None,
        human_position: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Tuple[CandidateViewpoint, Optional[ActionViewpointScore]]:
        """策略 2: 选择距离人体最近的合法视点。"""
        feas = [v for v in viewpoints if v.feasible]
        pool = feas if feas else viewpoints

        if human_position is not None:
            hx, hy, hz = [float(x) for x in human_position]
            h_pos = np.array([hx, hz], dtype=np.float32)

            def dist_fn(v: CandidateViewpoint) -> float:
                return float(np.linalg.norm(np.array([v.position[0], v.position[2]]) - h_pos))

            sel_vp = min(pool, key=dist_fn)
        else:
            sel_vp = min(pool, key=lambda v: v.radius)

        sel_score = None
        if action_scores:
            for s in action_scores:
                if s.viewpoint_id == sel_vp.viewpoint_id:
                    sel_score = s
                    break
        return sel_vp, sel_score

    @staticmethod
    def select_geometry_best(
        viewpoints: List[CandidateViewpoint],
        geometry_qualities: List[ViewpointQuality],
        action_scores: Optional[List[ActionViewpointScore]] = None,
    ) -> Tuple[CandidateViewpoint, Optional[ActionViewpointScore]]:
        """策略 3: 基于 v8 几何综合评分 Q_geom(v) 选择最优视点。"""
        vp_map = {v.viewpoint_id: v for v in viewpoints}
        feas_qs = [q for q in geometry_qualities if vp_map.get(q.viewpoint_id, CandidateViewpoint("tmp", [], 0, 0, 0)).feasible]
        pool = feas_qs if feas_qs else geometry_qualities

        best_q = max(pool, key=lambda q: (q.is_valid, q.visibility_score, -q.viewing_angle_deg))
        sel_vp = vp_map[best_q.viewpoint_id]

        sel_score = None
        if action_scores:
            for s in action_scores:
                if s.viewpoint_id == sel_vp.viewpoint_id:
                    sel_score = s
                    break
        return sel_vp, sel_score

    @staticmethod
    def select_action_conditioned(
        viewpoints: List[CandidateViewpoint],
        action_scores: List[ActionViewpointScore],
    ) -> Tuple[CandidateViewpoint, ActionViewpointScore]:
        """策略 4: 基于 v9 动作条件评分 Q(v|a) 选择最优视点。"""
        vp_map = {v.viewpoint_id: v for v in viewpoints}
        feas_scores = [s for s in action_scores if vp_map.get(s.viewpoint_id, CandidateViewpoint("tmp", [], 0, 0, 0)).feasible]
        pool = feas_scores if feas_scores else action_scores

        best_score = max(pool, key=lambda s: (s.total_score, s.action_delta, s.geometry_score))
        sel_vp = vp_map[best_score.viewpoint_id]
        return sel_vp, best_score

    @classmethod
    def select(
        cls,
        viewpoints: List[CandidateViewpoint],
        action_scores: List[ActionViewpointScore],
        geometry_qualities: Optional[List[ViewpointQuality]] = None,
        strategy: str = "action_conditioned",
        human_position: Optional[Union[List[float], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[CandidateViewpoint, ActionViewpointScore]:
        """统一视点选择入口。"""
        strat = strategy.lower().strip()
        def _get_fallback_score(sel_s):
            if sel_s is not None:
                return sel_s
            return action_scores[0] if action_scores else None

        if strat == "random":
            vp, s = cls.select_random(viewpoints, action_scores, seed=seed)
            return vp, _get_fallback_score(s)
        elif strat == "nearest":
            vp, s = cls.select_nearest(viewpoints, action_scores, human_position=human_position)
            return vp, _get_fallback_score(s)
        elif strat in ["geometry_best", "v8_geometry"]:
            if geometry_qualities:
                vp, s = cls.select_geometry_best(viewpoints, geometry_qualities, action_scores)
                return vp, _get_fallback_score(s)
            else:
                # 若无独立 quality 对象，根据 action_scores 中的 geometry_score 排序
                if not action_scores:
                    feas = [v for v in viewpoints if v.feasible]
                    return (feas[0] if feas else viewpoints[0]), None
                vp_map = {v.viewpoint_id: v for v in viewpoints}
                feas = [s for s in action_scores if vp_map[s.viewpoint_id].feasible]
                best_s = max(feas if feas else action_scores, key=lambda s: s.geometry_score)
                return vp_map[best_s.viewpoint_id], best_s
        elif strat in ["action_conditioned", "v9_action", "ours"]:
            if not action_scores:
                feas = [v for v in viewpoints if v.feasible]
                return (feas[0] if feas else viewpoints[0]), None
            return cls.select_action_conditioned(viewpoints, action_scores)
        else:
            logger.warning("Unknown strategy '%s', falling back to 'action_conditioned'", strategy)
            if not action_scores:
                feas = [v for v in viewpoints if v.feasible]
                return (feas[0] if feas else viewpoints[0]), None
            return cls.select_action_conditioned(viewpoints, action_scores)
