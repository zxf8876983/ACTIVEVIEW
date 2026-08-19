"""
视角选择策略模块 —— policies.py
=================================

功能：
    基础视角选择策略（基线 + EstimatedState-Ours 主策略 + GTState-Ours 特权基线）。
"""

from typing import List, Optional
import numpy as np

from .candidate_sampler import CandidateView


class FixedPolicy:
    """固定视角策略（基线）：不移动，停留在当前位置。"""
    name = "Fixed"

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        return current_view


class RandomPolicy:
    """随机策略（基线）：从有效候选点中随机选择一个。"""
    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return valid[self.rng.randint(len(valid))]


class NearestPolicy:
    """最近距离策略（基线）：选择测地距离最近的候选点。"""
    name = "Nearest"

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return min(valid, key=lambda c: c.geodesic_distance)


class EstimatedStateOursPolicy:
    """估计状态最优策略（v6.0 主策略）：基于纯视觉估计状态评分选择最优位姿。"""
    name = "EstimatedState-Ours"

    def __init__(self):
        self.last_selection_stats = {
            "excluded_invalid_occ_count": 0,
            "eligible_candidate_count": 0,
            "current_occ_valid": False,
            "selected_is_current": True,
            "fell_back_to_current": False,
            "fallback_reason": None,
        }

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
        is_state_valid: bool = True,
    ) -> CandidateView:
        """从候选视角中选择最优位姿。

        当人体状态估计失效 (is_state_valid=False) 时，安全停留在原地 (stay)，
        严禁回退到 GT 状态。
        """
        if not is_state_valid:
            self.last_selection_stats = {
                "excluded_invalid_occ_count": 0,
                "eligible_candidate_count": 0,
                "current_occ_valid": False,
                "selected_is_current": True,
                "fell_back_to_current": True,
                "fallback_reason": "perception_invalid_fallback",
            }
            return current_view

        def is_occ_valid(v):
            return bool(
                v.pred_score and v.pred_score.get("is_occlusion_valid_pred", False)
            )

        current_valid = bool(
            current_view.pred_score
            and current_view.pred_score.get("is_occlusion_valid_pred", False)
            and current_view.pred_score.get("Q_pred") is not None
        )

        eligible = []
        excluded = 0

        if current_valid:
            eligible.append(current_view)

        for c in candidates:
            if not c.is_valid:
                continue
            if not (c.pred_score and c.pred_score.get("Q_pred") is not None):
                continue
            if is_occ_valid(c):
                eligible.append(c)
            else:
                excluded += 1

        if eligible:
            selected = max(eligible, key=lambda v: v.pred_score["Q_pred"])
            self.last_selection_stats = {
                "excluded_invalid_occ_count": excluded,
                "eligible_candidate_count": len(eligible),
                "current_occ_valid": current_valid,
                "selected_is_current": selected is current_view,
                "fell_back_to_current": False,
                "fallback_reason": None,
            }
            return selected

        self.last_selection_stats = {
            "excluded_invalid_occ_count": excluded,
            "eligible_candidate_count": 0,
            "current_occ_valid": current_valid,
            "selected_is_current": True,
            "fell_back_to_current": True,
            "fallback_reason": "no_valid_occ_view",
        }
        return current_view


class GTStateOursPolicy:
    """GT 状态特权基线策略（对比基准）：基于真实 Humanoid GT 状态评分选择最优位姿。"""
    name = "GTState-Ours"

    def __init__(self):
        self.last_selection_stats = {
            "excluded_invalid_occ_count": 0,
            "eligible_candidate_count": 0,
            "current_occ_valid": False,
            "selected_is_current": True,
            "fell_back_to_current": False,
            "fallback_reason": None,
        }

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        def is_occ_valid(v):
            return bool(
                v.pred_score and v.pred_score.get("is_occlusion_valid_pred", False)
            )

        current_valid = bool(
            current_view.pred_score
            and current_view.pred_score.get("is_occlusion_valid_pred", False)
            and current_view.pred_score.get("Q_pred") is not None
        )

        eligible = []
        excluded = 0

        if current_valid:
            eligible.append(current_view)

        for c in candidates:
            if not c.is_valid:
                continue
            if not (c.pred_score and c.pred_score.get("Q_pred") is not None):
                continue
            if is_occ_valid(c):
                eligible.append(c)
            else:
                excluded += 1

        if eligible:
            selected = max(eligible, key=lambda v: v.pred_score["Q_pred"])
            self.last_selection_stats = {
                "excluded_invalid_occ_count": excluded,
                "eligible_candidate_count": len(eligible),
                "current_occ_valid": current_valid,
                "selected_is_current": selected is current_view,
                "fell_back_to_current": False,
                "fallback_reason": None,
            }
            return selected

        self.last_selection_stats = {
            "excluded_invalid_occ_count": excluded,
            "eligible_candidate_count": 0,
            "current_occ_valid": current_valid,
            "selected_is_current": True,
            "fell_back_to_current": True,
            "fallback_reason": "no_valid_occ_view",
        }
        return current_view


# 兼容别名
OursPolicy = EstimatedStateOursPolicy
