"""
视角选择策略模块 —— policies.py
=================================

功能：
    基础视角选择策略（基线 + 主策略），从 v3.0 迁移。

约束（v4.0 延续 v2.0/v3.0）：
    - Ours 只能使用 pred_score["Q_pred"]
    - Ours 必须比较 current_view 和所有候选点
    - Ours 必须允许不移动（选择 current_view）
    - 选择阶段禁止使用候选点未来 RGB / depth / Q_true

完整 v4.0 的 6 个模块化消融策略见 ablation_policies.py。
"""

from typing import List

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


class OursPolicy:
    """最优策略（Ours / Full）：选择 Q_pred 最大的位姿。

    v4.0 约束：
        - 只能使用 Q_pred 评分（含遮挡感知动作部位得分）
        - 将 current_view 与候选点一起比较
        - 允许选择 current_view（不移动）
    """
    name = "Ours"

    def __init__(self):
        self.last_selection_stats = {
            "excluded_invalid_occ_count": 0,
            "fell_back_to_current": False,
            "fallback_reason": None,
        }

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        """选择 Q_pred 最大的 eligible 位姿。

        v5.0 closure：过滤 is_occlusion_valid_pred=False 的候选（遮挡判断不可信
        的候选不能入选）。current_view 作为安全 fallback，但若其自身 occlusion
        也 invalid，不会假装它 valid。

        选后统计通过 self.last_selection_stats 暴露（主脚本读取）。
        """
        def is_occ_valid(v):
            return bool(v.pred_score and v.pred_score.get(
                "is_occlusion_valid_pred", False))

        eligible = []
        excluded = 0
        for c in candidates:
            if not c.is_valid:
                continue
            if not (c.pred_score and c.pred_score.get("Q_pred") is not None):
                continue
            if is_occ_valid(c):
                eligible.append(c)
            else:
                excluded += 1

        current_valid = bool(
            current_view.pred_score
            and current_view.pred_score.get("is_occlusion_valid_pred", False)
            and current_view.pred_score.get("Q_pred") is not None)

        if eligible:
            self.last_selection_stats = {
                "excluded_invalid_occ_count": excluded,
                "fell_back_to_current": False,
                "fallback_reason": None,
            }
            return max(eligible, key=lambda v: v.pred_score["Q_pred"])

        # 无 eligible → 退回 current（即使 current occlusion invalid 也作安全兜底）
        self.last_selection_stats = {
            "excluded_invalid_occ_count": excluded,
            "fell_back_to_current": True,
            "fallback_reason": (
                "no_valid_occ_candidate"
                if not current_valid else "no_scored_candidate"),
        }
        return current_view