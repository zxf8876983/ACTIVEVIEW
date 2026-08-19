"""
模块化消融策略 —— ablation_policies.py
========================================

功能：
    模块化消融策略，用于分析各个评分项（FOV/部位/朝向/遮挡）的独立贡献。
"""

from typing import List
from .candidate_sampler import CandidateView


class _BaseAblationPolicy:
    """消融策略基类。"""
    name = "ablation"
    score_key = "Q_pred"
    requires_occlusion_valid = False

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        all_views = []
        if current_view.pred_score and self._score_of(current_view) is not None:
            if (not self.requires_occlusion_valid
                    or current_view.pred_score.get("is_occlusion_valid_pred", False)):
                all_views.append(current_view)
        for c in candidates:
            if not c.is_valid or not c.pred_score:
                continue
            if self._score_of(c) is None:
                continue
            if self.requires_occlusion_valid and not c.pred_score.get(
                    "is_occlusion_valid_pred", False):
                continue
            all_views.append(c)

        if not all_views:
            return current_view
        return max(all_views, key=self._score_of)

    def _score_of(self, view: CandidateView):
        return view.pred_score.get(self.score_key)


class VisibilityOnlyPolicy(_BaseAblationPolicy):
    name = "VisibilityOnly"
    score_key = "S_kp_pred"


class ActionPartOnlyPolicy(_BaseAblationPolicy):
    name = "ActionPartOnly"
    score_key = "S_action_part_pred"


class OrientationOnlyPolicy(_BaseAblationPolicy):
    name = "OrientationOnly"
    score_key = "S_orient_pred"


class OcclusionOnlyPolicy(_BaseAblationPolicy):
    name = "OcclusionOnly"
    score_key = "S_kp_occ_pred"
    requires_occlusion_valid = True


class ActionOrientationPolicy(_BaseAblationPolicy):
    name = "ActionOrientation"

    def _score_of(self, view: CandidateView):
        ps = view.pred_score
        if not ps:
            return None
        return (
            0.50 * ps.get("S_action_part_pred", 0.0)
            + 0.15 * ps.get("S_orient_pred", 0.0)
            + 0.10 * ps.get("S_center_pred", 0.0)
            + 0.10 * ps.get("S_dist_pred", 0.0)
            - 0.15 * ps.get("C_move", 0.0)
        )


class FullOursPolicy(_BaseAblationPolicy):
    name = "FullOurs"
    score_key = "Q_pred"
    requires_occlusion_valid = True
