"""
Oracle 上界策略 —— oracle_policy.py
=====================================

功能：
    离线仿真性能上界与 Oracle Gap 计算。
"""

from typing import List, Optional
from .candidate_sampler import CandidateView

TRUE_SOURCE_DEPTH = "depth"
TRUE_SOURCE_GEOMETRY = "geometry_fallback"


def is_depth_true(view: CandidateView) -> bool:
    if view is None or not view.true_score:
        return False
    return view.true_score.get("true_evaluation_source") == TRUE_SOURCE_DEPTH


class OraclePolicy:
    """Oracle 离线上界策略。"""
    name = "Oracle"

    def __init__(self, min_depth_coverage: float = 0.8):
        self.min_depth_coverage = min_depth_coverage

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ):
        all_views = [current_view]
        all_views.extend(c for c in candidates if c.is_valid)

        depth_views = [
            v for v in all_views
            if is_depth_true(v) and v.true_score.get("Q_true") is not None
        ]
        eligible = []
        excluded = 0
        for v in all_views:
            if is_depth_true(v) and v.true_score.get("Q_true") is not None:
                dc = v.true_score.get("depth_coverage_true", 0.0)
                if dc >= self.min_depth_coverage:
                    eligible.append(v)
                else:
                    excluded += 1

        if not eligible:
            return None, {
                "valid_true_count": len(depth_views),
                "depth_eligible_count": 0,
                "excluded_low_depth_coverage_count": excluded,
            }
        best = max(eligible, key=lambda v: v.true_score["Q_true"])
        return best, {
            "valid_true_count": len(depth_views),
            "depth_eligible_count": len(eligible),
            "excluded_low_depth_coverage_count": excluded,
        }


def compute_oracle_gap(
    oracle_view: CandidateView,
    ours_view: CandidateView,
    current_view: CandidateView,
    min_depth_coverage: float = 0.8,
) -> dict:
    if oracle_view is None or not oracle_view.true_score:
        return {"oracle_gap": None, "oracle_gap_valid": False,
                "oracle_gap_reason": "oracle_unavailable"}

    oracle_q = float(oracle_view.true_score.get("Q_true", 0.0))

    if ours_view is None or not ours_view.true_score:
        return {"oracle_gap": None, "oracle_gap_valid": False,
                "oracle_gap_reason": "ours_true_missing"}
    ours_q = float(ours_view.true_score.get("Q_true", 0.0))
    if ours_view.true_score.get("true_evaluation_source") != "depth":
        return {"oracle_gap": None, "oracle_gap_valid": False,
                "oracle_gap_reason": "ours_geometry_fallback"}
    if float(ours_view.true_score.get("depth_coverage_true", 0.0)) < min_depth_coverage:
        return {"oracle_gap": None, "oracle_gap_valid": False,
                "oracle_gap_reason": "ours_low_depth_coverage"}

    return {"oracle_gap": float(oracle_q - ours_q), "oracle_gap_valid": True,
            "oracle_gap_reason": None}
