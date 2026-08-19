"""
Oracle 上界策略与 Gap 评估模块 —— oracle_policy.py
=================================================

功能：
    离线仿真性能上界与 Oracle Gap 严格同口径计算。
    具备严格的 Candidate Pool Identity Guard，杜绝跨 Pool 混淆，
    并杜绝使用 max(0) 静默隐藏 Oracle 上界破坏错误。
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
    """Oracle 离线上界策略（在指定 candidate pool 上选择 Q_true 最高位姿）。"""

    def __init__(self, name: str = "Oracle", min_depth_coverage: float = 0.8):
        self.name = name
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
    oracle_view: Optional[CandidateView],
    selected_view: Optional[CandidateView],
    min_depth_coverage: float = 0.8,
) -> dict:
    """计算同一 Candidate Pool 内选定位姿与 Oracle 上界的 Gap。

    科学要求：
        1. 必须具备 Pool Identity Guard，跨 pool 严格返回 pool_mismatch。
        2. 同 pool 下如果 selected_q > oracle_q（超出数值容差 1e-6），
           严格报错返回 oracle_not_upper_bound，禁止使用 max(0) 静默掩盖错误。
        3. 必须满足 depth 真实渲染来源与 depth coverage 阈值。
    """
    if oracle_view is None or not oracle_view.true_score:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "oracle_unavailable",
        }

    if selected_view is None or not selected_view.true_score:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "selected_true_missing",
        }

    # 1. Pool Identity Guard
    if oracle_view.pool_id is None or selected_view.pool_id is None:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "pool_id_missing",
        }

    if oracle_view.pool_id != selected_view.pool_id:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "pool_mismatch",
            "oracle_pool_id": oracle_view.pool_id,
            "selected_pool_id": selected_view.pool_id,
        }

    # 2. Q_true 与 Depth Coverage 校验
    oracle_q = oracle_view.true_score.get("Q_true")
    if oracle_q is None:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "oracle_q_true_none",
        }

    sel_q = selected_view.true_score.get("Q_true")
    if sel_q is None:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "selected_q_true_none",
        }

    if selected_view.true_score.get("true_evaluation_source") != TRUE_SOURCE_DEPTH:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "selected_geometry_fallback",
        }

    dc = float(selected_view.true_score.get("depth_coverage_true", 0.0))
    if dc < min_depth_coverage:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "selected_low_depth_coverage",
        }

    # 3. 严格上界性检验（禁止使用 max(0) 静默掩盖 Oracle 不是 Upper Bound 的错误）
    raw_gap = float(oracle_q - sel_q)
    if raw_gap < -1e-6:
        return {
            "oracle_gap": None,
            "oracle_gap_valid": False,
            "oracle_gap_reason": "oracle_not_upper_bound",
            "oracle_q_true": float(oracle_q),
            "selected_q_true": float(sel_q),
            "raw_gap": raw_gap,
        }

    gap = max(raw_gap, 0.0)
    return {
        "oracle_gap": float(gap),
        "oracle_gap_valid": True,
        "oracle_gap_reason": None,
    }
