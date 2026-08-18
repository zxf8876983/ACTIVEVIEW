"""
Oracle 上界策略 —— oracle_policy.py
=====================================

功能：
    离线仿真性能上界。
    Oracle 在评估阶段（evaluation phase）使用所有位姿的 true_score 的 Q_true
    选择上界位姿，用于衡量其他策略与理想选择的差距。

重要声明：
    Oracle 不是可部署策略，绝不能作为 Ours 的输入或训练标签在线使用。
    论文与代码中标注为：Oracle-NBV (offline upper bound)。
"""

from typing import List, Optional

from .candidate_sampler import CandidateView

# true_score 评价来源常量（与 true_evaluator 保持一致）
TRUE_SOURCE_DEPTH = "depth"
TRUE_SOURCE_GEOMETRY = "geometry_fallback"


def is_depth_true(view: CandidateView) -> bool:
    """判断该位姿的 true_score 是否基于真实渲染 depth 评价。"""
    if view is None or not view.true_score:
        return False
    return view.true_score.get("true_evaluation_source") == TRUE_SOURCE_DEPTH


class OraclePolicy:
    """Oracle 上界策略。

    运行时机：
        仅在 evaluation phase（评估阶段）离线运行，此时所有位姿的 true_score
        已经计算完毕。

    口径要求（v5.0 第三轮）：
        Oracle 只比较 评价来源 == "depth" 且 depth_coverage_true >= min_depth_coverage
        的候选位姿，绝不混用 geometry fallback / 低 depth 覆盖的候选，保证上界
        真正同口径。同时统计被 depth_coverage 门槛排除的候选数。

    ⚠ 在线选择阶段禁止使用 Oracle 的 Q_true 信息。
    """

    name = "Oracle"

    def __init__(self, min_depth_coverage: float = 0.8):
        self.min_depth_coverage = min_depth_coverage

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ):
        """根据 depth 口径 + depth_coverage 门槛选择最大位姿。

        参数：
            current_view: 当前视角。
            candidates: 候选位姿列表。

        返回：
            (best_view, detail)
            detail = {
                "valid_true_count": 满足"depth 来源"的位姿数,
                "depth_eligible_count": depth 覆盖达标位姿数,
                "excluded_low_depth_coverage_count": 因覆盖不足被排除的位姿数,
            }
            best_view：Q_true 最大的达标位姿；无达标位姿时返回 None。
        """
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
) -> float:
    """计算 Oracle 与 Ours 的真实得分差距。

    定义：
        oracle_gap = oracle_Q_true - ours_Q_true

    其中 ours_Q_true 取 Ours 选中位姿的 Q_true。
    若 Ours 或 current 缺少 true_score，则按 0.0 处理。

    参数：
        oracle_view: Oracle 选中的上界位姿。
        ours_view: Ours（FullOurs）选中的位姿。
        current_view: 当前视角（作为 fallback 参考）。

    返回：
        oracle_gap 浮点数。
    """
    oracle_q = _safe_q_true(oracle_view, current_view)
    ours_q = _safe_q_true(ours_view, current_view)
    return float(oracle_q - ours_q)


def _safe_q_true(view: CandidateView, fallback: CandidateView) -> float:
    """安全获取 view 的 Q_true；缺失时退回 fallback 的 Q_true。"""
    if view is None:
        view = fallback
    if view.true_score is None or not view.true_score:
        view = fallback
    return float(view.true_score.get("Q_true", 0.0))