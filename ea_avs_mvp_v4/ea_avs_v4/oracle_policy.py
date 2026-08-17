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


class OraclePolicy:
    """Oracle 上界策略。

    运行时机：
        仅在 evaluation phase（评估阶段）离线运行，此时所有位姿的 true_score
        已经计算完毕。

    ⚠ 在线选择阶段禁止使用 Oracle 的 Q_true 信息。
    """

    name = "Oracle"

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        """根据 true_score 的 Q_true 选择最大位姿（允许停留在当前位置）。

        参数：
            current_view: 当前视角。
            candidates: 候选位姿列表。

        返回：
            Q_true 最大的位姿；无有效评分时返回 current_view。
        """
        all_views = [current_view]
        all_views.extend(c for c in candidates if c.is_valid)

        scored = [
            v for v in all_views
            if v.true_score and v.true_score.get("Q_true") is not None
        ]
        if not scored:
            return current_view
        return max(scored, key=lambda v: v.true_score["Q_true"])


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