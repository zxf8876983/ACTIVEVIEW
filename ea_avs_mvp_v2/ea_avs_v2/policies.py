"""
视角选择策略模块 —— policies.py
=================================

功能：
    实现四种视角选择策略。

v2.0 重要约束：
    - OursPolicy 必须允许选择 current_view（不移动）
    - OursPolicy 只能使用 pred_score，不能使用 true_score
    - OursPolicy 将 current_view 与候选点一起比较 Q_pred
"""

from typing import List
import numpy as np

from .candidate_sampler import CandidateView


class FixedPolicy:
    """固定视角策略（基线）：机器人不移动，使用初始视角。"""

    name = "Fixed"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        return current_view


class RandomPolicy:
    """随机策略：从有效候选点中随机选择一个。"""

    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        idx = self.rng.randint(len(valid))
        return valid[idx]


class NearestPolicy:
    """最近距离策略：选择测地距离最小的候选点。"""

    name = "Nearest"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return min(valid, key=lambda c: c.geodesic_distance)


class OursPolicy:
    """最优策略（Ours）：选择 Q_pred 最大的位姿。

    v2.0 关键设计：
        - 将 current_view 与所有有效候选点一起按 Q_pred 排序
        - 如果 current_view 的 Q_pred 最大，则允许不移动
        - 只能使用 pred_score，不能使用 true_score
    """

    name = "Ours"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        """选择 Q_pred 最大的位姿（包括 current_view）。

        参数：
            current_view: 初始视角（参与 Q_pred 比较）。
            candidates: 候选视角列表。

        返回：
            Q_pred 最大的位姿（可能是 current_view 或某个候选点）。
        """
        # 收集所有候选（包括 current_view）
        all_views = [current_view]
        all_views.extend(c for c in candidates if c.is_valid)

        # 过滤掉没有 Q_pred 评分的
        scored = [v for v in all_views
                  if v.pred_score and v.pred_score.get("Q_pred") is not None]
        if not scored:
            return current_view

        return max(scored, key=lambda v: v.pred_score["Q_pred"])
