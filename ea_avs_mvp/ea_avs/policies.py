"""
视角选择策略模块 —— policies.py
=================================

本文件实现 EA-AVS-MVP v0.1.1 的四种观察位姿选择策略。

v0.1.1 的关键修改：
    OursPolicy 不再强制机器人移动到某个候选点，而是把 current_view
    和所有有效候选点放在一起比较。如果当前视角的预测质量最高，
    则机器人保持原地不动。

这样更符合主动感知逻辑：
    当前视角足够好时不移动；当前视角质量不足时再选择下一观察位姿。
"""

from typing import List
import numpy as np

from .candidate_sampler import CandidateView


def _get_q(score: dict) -> float:
    """
    从评分字典中读取预测综合质量分数。

    说明：
        v0.1.1 开始推荐使用 Q_pred 表示“移动前预测评分”。
        为了兼容旧版本输出，如果没有 Q_pred，则尝试读取旧字段 Q。

    参数：
        score: 由 ViewpointEvaluator.score_view() 返回的评分字典。

    返回：
        预测综合质量分数；如果评分不存在，则返回负无穷，避免被策略选中。
    """
    if not score:
        return float("-inf")
    if "Q_pred" in score:
        return float(score["Q_pred"])
    if "Q" in score:
        return float(score["Q"])
    return float("-inf")


class FixedPolicy:
    """
    固定视角基线策略。

    策略含义：
        机器人不主动移动，始终使用初始观察位姿。

    作用：
        作为“不做主动视角选择”的基线，用于衡量主动重观测是否有收益。
    """

    name = "Fixed"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        # Fixed 策略完全忽略候选点，直接返回当前视角。
        return current_view


class RandomPolicy:
    """
    随机候选视角策略。

    策略含义：
        从所有有效候选观察位姿中随机选择一个。

    作用：
        用于判断“随便移动一下”是否就能带来收益。如果 Ours 明显优于
        Random，说明评分函数确实提供了有用的选择依据。
    """

    name = "Random"

    def __init__(self, seed: int = 42):
        # 使用固定随机种子，保证实验可复现。
        self.rng = np.random.RandomState(seed)

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            # 没有有效候选点时退回当前视角，保证程序不崩溃。
            return current_view
        idx = self.rng.randint(len(valid))
        return valid[idx]


class NearestPolicy:
    """
    最近候选视角策略。

    策略含义：
        从所有有效候选点中选择测地距离最短的一个。

    作用：
        代表“最小移动代价”基线。它不关心视角质量，只关心走得近。
    """

    name = "Nearest"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return min(valid, key=lambda c: c.geodesic_distance)


class OursPolicy:
    """
    本方法策略：选择预测质量最高的观察位姿。

    v0.1.1 关键点：
        本策略不会强制机器人移动。它会比较：
            1. current_view 当前视角；
            2. 所有有效候选观察位姿。
        然后选择 Q_pred 最大的观察位姿。

    如果当前视角已经比所有候选点更好，机器人就保持原地不动。
    """

    name = "Ours"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        # 只保留有效并且有预测评分的候选点。
        valid = [c for c in candidates if c.is_valid and _get_q(c.score) != float("-inf")]

        # 将 current_view 也加入候选池，允许策略选择“不移动”。
        pool = [current_view] + valid

        # 返回预测综合质量最高的观察位姿。
        return max(pool, key=lambda c: _get_q(c.score))
