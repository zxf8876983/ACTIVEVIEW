"""
视角选择策略模块 —— policies.py
=================================

功能：
    实现四种不同的视角选择策略（Policy），用于对比实验。

MVP0.1 版本定义了四种策略：
    1. Fixed —— 基线策略：机器人不移动，使用初始视角
    2. Random —— 随机策略：从有效候选点中随机选择一个
    3. Nearest —— 最近策略：选择测地距离最小的候选点
    4. Ours —— 最优策略：选择综合评分 Q 最大的候选点

实验目标：验证 Ours 策略是否优于 Fixed 基线策略。
"""

from typing import List
import numpy as np

from .candidate_sampler import CandidateView


class FixedPolicy:
    """
    固定视角策略（基线策略）。
    
    策略行为：
        机器人始终保持初始位置不动，不选择任何候选点。
    
    作为基线的意义：
        - 代表"不做任何主动视角选择"的情况
        - 如果 Ours 策略的评分不如 Fixed，说明主动选择没有意义
        - 用于计算主动选择带来的性能提升
    
    实现：
        select() 直接返回 current_view（初始视角），忽略所有候选点。
    """

    name = "Fixed"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        """
        选择当前视角（不移动机器人）。

        参数：
            current_view: 机器人的初始视角。
            candidates: 候选视角列表（此策略不使用）。

        返回：
            current_view 本身。
        """
        return current_view


class RandomPolicy:
    """
    随机视角选择策略。
    
    策略行为：
        从所有有效候选点中随机选择一个。
        如果没有任何有效候选点，则退回使用初始视角。
    
    种子控制：
        使用固定的随机种子（与 project.seed 一致），
        保证实验结果可复现。
    """

    name = "Random"

    def __init__(self, seed: int = 42):
        """
        初始化随机策略。

        参数：
            seed: 随机种子，默认为 42。应与配置文件中的 project.seed 一致。
        """
        self.rng = np.random.RandomState(seed)

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        """
        随机选择一个有效候选点。

        参数：
            current_view: 初始视角（当没有有效候选点时作为 fallback）。
            candidates: 候选视角列表。

        返回：
            随机选中的一个有效候选点；如果没有候选点则返回 current_view。
        """
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view  # fallback 到初始视角
        idx = self.rng.randint(len(valid))
        return valid[idx]


class NearestPolicy:
    """
    最近距离策略。
    
    策略行为：
        选择测地距离（geodesic distance）最小的有效候选点。
        测地距离表示机器人需要实际行走的路径长度。
    
    设计意图：
        - 代表"最小移动代价"的策略
        - 与 Ours 策略对比：如果 Ours 选择更远的点但评分更高，
          说明移动是值得的
    """

    name = "Nearest"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        """
        选择测地距离最小的有效候选点。

        参数：
            current_view: 初始视角（fallback）。
            candidates: 候选视角列表。

        返回：
            测地距离最小的有效候选点。
        """
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return min(valid, key=lambda c: c.geodesic_distance)


class OursPolicy:
    """
    最优视角选择策略（Ours —— 本项目的核心策略）。
    
    策略行为：
        选择综合评分 Q 值最大的有效候选点。
        Q 值由 ViewpointEvaluator 计算，综合考虑了：
        - S_kp（关键点可见率，权重 0.6）
        - S_center（居中度，权重 0.2）
        - S_dist（距离评分，权重 0.2）
        - C_move（运动代价，权重 0.2，惩罚项）
    
    设计意图：
        - 代表"主动感知"的核心思想
        - 不追求单一指标最大化，而是综合权衡可见性、视角质量和运动代价
        - 期望在平均性能上优于其他三种策略
    """

    name = "Ours"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        """
        选择综合评分 Q 值最大的有效候选点。

        参数：
            current_view: 初始视角（fallback）。
            candidates: 候选视角列表。
                        每个候选点的 score 字典必须包含 "Q" 字段，
                        由 ViewpointEvaluator.score_view() 预先计算。

        返回：
            Q 值最大的有效候选点。
        """
        # 过滤出有效且已有 Q 评分的候选点
        valid = [c for c in candidates if c.is_valid and c.score.get("Q") is not None]
        if not valid:
            return current_view
        return max(valid, key=lambda c: c.score["Q"])
