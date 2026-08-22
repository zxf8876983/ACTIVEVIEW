"""
主动视角选择决策策略接口 —— selector.py (v11.4 Interface Stub)
=============================================================

职责：
    1. 为 v11.4 (Single-step Active View Policy) 定义标准接口；
    2. 基于效用增益与移动开销联合评分 Score(v) = Gain(v) - lambda * Cost(v) 选择最优观察位姿；
    3. 当前阶段保持接口定义，待 v11.4 阶段正式实现。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("active_selector")


class ActiveViewSelector:
    """单步主动视角选择器接口 (v11.4 待实现)。"""

    def __init__(self, cost_lambda: float = 0.1, config: Optional[Dict[str, Any]] = None):
        self.cost_lambda = cost_lambda
        self.config = config or {}

    def select_best_viewpoint(
        self,
        candidate_viewpoints: List[Viewpoint],
        utility_scores: List[float],
    ) -> Tuple[Viewpoint, float]:
        """选择综合得分最高的最优观察视点。"""
        raise NotImplementedError("ActiveViewSelector will be implemented in ACTIVEVIEW v11.4.")
