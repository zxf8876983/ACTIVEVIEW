"""
视点效用预测器模块接口 —— utility_predictor.py (v11.3 Interface Stub)
====================================================================

职责：
    1. 为 v11.3 (Viewpoint Utility Prediction) 定义标准接口；
    2. 基于当前机器人观测状态与候选视点特征预测未来动作识别信息增益；
    3. 当前阶段保持接口定义，待 v11.3 阶段正式实现。
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("utility_predictor")


class ViewpointUtilityPredictor:
    """视点识别效用预测器接口 (v11.3 待实现)。"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.config = config or {}

    def predict_utility(
        self,
        current_observation: Dict[str, Any],
        candidate_viewpoints: List[Viewpoint],
    ) -> List[float]:
        """预测每个候选视点的未来动作识别效用得分。"""
        raise NotImplementedError("ViewpointUtilityPredictor will be implemented in ACTIVEVIEW v11.3.")
