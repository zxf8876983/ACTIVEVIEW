"""
视点效用预测神经网络模型 —— active_view/models/utility_predictor.py
=============================================================

职责：
    1. 提供轻量级 3 层 MLP 神经网络架构 (ViewpointUtilityPredictorNet)；
    2. 输入：11 维状态-候选视角联合连续特征向量；
    3. 输出：标量预测效用得分 U_hat(v) = Delta H (不确定度降低量)；
    4. 结构：Linear(in_dim, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 32) -> ReLU -> Linear(32, 1)。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("utility_predictor_model")


class ViewpointUtilityPredictorNet(nn.Module):
    """用于视点效用预测的轻量级 3 层 MLP 神经网络。"""

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
            x: (B, in_dim) 连续特征张量
        返回:
            utility: (B, 1) 预测效用得分
        """
        return self.net(x)
