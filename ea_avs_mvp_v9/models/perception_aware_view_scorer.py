"""
感知感知型主动视角打分模型 —— perception_aware_view_scorer.py
============================================================

职责：
    1. 融合当前不完整感知状态 (Observation Embedding, 32d) 与候选视点几何描述子 (View Embedding, 32d)；
    2. 预测视角迁移带来的信息增益 (Information Gain) G_hat(v | O_t) in [0.0, 1.0]；
    3. 严禁使用 GT 姿态、SMPL 参数或动作标签作为输入。

# GT is only used for supervision/evaluation.
# It must never enter model forward pass.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .observation_encoder import ObservationEncoder
from .view_encoder import ViewFeatureEncoder


class PerceptionAwareViewScorer(nn.Module):
    """v9.1 基于感知质量的人体主动视角打分网络 (G(v | O_t))。"""

    def __init__(
        self,
        obs_input_dim: int = 71,
        obs_embed_dim: int = 32,
        view_input_dim: int = 13,
        view_embed_dim: int = 32,
        fusion_hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_encoder = ObservationEncoder(
            input_dim=obs_input_dim,
            hidden_dim=64,
            embed_dim=obs_embed_dim,
            dropout=dropout,
        )

        self.view_encoder = ViewFeatureEncoder(
            input_dim=view_input_dim,
            hidden_dim=64,
            embed_dim=view_embed_dim,
            dropout=dropout,
        )

        fusion_in_dim = obs_embed_dim + view_embed_dim
        h1, h2 = fusion_hidden_dims

        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Sigmoid(),  # 输出预测的信息增益 [0.0, 1.0]
        )

    def forward(
        self,
        obs_input: torch.Tensor,
        view_input: torch.Tensor,
        ablate_obs: bool = False,
    ) -> torch.Tensor:
        """
        # GT is only used for supervision/evaluation.
        # It must never enter model forward pass.

        Args:
            obs_input: (B, 71) 当前人体观测感知质量特征 (估计坐标 + 关节置信度 + 部位置信度)
            view_input: (B, N, 13) 或 (B, 13) 候选视角多维几何描述子
            ablate_obs: 是否消融当前感知特征
        Returns:
            gains: (B, N) 或 (B, 1) 预测信息增益 G_hat(v | O_t)
        """
        is_multi_view = (view_input.dim() == 3)
        b_size = view_input.size(0)

        # 1. 编码当前感知特征
        if ablate_obs:
            obs_feat = torch.zeros(b_size, self.obs_encoder.net[-2].out_features, device=obs_input.device)
        else:
            obs_feat = self.obs_encoder(obs_input)  # (B, 32)

        # 2. 编码视角特征并执行融合
        if is_multi_view:
            n_views = view_input.size(1)
            flat_views = view_input.view(-1, view_input.size(-1))
            view_feat = self.view_encoder(flat_views)  # (B*N, 32)

            exp_obs = obs_feat.unsqueeze(1).expand(-1, n_views, -1).contiguous().view(-1, obs_feat.size(-1))

            fused_input = torch.cat([exp_obs, view_feat], dim=-1)  # (B*N, 64)
            out = self.fusion_net(fused_input)  # (B*N, 1)
            scores = out.view(b_size, n_views)  # (B, N)
        else:
            view_feat = self.view_encoder(view_input)  # (B, 32)
            fused_input = torch.cat([obs_feat, view_feat], dim=-1)  # (B, 64)
            scores = self.fusion_net(fused_input)  # (B, 1)

        return scores


# 别名兼容
LearnableViewScorer = PerceptionAwareViewScorer
