"""
人体状态感知学习型视点打分模型 —— view_scorer.py
=================================================

职责：
    1. 融合人体物理状态特征 (Human Pose Embedding, 32d) 与候选视点描述子 (View Embedding, 32d)；
    2. 通过全连接 Fusion MLP 输出视角连续效用评分 Q_hat(v | H) ∈ [0.0, 1.0]；
    3. 严禁任何 Action Label 参与模型结构与前向推理；
    4. 支持姿态消融开关 (ablate_pose)。
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .pose_encoder import HumanPoseEncoder
from .view_encoder import ViewFeatureEncoder


class LearnableViewScorer(nn.Module):
    """v9.1 人体状态感知学习型视点质量打分网络 (Q(v | H))。"""

    def __init__(
        self,
        pose_input_dim: int = 49,
        pose_embed_dim: int = 32,
        view_input_dim: int = 13,
        view_embed_dim: int = 32,
        fusion_hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_encoder = HumanPoseEncoder(
            input_dim=pose_input_dim,
            hidden_dim=64,
            embed_dim=pose_embed_dim,
            dropout=dropout,
        )

        self.view_encoder = ViewFeatureEncoder(
            input_dim=view_input_dim,
            hidden_dim=64,
            embed_dim=view_embed_dim,
            dropout=dropout,
        )

        fusion_in_dim = pose_embed_dim + view_embed_dim
        h1, h2 = fusion_hidden_dims

        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Sigmoid(),  # 输出连续效用得分 [0.0, 1.0]
        )

    def forward(
        self,
        pose_input: torch.Tensor,
        view_input: torch.Tensor,
        ablate_pose: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pose_input: (B, 49) 人体 16 骨骼关键点相对坐标及朝向
            view_input: (B, N, 13) 或 (B, 13) 候选视角多维几何与 7 大部位观测特征
            ablate_pose: 是否消融人体姿态特征 (置零)
        Returns:
            scores: (B, N) 或 (B, 1) 候选视角预测效用得分 Q_hat(v | H)
        """
        is_multi_view = (view_input.dim() == 3)
        b_size = view_input.size(0)

        # 1. 编码姿态特征 (Human State Embedding)
        if ablate_pose:
            pose_feat = torch.zeros(b_size, self.pose_encoder.net[-3].out_features, device=pose_input.device)
        else:
            pose_feat = self.pose_encoder(pose_input)  # (B, 32)

        # 2. 编码视角特征 (View Embedding)
        if is_multi_view:
            n_views = view_input.size(1)
            flat_views = view_input.view(-1, view_input.size(-1))
            view_feat = self.view_encoder(flat_views)  # (B*N, 32)

            exp_pose = pose_feat.unsqueeze(1).expand(-1, n_views, -1).contiguous().view(-1, pose_feat.size(-1))

            fused_input = torch.cat([exp_pose, view_feat], dim=-1)  # (B*N, 64)
            out = self.fusion_net(fused_input)  # (B*N, 1)
            scores = out.view(b_size, n_views)  # (B, N)
        else:
            view_feat = self.view_encoder(view_input)  # (B, 32)
            fused_input = torch.cat([pose_feat, view_feat], dim=-1)  # (B, 64)
            scores = self.fusion_net(fused_input)  # (B, 1)

        return scores
