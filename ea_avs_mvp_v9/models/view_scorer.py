"""
学习型动作感知视点打分模型 —— view_scorer.py
===========================================

职责：
    1. 融合人体姿态特征 (Pose Embedding, 32d)、动作分类嵌入 (Action Embedding, 16d) 与候选视角特征 (View Embedding, 32d)；
    2. 通过全连接 Fusion MLP 输出视角连续效用评分 Q_hat(v | H, A) ∈ [0.0, 1.0]；
    3. 支持消融实验接口 (ablate_action, ablate_pose)。
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .pose_encoder import HumanPoseEncoder
from .view_encoder import ViewFeatureEncoder


class LearnableViewScorer(nn.Module):
    """v9.1 学习型动作条件化视点质量打分网络。"""

    def __init__(
        self,
        pose_input_dim: int = 49,
        pose_embed_dim: int = 32,
        action_input_dim: int = 5,
        action_embed_dim: int = 16,
        view_input_dim: int = 11,
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

        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, action_embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(action_embed_dim),
        )

        self.view_encoder = ViewFeatureEncoder(
            input_dim=view_input_dim,
            hidden_dim=64,
            embed_dim=view_embed_dim,
            dropout=dropout,
        )

        fusion_in_dim = pose_embed_dim + action_embed_dim + view_embed_dim
        h1, h2 = fusion_hidden_dims

        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Sigmoid(),  # 输出归一化效用得分 [0.0, 1.0]
        )

    def forward(
        self,
        pose_input: torch.Tensor,
        action_input: torch.Tensor,
        view_input: torch.Tensor,
        ablate_action: bool = False,
        ablate_pose: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pose_input: (B, 49) 人体 16 关节相对坐标及朝向
            action_input: (B, 5) 动作类别 One-hot
            view_input: (B, N, 11) 或 (B, 11) 候选视角多维几何观测特征
            ablate_action: 是否消融动作特征 (设为 0)
            ablate_pose: 是否消融人体姿态特征 (设为 0)
        Returns:
            scores: (B, N) 或 (B, 1) 候选视角预测效用得分
        """
        is_multi_view = (view_input.dim() == 3)
        b_size = view_input.size(0)

        # 1. 编码姿态特征
        if ablate_pose:
            pose_feat = torch.zeros(b_size, self.pose_encoder.net[-3].out_features, device=pose_input.device)
        else:
            pose_feat = self.pose_encoder(pose_input)  # (B, 32)

        # 2. 编码动作特征
        if ablate_action:
            action_feat = torch.zeros(b_size, self.action_encoder[0].out_features, device=action_input.device)
        else:
            action_feat = self.action_encoder(action_input)  # (B, 16)

        # 3. 编码视角特征
        if is_multi_view:
            n_views = view_input.size(1)
            # view_input: (B, N, 11) -> (B*N, 11)
            flat_views = view_input.view(-1, view_input.size(-1))
            view_feat = self.view_encoder(flat_views)  # (B*N, 32)

            # 扩展 pose 和 action 特征以对齐 (B*N, ...)
            exp_pose = pose_feat.unsqueeze(1).expand(-1, n_views, -1).contiguous().view(-1, pose_feat.size(-1))
            exp_act = action_feat.unsqueeze(1).expand(-1, n_views, -1).contiguous().view(-1, action_feat.size(-1))

            fused_input = torch.cat([exp_pose, exp_act, view_feat], dim=-1)  # (B*N, 80)
            out = self.fusion_net(fused_input)  # (B*N, 1)
            scores = out.view(b_size, n_views)  # (B, N)
        else:
            view_feat = self.view_encoder(view_input)  # (B, 32)
            fused_input = torch.cat([pose_feat, action_feat, view_feat], dim=-1)  # (B, 80)
            scores = self.fusion_net(fused_input)  # (B, 1)

        return scores
