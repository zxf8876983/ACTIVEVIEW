"""
视角排序与回归损失函数 —— losses.py
====================================

职责：
    1. PairwiseRankingLoss: 学习视点相对偏好排序 (保证高质量视角得分 > 低质量视角)；
    2. CombinedRankingRegressionLoss: 结合排序损失与平滑 L1 回归损失，兼顾排序精度与数值校准。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """候选视点两两配对排序损失 (Margin-based Pairwise Ranking Loss)。"""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: (B, N) 模型对各视点的预测得分
            target_scores: (B, N) 目标基准效用得分
        Returns:
            Scalar ranking loss
        """
        # pred_diff[b, i, j] = pred[b, i] - pred[b, j]
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # (B, N, N)
        target_diff = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)  # (B, N, N)

        # 标签 y_ij = sign(target_i - target_j)
        sign = torch.sign(target_diff)  # (B, N, N)

        # 仅对存在明显目标分差的 pair 计算损失 (|target_diff| > 1e-4)
        mask = (torch.abs(target_diff) > 1e-4).float()

        # margin loss = max(0, -sign * (pred_i - pred_j) + margin)
        loss_matrix = F.relu(-sign * pred_diff + self.margin) * mask

        valid_pairs = torch.sum(mask)
        if valid_pairs > 0:
            return torch.sum(loss_matrix) / valid_pairs
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)


class CombinedRankingRegressionLoss(nn.Module):
    """排序与数值回归混合损失函数。"""

    def __init__(
        self,
        margin: float = 0.1,
        ranking_weight: float = 1.0,
        regression_weight: float = 0.5,
    ):
        super().__init__()
        self.ranking_loss = PairwiseRankingLoss(margin=margin)
        self.regression_loss = nn.SmoothL1Loss()
        self.ranking_weight = ranking_weight
        self.regression_weight = regression_weight

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        r_loss = self.ranking_loss(pred_scores, target_scores)
        reg_loss = self.regression_loss(pred_scores, target_scores)
        return self.ranking_weight * r_loss + self.regression_weight * reg_loss
