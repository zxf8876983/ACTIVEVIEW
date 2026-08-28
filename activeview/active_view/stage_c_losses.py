"""Regression and listwise losses for Stage C Utility prediction."""

from __future__ import annotations

import torch
from torch import nn


def stage_c_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    lambda_reg: float = 1.0,
    lambda_rank: float = 1.0,
    tau: float = 0.5,
) -> dict[str, torch.Tensor]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    valid = candidate_mask.bool()
    if not bool(valid.any()):
        raise ValueError("candidate_mask contains no valid candidates")
    huber = nn.functional.smooth_l1_loss(predicted_utility[valid], target_utility[valid], reduction="mean")
    scores_gt = torch.cat([torch.zeros((target_utility.size(0), 1), device=target_utility.device), target_utility], dim=1)
    scores_pred = torch.cat([torch.zeros((predicted_utility.size(0), 1), device=predicted_utility.device), predicted_utility], dim=1)
    rank_mask = torch.cat([torch.ones((candidate_mask.size(0), 1), dtype=torch.bool, device=candidate_mask.device), valid], dim=1)
    masked_gt = scores_gt.masked_fill(~rank_mask, float("-inf"))
    masked_pred = scores_pred.masked_fill(~rank_mask, float("-inf"))
    target_distribution = torch.softmax(masked_gt / tau, dim=1)
    log_prediction = torch.log_softmax(masked_pred / tau, dim=1)
    log_prediction = torch.where(rank_mask, log_prediction, torch.zeros_like(log_prediction))
    listwise = -(target_distribution * log_prediction).sum(dim=1).mean()
    total = float(lambda_reg) * huber + float(lambda_rank) * listwise
    return {"total": total, "regression": huber, "ranking": listwise}
