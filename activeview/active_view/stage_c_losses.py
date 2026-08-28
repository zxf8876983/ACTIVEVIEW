"""Regression and listwise losses for Stage C Utility prediction."""

from __future__ import annotations

import torch
from torch import nn


def utility_gap_pairwise_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    tau_gap: float = 1.0,
    max_weight: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weight pairwise ordering errors by the ground-truth utility gap.

    The implicit Stay action has utility and score zero and is included in the
    pair set. Padded candidates are excluded through ``candidate_mask``.
    """
    if tau_gap <= 0:
        raise ValueError("tau_gap must be positive")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if predicted_utility.shape != target_utility.shape:
        raise ValueError("predicted_utility and target_utility must have the same shape")
    if candidate_mask.shape != target_utility.shape:
        raise ValueError("candidate_mask must match utility tensors")
    if predicted_utility.ndim != 2:
        raise ValueError("utility tensors must have shape (batch, candidates)")
    valid = candidate_mask.bool()
    zeros = torch.zeros((target_utility.size(0), 1), device=target_utility.device, dtype=target_utility.dtype)
    target_scores = torch.cat([zeros, target_utility], dim=1)
    predicted_scores = torch.cat([zeros, predicted_utility], dim=1)
    action_mask = torch.cat([
        torch.ones((valid.size(0), 1), device=valid.device, dtype=torch.bool),
        valid,
    ], dim=1)
    ordered = target_scores.unsqueeze(2) > target_scores.unsqueeze(1)
    pair_mask = action_mask.unsqueeze(2) & action_mask.unsqueeze(1) & ordered
    gap = (target_scores.unsqueeze(2) - target_scores.unsqueeze(1)).clamp_min(0.0)
    weights = (gap / float(tau_gap)).clamp_max(float(max_weight))
    score_delta = predicted_scores.unsqueeze(2) - predicted_scores.unsqueeze(1)
    pair_loss = nn.functional.softplus(-score_delta)
    selected_weights = weights[pair_mask]
    if selected_weights.numel() == 0:
        return predicted_utility.sum() * 0.0
    return (selected_weights * pair_loss[pair_mask]).sum() / selected_weights.sum().clamp_min(float(eps))


def stage_c_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    lambda_reg: float = 1.0,
    lambda_rank: float = 1.0,
    tau: float = 0.5,
    lambda_gap: float = 0.0,
    tau_gap: float = 1.0,
    max_gap_weight: float = 10.0,
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
    gap_loss = (
        utility_gap_pairwise_loss(
            predicted_utility,
            target_utility,
            candidate_mask,
            tau_gap=tau_gap,
            max_weight=max_gap_weight,
        )
        if float(lambda_gap) != 0.0
        else predicted_utility.sum() * 0.0
    )
    total = float(lambda_reg) * huber + float(lambda_rank) * listwise + float(lambda_gap) * gap_loss
    return {"total": total, "regression": huber, "ranking": listwise, "gap_ranking": gap_loss}
