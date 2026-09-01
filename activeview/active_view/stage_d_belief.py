"""Small pure helpers and models for the EXP038--EXP040 belief campaign."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


ACTION_NAMES = ("Stay", "p2", "p3")


def normalize_belief(values: Sequence[float]) -> np.ndarray:
    """Normalize a finite non-negative class belief without changing support."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != 16 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("belief must be a finite non-negative 16-D vector")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("belief must have positive mass")
    return (array / total).astype(np.float32)


def belief_from_log_probs(log_probs: Sequence[float]) -> np.ndarray:
    """Convert frozen ST-GCN log probabilities to a normalized belief."""
    values = np.asarray(log_probs, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("log_probs must have shape (16,)")
    shifted = values - np.max(values)
    return normalize_belief(np.exp(shifted))


def fuse_beliefs(beliefs: Sequence[Sequence[float]], mode: str, eps: float = 1e-8) -> np.ndarray:
    """Apply the fixed latest/mean/geometric posterior fusion rules."""
    matrix = np.asarray([normalize_belief(item) for item in beliefs], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 16 or len(matrix) == 0:
        raise ValueError("belief collection must be non-empty and 16-D")
    if mode == "latest":
        return normalize_belief(matrix[-1])
    if mode == "mean":
        return normalize_belief(matrix.mean(axis=0))
    if mode == "geometric":
        log_values = np.mean(np.log(np.maximum(matrix, eps)), axis=0)
        return normalize_belief(np.exp(log_values - np.max(log_values)))
    raise ValueError(f"unknown belief fusion mode: {mode}")


def top_k_belief(values: Sequence[float], k: int = 3) -> np.ndarray:
    """Keep and renormalize the fixed top-k posterior classes."""
    belief = normalize_belief(values)
    if k <= 0 or k > len(belief):
        raise ValueError("k must be in [1, 16]")
    indices = np.argsort(-belief, kind="mergesort")[:k]
    output = np.zeros_like(belief)
    output[indices] = belief[indices]
    return normalize_belief(output)


def select_min_risk(scores: Sequence[float]) -> int:
    """Select a Stay/p2/p3 action with deterministic Stay-first ties."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("action scores must have shape (3,)")
    return int(np.argmin(values))


def select_max_correctness(scores: Sequence[float]) -> int:
    """Select a Stay/p2/p3 action with deterministic Stay-first ties."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("action scores must have shape (3,)")
    return int(np.argmax(values))


def oracle_action(true_u2: Sequence[float]) -> int:
    """Frozen Fixed-first oracle action, including cache-order tie behavior."""
    values = np.asarray([0.0, *[float(item) for item in true_u2]], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("true_u2 must contain exactly two finite values")
    return int(np.argmax(values))


def binary_action(action: int) -> bool:
    """Return whether an action is a move (p2 or p3)."""
    if int(action) not in (0, 1, 2):
        raise ValueError("action must be 0, 1 or 2")
    return int(action) > 0


def action_diagnostics(actions: Sequence[int], truth: Sequence[int]) -> dict[str, Any]:
    """Summarize exact and binary action agreement."""
    predicted = np.asarray(actions, dtype=np.int64)
    target = np.asarray(truth, dtype=np.int64)
    if predicted.shape != target.shape:
        raise ValueError("action arrays must have equal shape")
    return {
        "count": int(len(target)),
        "exact_match": float(np.mean(predicted == target)) if len(target) else None,
        "binary_move_stay_match": float(np.mean((predicted > 0) == (target > 0)) if len(target) else None),
        "stay_count": int(np.sum(predicted == 0)),
        "p2_count": int(np.sum(predicted == 1)),
        "p3_count": int(np.sum(predicted == 2)),
    }


def masked_regression_loss(prediction: Any, target: Any, labels: Any) -> Any:
    """Smooth-L1 loss on the true activity head only (torch tensors)."""
    import torch

    row = torch.arange(prediction.shape[0], device=prediction.device)
    return torch.nn.functional.smooth_l1_loss(prediction[row, labels], target)


def masked_binary_loss(prediction: Any, target: Any, labels: Any) -> Any:
    """BCE loss on the true activity head only (torch tensors)."""
    import torch

    row = torch.arange(prediction.shape[0], device=prediction.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(prediction[row, labels], target)


def model_input_audit(features: np.ndarray, forbidden_values: Sequence[float]) -> None:
    """Test helper ensuring forbidden target values are not concatenated as input."""
    if not np.isfinite(features).all():
        raise ValueError("model features contain non-finite values")
    forbidden = np.asarray(forbidden_values, dtype=np.float32)
    if forbidden.size and np.array_equal(features.reshape(-1)[: forbidden.size], forbidden):
        raise AssertionError("forbidden target values appear as model input")
