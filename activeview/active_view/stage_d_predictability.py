"""Pure helpers for the EXP028 oracle-action predictability audit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_NAMES = ("Stay", "p2", "p3")
MARGIN_BINS = ((0.0, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf")))


def oracle_action_index(true_utilities: Sequence[float]) -> int:
    """Match the frozen Fixed-first oracle (Stay first, cache-order ties)."""
    values = np.asarray([0.0, *[float(value) for value in true_utilities]], dtype=np.float64)
    if values.ndim != 1 or values.size not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("true_utilities must contain one or two finite values")
    return int(np.argmax(values))


def oracle_margin(true_utilities: Sequence[float]) -> dict[str, float | None]:
    """Return deterministic oracle utility margins for one episode."""
    values = np.asarray([0.0, *[float(value) for value in true_utilities]], dtype=np.float64)
    if values.size not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("true_utilities must contain one or two finite values")
    ordered = np.sort(values)[::-1]
    best_index = oracle_action_index(true_utilities)
    candidate_margin = None
    if best_index > 0 and len(true_utilities) == 2:
        candidate_margin = float(abs(float(true_utilities[0]) - float(true_utilities[1])))
    return {
        "margin_1": float(ordered[0] - ordered[1]),
        "move_gain": float(max([0.0, *[float(value) for value in true_utilities]])),
        "candidate_margin": candidate_margin,
        "best_utility": float(ordered[0]),
    }


def margin_bin_index(value: float) -> int:
    """Return the fixed top-1 margin bin index."""
    margin = float(value)
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    for index, (lower, upper) in enumerate(MARGIN_BINS):
        if lower <= margin < upper:
            return index
    return len(MARGIN_BINS) - 1


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy normalized to [0, 1], with deterministic zero handling."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0.0):
        raise ValueError("probabilities must be a non-negative vector")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    values = values / total
    nonzero = values[values > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / float(np.log(values.size)) if values.size > 1 else 0.0
    return float(np.clip(entropy, 0.0, 1.0))


def majority_action(labels: Sequence[int]) -> int:
    """Majority vote with frozen tie priority Stay > p2 > p3."""
    counts = Counter(int(value) for value in labels)
    return max(ACTION_NAMES and range(3), key=lambda value: (counts[value], -value))


def neighbor_agreement(
    train_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    val_labels: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Compute 3-way and binary agreement from a Train-only neighbor index."""
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[1] < k:
        raise ValueError("neighbor_indices does not contain requested k")
    neighbors = np.asarray(train_labels, dtype=np.int64)[neighbor_indices[:, :k]]
    pred = np.asarray([majority_action(row.tolist()) for row in neighbors], dtype=np.int64)
    labels = np.asarray(val_labels, dtype=np.int64)
    if labels.shape != (neighbors.shape[0],):
        raise ValueError("val_labels shape mismatch")
    return {
        "k": int(k),
        "three_way_accuracy": float(np.mean(pred == labels)),
        "binary_accuracy": float(np.mean((pred > 0) == (labels > 0))),
    }


def neighbor_entropy(neighbor_labels: Sequence[int]) -> dict[str, float]:
    """Return 3-way and binary normalized entropy for one neighborhood."""
    labels = np.asarray([int(value) for value in neighbor_labels], dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    binary = np.asarray([counts[0], counts[1] + counts[2]], dtype=np.float64)
    return {"three_way": normalized_entropy(counts), "binary": normalized_entropy(binary)}


def quantized_context_key(row: Mapping[str, Any], precision: int = 3) -> tuple[Any, ...]:
    """Build a deterministic scene/region/geometry key for cross-motion audit."""
    geometry = np.asarray(row["second_step_candidate_geometry"], dtype=np.float64).reshape(-1)
    if not np.isfinite(geometry).all():
        raise ValueError("candidate geometry must be finite")
    return (
        str(row["scene_id"]),
        str(row["region"]),
        tuple(float(value) for value in np.round(geometry, precision)),
    )
