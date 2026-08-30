"""Train-only scalar gate calibration for Stage D EXP017.

EXP017 deliberately does not train a model.  It calibrates the strict
``gate_score > tau`` Move/Stay boundary of the frozen EXP014 second-step
predictions, then applies the frozen threshold to Val exactly once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_d_error_decomposition import (
    _index,
)
from activeview.active_view.stage_d_policy import order_candidates


NEAR_ZERO = 1e-12


def validate_exp017_split(split: str, expected: str) -> None:
    """Reject any split other than the explicitly requested phase split."""
    if str(split).lower() != str(expected).lower():
        raise ValueError(f"EXP017 requires {expected} rows; received {split}")


def _explicit_split(row: Mapping[str, Any]) -> str | None:
    value = row.get("policy_split")
    return None if value is None else str(value).lower()


def _assert_rows_for_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    invalid = sorted(
        {
            value
            for row in rows
            if (value := _explicit_split(row)) is not None and value != split
        }
    )
    if invalid:
        raise ValueError(f"{name} contains non-{split} rows: {invalid}")


def gate_score(prediction_row: Mapping[str, Any]) -> float:
    """Return the maximum frozen EXP014 predicted U2 score."""
    values = np.asarray(prediction_row["predicted_utilities"], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("predicted_utilities must be a non-empty finite vector")
    return float(values.max())


def oracle_move_label(cache_row: Mapping[str, Any]) -> bool:
    """Return the offline Train/Val diagnostic label from true U2 only."""
    values = np.asarray(cache_row["second_step_utility_targets"], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("second_step_utility_targets must be a non-empty finite vector")
    return bool(float(values.max()) > 0.0)


def _aligned_gate_examples(
    prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate prediction/cache IDs and return score plus oracle-label arrays."""
    _assert_rows_for_split(prediction_rows, split, "EXP014 predictions")
    _assert_rows_for_split(cache_rows, split, "Stage D cache")
    predictions = _index(prediction_rows, "EXP014 prediction")
    cache = _index(cache_rows, "Stage D cache")
    if set(predictions) != set(cache):
        missing = sorted(set(cache) - set(predictions))
        extra = sorted(set(predictions) - set(cache))
        raise ValueError(f"EXP014/cache episode IDs mismatch; missing={missing[:5]} extra={extra[:5]}")

    scores: list[float] = []
    labels: list[bool] = []
    for episode_id in cache:
        prediction = predictions[episode_id]
        cached = cache[episode_id]
        prediction_ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        cache_ids = [int(value) for value in cached["remaining_candidate_ids"]]
        predicted_values = prediction["predicted_utilities"]
        true_values = cached["second_step_utility_targets"]
        if prediction_ids != cache_ids:
            raise ValueError(f"EXP014/cache candidate IDs disagree for {episode_id}")
        if not (len(predicted_values) == len(true_values) == len(cache_ids)):
            raise ValueError(f"EXP014/cache candidate arrays disagree for {episode_id}")
        scores.append(gate_score(prediction))
        labels.append(oracle_move_label(cached))
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def _threshold_candidates(scores: Sequence[float]) -> list[float]:
    """Build deterministic strict-`>` decision boundaries from finite scores."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("threshold scores must be a non-empty finite vector")
    unique = np.unique(values)
    candidates: set[float] = {0.0}
    candidates.add(float(np.nextafter(unique[0], -np.inf)))
    candidates.add(float(np.nextafter(unique[-1], np.inf)))
    for left, right in zip(unique[:-1], unique[1:]):
        midpoint = float(left / 2.0 + right / 2.0)
        if not np.isfinite(midpoint) or midpoint <= left or midpoint >= right:
            midpoint = float(np.nextafter(left, right))
        candidates.add(midpoint)
    return sorted(candidates)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def gate_metrics(
    scores: Sequence[float], oracle_moves: Sequence[bool], tau: float
) -> dict[str, Any]:
    """Compute the complete binary gate diagnostics for one strict threshold."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(oracle_moves, dtype=bool)
    if values.ndim != 1 or labels.ndim != 1 or values.size != labels.size or values.size == 0:
        raise ValueError("scores and oracle_moves must be aligned non-empty vectors")
    if not np.isfinite(values).all() or not np.isfinite(float(tau)):
        raise ValueError("scores and tau must be finite")
    moves = values > float(tau)
    true_moves = labels
    true_stays = ~labels
    tp = int(np.sum(moves & true_moves))
    fp = int(np.sum(moves & true_stays))
    fn = int(np.sum(~moves & true_moves))
    tn = int(np.sum(~moves & true_stays))
    move_precision = _safe_ratio(tp, tp + fp)
    move_recall = _safe_ratio(tp, tp + fn)
    move_f1 = _safe_ratio(2 * move_precision * move_recall, move_precision + move_recall)
    stay_precision = _safe_ratio(tn, tn + fn)
    stay_recall = _safe_ratio(tn, tn + fp)
    balanced_accuracy = 0.5 * (move_recall + stay_recall)
    return {
        "tau": float(tau),
        "episode_count": int(values.size),
        "accuracy": _safe_ratio(tp + tn, int(values.size)),
        "move_rate": _safe_ratio(int(np.sum(moves)), int(values.size)),
        "stay_rate": _safe_ratio(int(np.sum(~moves)), int(values.size)),
        "move_precision": move_precision,
        "move_recall": move_recall,
        "move_f1": move_f1,
        "stay_precision": stay_precision,
        "stay_recall": stay_recall,
        "balanced_accuracy": balanced_accuracy,
        "confusion": {
            "learned_stay_oracle_stay": tn,
            "learned_stay_oracle_move": fn,
            "learned_move_oracle_stay": fp,
            "learned_move_oracle_move": tp,
        },
    }


def select_train_threshold(
    train_scores: Sequence[float], train_oracle_moves: Sequence[bool]
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Select one tau using only Train balanced accuracy and fixed tie rules."""
    candidates = _threshold_candidates(train_scores)
    evaluations = [gate_metrics(train_scores, train_oracle_moves, tau) for tau in candidates]
    selected = max(
        evaluations,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["move_f1"]),
            -abs(float(item["tau"])),
            float(item["tau"]),
        ),
    )
    return float(selected["tau"]), selected, evaluations


def calibrate_train_threshold(
    train_prediction_rows: Sequence[Mapping[str, Any]],
    train_cache_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit the EXP017 threshold from Train rows only and return its artifact."""
    validate_exp017_split("train", "train")
    scores, labels = _aligned_gate_examples(train_prediction_rows, train_cache_rows, split="train")
    tau, selected, evaluations = select_train_threshold(scores, labels)
    return {
        "experiment_id": "EXP017",
        "split": "train",
        "selected_tau": tau,
        "train_gate_metrics": selected,
        "threshold_candidate_count": len(evaluations),
        "threshold_selection_rule": {
            "decision": "Move iff gate_score > tau",
            "primary": "maximize Train gate balanced accuracy",
            "tie_break": [
                "greater Train Move F1",
                "threshold closest to zero",
                "numerically larger tau",
            ],
            "candidates": "unique finite scores, adjacent midpoints, tau=0, all-Move and all-Stay boundaries",
        },
        "test_used": False,
    }


def build_thresholded_prediction_rows(
    exp014_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    tau: float,
) -> list[dict[str, Any]]:
    """Apply a frozen threshold while retaining EXP014 learned candidate ranking."""
    predictions = _index(exp014_prediction_rows, "EXP014 prediction")
    cache = _index(cache_rows, "Stage D cache")
    if set(predictions) != set(cache):
        raise ValueError("EXP014/cache episode IDs mismatch")
    output: list[dict[str, Any]] = []
    for episode_id, prediction in predictions.items():
        cached = cache[episode_id]
        ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        cache_ids = [int(value) for value in cached["remaining_candidate_ids"]]
        values = [float(value) for value in prediction["predicted_utilities"]]
        distances = [float(value) for value in cached["second_step_candidate_geodesic"]]
        if ids != cache_ids or len(values) != len(ids) or len(distances) != len(ids):
            raise ValueError(f"EXP014/cache candidate arrays disagree for {episode_id}")
        ordered = order_candidates(values, ids, distances)
        if not ordered:
            raise ValueError(f"Empty candidate ranking for {episode_id}")
        score = max(values)
        stays = score <= float(tau)
        updated = dict(prediction)
        updated["predicted_stays"] = bool(stays)
        updated["predicted_candidate_viewpoint_id"] = None if stays else int(ordered[0])
        updated["max_predicted_utility"] = float(score)
        updated["calibration_tau"] = float(tau)
        output.append(updated)
    return output


def candidate_identity_audit(
    zero_rows: Sequence[Mapping[str, Any]], calibrated_rows: Sequence[Mapping[str, Any]]
) -> dict[str, int | bool]:
    """Confirm a gate-only change never changes candidate identity when both move."""
    zero = _index(zero_rows, "EXP014 tau=0 prediction")
    calibrated = _index(calibrated_rows, "EXP017 calibrated prediction")
    if set(zero) != set(calibrated):
        raise ValueError("EXP014/EXP017 prediction IDs mismatch")
    both_move = 0
    mismatches = 0
    for episode_id in zero:
        if not bool(zero[episode_id]["predicted_stays"]) and not bool(calibrated[episode_id]["predicted_stays"]):
            both_move += 1
            mismatches += int(
                int(zero[episode_id]["predicted_candidate_viewpoint_id"])
                != int(calibrated[episode_id]["predicted_candidate_viewpoint_id"])
            )
    return {
        "both_move_episode_count": both_move,
        "candidate_identity_mismatch_count": mismatches,
        "candidate_identity_unchanged": mismatches == 0,
    }


def binary_roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Compute ROC-AUC with average ranks for ties; return None for one class."""
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[truth].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def binary_average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Compute deterministic threshold-group average precision."""
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    cumulative = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        cumulative += int(truth[order[start:end]].sum())
        recall = cumulative / positives
        precision = cumulative / end
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


def load_calibration_artifact(path: Path) -> dict[str, Any]:
    """Load and validate a frozen Train calibration artifact for Val use."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != "EXP017" or artifact.get("split") != "train":
        raise ValueError("Invalid EXP017 Train calibration artifact")
    tau = artifact.get("selected_tau")
    if not isinstance(tau, (int, float)) or not math.isfinite(float(tau)):
        raise ValueError("Calibration artifact selected_tau must be finite")
    if artifact.get("test_used") is not False:
        raise ValueError("Calibration artifact must record test_used=false")
    return artifact
