"""EXP018 executed-candidate gate alignment diagnostics.

The helper keeps the Stage C-v0 first decision and EXP014 candidate ranking
frozen.  True second-step utilities are used only to form the offline
executed-candidate gate target; they never choose a candidate for a learned
policy branch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_d_error_decomposition import (
    _index,
    validate_exp016_episode_alignment,
)
from activeview.active_view.stage_d_evaluation import _by_id, _selected_row
from activeview.active_view.stage_d_policy import order_candidates, trajectory_cost


def executed_candidate_decision(
    *,
    learned_utilities: Sequence[float],
    true_utilities: Sequence[float],
    candidate_ids: Sequence[int],
    candidate_geodesics: Sequence[float],
) -> dict[str, Any]:
    """Select the learned candidate and evaluate its true U2 offline.

    ``order_candidates`` is the sole source of learned candidate ordering.
    The true utilities are only looked up at that already-selected ID for the
    executed-candidate gate target.
    """
    ids = [int(value) for value in candidate_ids]
    learned = np.asarray(learned_utilities, dtype=np.float64)
    true = np.asarray(true_utilities, dtype=np.float64)
    geodesics = np.asarray(candidate_geodesics, dtype=np.float64)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be non-empty and unique")
    if not (learned.ndim == true.ndim == geodesics.ndim == 1):
        raise ValueError("candidate arrays must be one-dimensional")
    if not (len(ids) == learned.size == true.size == geodesics.size):
        raise ValueError("candidate arrays must have equal lengths")
    if not np.isfinite(learned).all() or not np.isfinite(true).all() or not np.isfinite(geodesics).all():
        raise ValueError("candidate arrays must be finite")
    ordered = order_candidates(learned.tolist(), ids, geodesics.tolist())
    selected_id = int(ordered[0])
    selected_index = ids.index(selected_id)
    selected_true_utility = float(true[selected_index])
    return {
        "learned_candidate_id": selected_id,
        "executed_true_utility": selected_true_utility,
        "any_positive": bool(float(true.max()) > 0.0),
        "executed_positive": bool(selected_true_utility > 0.0),
        "candidate_ids": ids,
        "candidate_geodesics": geodesics.tolist(),
    }


def build_executed_candidate_oracle_gate_trajectories(
    *,
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    exp014_prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Build the offline executed-candidate oracle-gate trajectories.

    Stage C-v0 Stay episodes remain at ``s0``.  For v0-Move episodes, p1 is
    frozen and the learned EXP014 ranking chooses ``c_hat``.  Only the true
    utility at ``c_hat`` decides whether the second step is taken.
    """
    v0 = _index(v0_prediction_rows, "v0 prediction")
    cache = _index(cache_rows, "Stage D cache")
    predictions = _index(exp014_prediction_rows, "EXP014 prediction")
    trajectories: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for record in stage_b_rows:
        episode_id = str(record["episode_id"])
        first = v0.get(episode_id)
        if first is None:
            raise ValueError(f"Missing v0 prediction for {episode_id}")
        if bool(first["predicted_stays"]):
            counters["v0_stay"] += 1
            trajectories.append(_selected_row(record, None, moves=0, cost=0.0))
            continue

        counters["v0_move"] += 1
        first_id = int(first["predicted_candidate_viewpoint_id"])
        by_id = _by_id(record)
        if first_id not in by_id:
            raise ValueError(f"Frozen v0 selected unknown p1 for {episode_id}")
        cached = cache.get(episode_id)
        prediction = predictions.get(episode_id)
        if cached is None or prediction is None:
            raise ValueError(f"Missing aligned second-step data for {episode_id}")
        ids = [int(value) for value in cached["remaining_candidate_ids"]]
        prediction_ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        if ids != prediction_ids:
            raise ValueError(f"EXP014/cache candidate IDs disagree for {episode_id}")
        decision = executed_candidate_decision(
            learned_utilities=prediction["predicted_utilities"],
            true_utilities=cached["second_step_utility_targets"],
            candidate_ids=ids,
            candidate_geodesics=cached["second_step_candidate_geodesic"],
        )
        decision["episode_id"] = episode_id
        decisions.append(decision)
        p1_cost = float(by_id[first_id]["geodesic_distance_m"])
        if decision["executed_positive"]:
            counters["second_move"] += 1
            selected_id = int(decision["learned_candidate_id"])
            second_index = ids.index(selected_id)
            trajectories.append(
                _selected_row(
                    record,
                    selected_id,
                    moves=2,
                    cost=trajectory_cost(p1_cost, float(cached["second_step_candidate_geodesic"][second_index])),
                )
            )
        else:
            counters["second_stay"] += 1
            trajectories.append(_selected_row(record, first_id, moves=1, cost=p1_cost))
    return trajectories, decisions, {key: int(value) for key, value in counters.items()}


def summarize_target_alignment(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize any-positive versus executed-candidate gate targets."""
    any_positive = sum(bool(row["any_positive"]) for row in decisions)
    executed_positive = sum(bool(row["executed_positive"]) for row in decisions)
    combinations = {
        "any_stay_executed_stay": sum(not row["any_positive"] and not row["executed_positive"] for row in decisions),
        "any_move_executed_move": sum(row["any_positive"] and row["executed_positive"] for row in decisions),
        "any_move_executed_stay": sum(row["any_positive"] and not row["executed_positive"] for row in decisions),
        "any_stay_executed_move": sum(not row["any_positive"] and row["executed_positive"] for row in decisions),
    }
    mismatch = combinations["any_move_executed_stay"]
    if combinations["any_stay_executed_move"]:
        raise ValueError("Executed-positive gate target cannot exceed any-positive target")
    return {
        "eligible_episode_count": len(decisions),
        "any_positive_count": int(any_positive),
        "executed_positive_count": int(executed_positive),
        "combinations": {key: int(value) for key, value in combinations.items()},
        "ranking_induced_gate_mismatch_count": int(mismatch),
        "ranking_induced_gate_mismatch_rate": float(mismatch / any_positive) if any_positive else 0.0,
    }


def validate_exp018_episode_alignment(
    *,
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    exp014_prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Reuse EXP016's exact Val episode-universe check for EXP018."""
    return validate_exp016_episode_alignment(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_prediction_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_prediction_rows,
    )
