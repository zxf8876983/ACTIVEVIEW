"""EXP016 second-step gate/candidate error decomposition primitives.

This module is deliberately analysis-only.  It materializes counterfactual
second-step decisions after the frozen Stage C-v0 first decision, using true
U2 values only for offline oracle branches.  No learned-model computation is
performed here and no Test data is accepted by the public analysis entry
point.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_d_evaluation import (
    _selected_row,
    _by_id,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_policy import order_candidates, trajectory_cost


EXP016_VARIANTS = (
    "EXP014",
    "OracleGate_LearnedCandidate",
    "LearnedGate_OracleCandidate",
    "FixedFirstSecondStepOracle",
)


def validate_exp016_split(split: str) -> None:
    """Reject every split except the explicitly frozen Val analysis split."""
    if str(split).lower() != "val":
        raise ValueError("EXP016 accepts Val only; Test is locked")


def _index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in indexed:
            raise ValueError(f"Duplicate {name} episode_id: {episode_id}")
        indexed[episode_id] = row
    return indexed


def _finite_values(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not np.isfinite(np.asarray(result, dtype=np.float64)).all():
        raise ValueError(f"{name} must be a non-empty finite sequence")
    return result


def _candidate_decision(
    values: Sequence[float],
    candidate_ids: Sequence[int],
    geodesics: Sequence[float],
) -> tuple[int, float]:
    """Select one candidate with the Stage D utility/geodesic/ID tie-break."""
    ordered = order_candidates(values, candidate_ids, geodesics)
    if not ordered:
        raise ValueError("candidate set must not be empty")
    return int(ordered[0]), float(values[list(candidate_ids).index(ordered[0])])


def second_step_variant_decision(
    *,
    gate: str,
    candidate: str,
    learned_utilities: Sequence[float],
    true_utilities: Sequence[float],
    candidate_ids: Sequence[int],
    candidate_geodesics: Sequence[float],
) -> dict[str, Any]:
    """Return an offline counterfactual second-step decision.

    ``gate`` controls only Stay versus Move.  ``candidate`` controls only the
    p2/p3 identity after Move.  This separation is the central EXP016
    contract: an OracleCandidate branch never changes a learned Stay, and an
    OracleGate branch never chooses a candidate identity.
    """
    if gate not in {"learned", "oracle"}:
        raise ValueError(f"Unknown gate mode: {gate}")
    if candidate not in {"learned", "oracle"}:
        raise ValueError(f"Unknown candidate mode: {candidate}")
    ids = [int(value) for value in candidate_ids]
    geodesics = _finite_values(candidate_geodesics, "candidate_geodesics")
    learned = _finite_values(learned_utilities, "learned_utilities")
    true = _finite_values(true_utilities, "true_utilities")
    if not (len(ids) == len(geodesics) == len(learned) == len(true)):
        raise ValueError("learned/true candidate arrays must have equal lengths")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")

    gate_values = learned if gate == "learned" else true
    gate_best = max(gate_values)
    stays = gate_best <= 0.0
    if stays:
        return {
            "stays": True,
            "candidate_id": None,
            "gate_best_utility": float(gate_best),
            "gate_source": gate,
            "candidate_source": candidate,
        }

    selected_values = learned if candidate == "learned" else true
    selected_id, selected_value = _candidate_decision(selected_values, ids, geodesics)
    return {
        "stays": False,
        "candidate_id": selected_id,
        "selected_utility_for_ordering": selected_value,
        "gate_best_utility": float(gate_best),
        "gate_source": gate,
        "candidate_source": candidate,
    }


def build_exp016_variant_trajectories(
    *,
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    exp014_prediction_rows: Sequence[Mapping[str, Any]],
    gate: str,
    candidate: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Materialize one fixed-first EXP016 variant and decision counters."""
    v0 = _index(v0_prediction_rows, "v0 prediction")
    cache = _index(cache_rows, "Stage D cache")
    learned_predictions = _index(exp014_prediction_rows, "EXP014 prediction")
    output: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for record in stage_b_rows:
        episode_id = str(record["episode_id"])
        first = v0.get(episode_id)
        if first is None:
            raise ValueError(f"Missing v0 prediction for {episode_id}")
        if bool(first["predicted_stays"]):
            counters["v0_stay"] += 1
            output.append(_selected_row(record, None, moves=0, cost=0.0))
            continue

        counters["v0_move"] += 1
        first_id = int(first["predicted_candidate_viewpoint_id"])
        by_id = _by_id(record)
        if first_id not in by_id:
            raise ValueError(f"Frozen v0 selected unknown p1 for {episode_id}")
        p1_cost = float(by_id[first_id]["geodesic_distance_m"])
        cached = cache.get(episode_id)
        learned_prediction = learned_predictions.get(episode_id)
        if cached is None or learned_prediction is None:
            raise ValueError(f"Missing aligned second-step data for {episode_id}")

        ids = [int(value) for value in cached["remaining_candidate_ids"]]
        cached_ids = [int(value) for value in learned_prediction["remaining_candidate_ids"]]
        if ids != cached_ids:
            raise ValueError(f"EXP014/cache candidate IDs disagree for {episode_id}")
        learned_values = [float(value) for value in learned_prediction["predicted_utilities"]]
        true_values = [float(value) for value in cached["second_step_utility_targets"]]
        distances = [float(value) for value in cached["second_step_candidate_geodesic"]]
        decision = second_step_variant_decision(
            gate=gate,
            candidate=candidate,
            learned_utilities=learned_values,
            true_utilities=true_values,
            candidate_ids=ids,
            candidate_geodesics=distances,
        )
        if decision["stays"]:
            counters["second_stay"] += 1
            output.append(_selected_row(record, first_id, moves=1, cost=p1_cost))
            continue

        counters["second_move"] += 1
        selected_id = int(decision["candidate_id"])
        index = ids.index(selected_id)
        output.append(
            _selected_row(
                record,
                selected_id,
                moves=2,
                cost=trajectory_cost(p1_cost, distances[index]),
            )
        )
    return output, {key: int(value) for key, value in counters.items()}


def exp016_decision_diagnostics(
    *,
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    exp014_prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize second-step gate and candidate decisions on frozen v0-Move."""
    v0 = _index(v0_prediction_rows, "v0 prediction")
    cache = _index(cache_rows, "Stage D cache")
    learned_predictions = _index(exp014_prediction_rows, "EXP014 prediction")
    total = oracle_stay = learned_stay = 0
    stay_tp = stay_fp = stay_fn = stay_tn = 0
    oracle_gate_candidate_hits = 0
    oracle_gate_candidate_total = 0
    learned_gate_candidate_hits = 0
    learned_gate_candidate_total = 0

    for episode_id, first in v0.items():
        if bool(first["predicted_stays"]):
            continue
        cached = cache.get(episode_id)
        prediction = learned_predictions.get(episode_id)
        if cached is None or prediction is None:
            raise ValueError(f"Missing aligned second-step data for {episode_id}")
        ids = [int(value) for value in cached["remaining_candidate_ids"]]
        learned_ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        learned_values = [float(value) for value in prediction["predicted_utilities"]]
        true_values = [float(value) for value in cached["second_step_utility_targets"]]
        distances = [float(value) for value in cached["second_step_candidate_geodesic"]]
        if ids != learned_ids:
            raise ValueError(f"EXP014/cache candidate IDs disagree for {episode_id}")
        learned_id, _ = _candidate_decision(learned_values, ids, distances)
        oracle_id, _ = _candidate_decision(true_values, ids, distances)
        learned_is_stay = max(learned_values) <= 0.0
        oracle_is_stay = max(true_values) <= 0.0
        total += 1
        learned_stay += int(learned_is_stay)
        oracle_stay += int(oracle_is_stay)
        stay_tp += int(learned_is_stay and oracle_is_stay)
        stay_fp += int(learned_is_stay and not oracle_is_stay)
        stay_fn += int(not learned_is_stay and oracle_is_stay)
        stay_tn += int(not learned_is_stay and not oracle_is_stay)
        if not oracle_is_stay:
            oracle_gate_candidate_total += 1
            oracle_gate_candidate_hits += int(learned_id == oracle_id)
        if not learned_is_stay:
            learned_gate_candidate_total += 1
            learned_gate_candidate_hits += int(learned_id == oracle_id)

    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "eligible_v0_move_episode_count": total,
        "learned_second_step_stay_rate": _rate(learned_stay, total),
        "oracle_second_step_stay_rate": _rate(oracle_stay, total),
        "stay_precision": _rate(stay_tp, stay_tp + stay_fp),
        "stay_recall": _rate(stay_tp, stay_tp + stay_fn),
        "move_precision": _rate(stay_tn, stay_tn + stay_fn),
        "move_recall": _rate(stay_tn, stay_tn + stay_fp),
        "stay_move_confusion": {
            "learned_stay_oracle_stay": stay_tp,
            "learned_stay_oracle_move": stay_fp,
            "learned_move_oracle_stay": stay_fn,
            "learned_move_oracle_move": stay_tn,
        },
        "candidate_exact_hit": {
            "oracle_gate_move_denominator": oracle_gate_candidate_total,
            "oracle_gate_move_rate": _rate(oracle_gate_candidate_hits, oracle_gate_candidate_total),
            "learned_gate_move_denominator": learned_gate_candidate_total,
            "learned_gate_move_rate": _rate(learned_gate_candidate_hits, learned_gate_candidate_total),
        },
    }


def summarize_variant_rows(
    rows: Sequence[Mapping[str, Any]], categories: Sequence[str]
) -> dict[str, Any]:
    """Expose the canonical Stage D metric summary for one variant."""
    return summarize_trajectory_rows(rows, categories)
