"""Frozen baseline policies for the final ActiveView evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _by_id(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(item["viewpoint_id"]): item for item in record["candidates"]}


def _selected_row(record: Mapping[str, Any], selected_id: int | None, *, moves: int, cost: float) -> dict[str, Any]:
    by_id = _by_id(record)
    current = record["current"]
    selected = current if selected_id is None else by_id[int(selected_id)]
    safe_utility = float(record["oracle"]["safe_oracle_utility"])
    selected_utility = 0.0 if selected_id is None else float(selected["utility"])
    return {
        "episode_id": str(record["episode_id"]), "record_id": str(record["record_id"]), "policy_split": str(record["policy_split"]),
        "scene_id": str(record["scene_id"]), "region": str(record["region"]), "label_id": int(record["label_id"]),
        "selected_viewpoint_id": int(current["viewpoint_id"] if selected_id is None else selected_id),
        "predicted_label_id": int(selected["predicted_label_id"]), "selected_true_utility": selected_utility,
        "safe_oracle_utility": safe_utility, "regret": safe_utility - selected_utility,
        "moves": int(moves), "trajectory_geodesic_cost_m": float(cost),
        "safe_oracle_stays": bool(record["oracle"]["safe_oracle_stays"]),
    }


def build_baseline_trajectories(stage_b_rows: Sequence[Mapping[str, Any]], v0_prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build NoMove and frozen Stage-C-v0 trajectories."""
    predictions = {str(row["episode_id"]): row for row in v0_prediction_rows}
    if len(predictions) != len(v0_prediction_rows):
        raise ValueError("Duplicate frozen-v0 prediction episode_id")
    no_move: list[dict[str, Any]] = []
    frozen_v0: list[dict[str, Any]] = []
    for record in stage_b_rows:
        prediction = predictions.get(str(record["episode_id"]))
        if prediction is None:
            raise ValueError(f"Missing frozen-v0 prediction for {record['episode_id']}")
        no_move.append(_selected_row(record, None, moves=0, cost=0.0))
        if bool(prediction["predicted_stays"]):
            frozen_v0.append(_selected_row(record, None, moves=0, cost=0.0))
            continue
        selected_id = int(prediction["predicted_candidate_viewpoint_id"])
        by_id = _by_id(record)
        if selected_id not in by_id:
            raise ValueError(f"Frozen-v0 selected unknown candidate {selected_id}")
        frozen_v0.append(_selected_row(record, selected_id, moves=1, cost=float(by_id[selected_id]["geodesic_distance_m"])))
    return {"NoMove": no_move, "FrozenStageCv0": frozen_v0}


def build_single_step_oracles(stage_b_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build frozen CandidateOracle and SafeOracle baselines."""
    candidate: list[dict[str, Any]] = []
    safe: list[dict[str, Any]] = []
    for record in stage_b_rows:
        by_id = _by_id(record)
        oracle = record["oracle"]
        candidate_id = int(oracle["candidate_oracle_viewpoint_id"])
        candidate.append(_selected_row(record, candidate_id, moves=1, cost=float(by_id[candidate_id]["geodesic_distance_m"])))
        if bool(oracle["safe_oracle_stays"]):
            safe.append(_selected_row(record, None, moves=0, cost=0.0))
        else:
            safe_id = int(oracle["safe_oracle_viewpoint_id"])
            safe.append(_selected_row(record, safe_id, moves=1, cost=float(by_id[safe_id]["geodesic_distance_m"])))
    return {"CandidateOracle": candidate, "SafeOracle": safe}


__all__ = ["build_baseline_trajectories", "build_single_step_oracles"]
