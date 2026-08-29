"""Top-K reachability audit for frozen Stage C-v0 Val predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_c_v3_teacher import load_jsonl
from activeview.active_view.utility_label_builder import file_sha256


SMALL_REGRET = 1e-2
TOP_K_VALUES = (1, 2, 3, 5)


def _load_utility_lookup(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = load_jsonl(path)
    lookup = {str(row["episode_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate Stage B episode_id in {path}")
    return lookup


def _ordered_candidates(prediction: Mapping[str, Any], utility: Mapping[str, Any]) -> list[tuple[int, float, float]]:
    ids = [int(value) for value in prediction["candidate_viewpoint_ids"]]
    predicted = [float(value) for value in prediction["predicted_utilities"]]
    if len(ids) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError(f"Invalid predicted utilities for {prediction.get('episode_id')}")
    by_id = {int(item["viewpoint_id"]): item for item in utility["candidates"]}
    if set(ids) != set(by_id):
        raise ValueError(f"Candidate IDs mismatch for {prediction.get('episode_id')}")
    return sorted(
        [(candidate_id, predicted[index], float(by_id[candidate_id]["geodesic_distance_m"])) for index, candidate_id in enumerate(ids)],
        key=lambda item: (-item[1], item[2], item[0]),
    )


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90))}


def run_topk_audit(*, predictions_path: Path, stage_b_utility_path: Path, output_path: Path, k_values: Sequence[int] = TOP_K_VALUES) -> dict[str, Any]:
    if tuple(k_values) != TOP_K_VALUES:
        raise ValueError("EXP013 fixes K to 1,2,3,5; do not sweep K")
    predictions = load_jsonl(predictions_path)
    utilities = _load_utility_lookup(stage_b_utility_path)
    if len({str(row["episode_id"]) for row in predictions}) != len(predictions):
        raise ValueError("Duplicate Stage C prediction episode_id")
    reports: dict[str, dict[str, Any]] = {}
    for k in k_values:
        candidate_hits = 0
        safe_hits = 0
        near_optimal_hits = 0
        positive_recalls = 0
        move_count = 0
        stay_count = 0
        regrets_all: list[float] = []
        regrets_move: list[float] = []
        for prediction in predictions:
            episode_id = str(prediction["episode_id"])
            utility = utilities.get(episode_id)
            if utility is None:
                raise ValueError(f"Missing Stage B utility for {episode_id}")
            ordered = _ordered_candidates(prediction, utility)
            top = ordered[: min(k, len(ordered))]
            top_ids = {item[0] for item in top}
            oracle = utility["oracle"]
            candidate_oracle_id = int(oracle["candidate_oracle_viewpoint_id"])
            candidate_hits += int(candidate_oracle_id in top_ids)
            safe_stays = bool(oracle["safe_oracle_stays"])
            safe_utility = float(oracle["safe_oracle_utility"])
            if safe_stays:
                stay_count += 1
                continue
            move_count += 1
            safe_id = int(oracle["safe_oracle_viewpoint_id"])
            safe_hits += int(safe_id in top_ids)
            by_id = {int(item["viewpoint_id"]): item for item in utility["candidates"]}
            near_optimal_hits += int(any(float(by_id[item[0]]["utility"]) >= safe_utility - SMALL_REGRET for item in top))
            positive_recalls += int(any(float(by_id[item[0]]["utility"]) > 0.0 for item in top))
            best_true = max((float(by_id[item[0]]["utility"]) for item in top), default=0.0)
            regrets_move.append(safe_utility - max(0.0, best_true))
            regrets_all.append(safe_utility - max(0.0, best_true))
        reports[str(k)] = {
            "k": int(k),
            "candidate_oracle_hit_rate": candidate_hits / len(predictions) if predictions else 0.0,
            "safe_oracle_move_count": move_count,
            "safe_oracle_stay_count": stay_count,
            "safe_oracle_hit_rate_move_only": safe_hits / move_count if move_count else 0.0,
            "near_optimal_hit_rate_move_only": near_optimal_hits / move_count if move_count else 0.0,
            "positive_candidate_recall_move_only": positive_recalls / move_count if move_count else 0.0,
            "topk_oracle_regret_move_only": _summary(regrets_move),
            "topk_oracle_regret_all_move_episodes": _summary(regrets_all),
        }
    result = {
        "protocol": "ACTIVEVIEW Stage C-v3 Top-K reachability audit",
        "diagnostic_only": True,
        "source_model": "frozen Stage C-v0 Set Ranker",
        "source_split": "val",
        "test_used": False,
        "near_optimal_epsilon": SMALL_REGRET,
        "k_values": list(TOP_K_VALUES),
        "episode_count": len(predictions),
        "reports": reports,
        "provenance": {
            "v0_val_predictions_sha256": file_sha256(predictions_path),
            "stage_b_val_utility_sha256": file_sha256(stage_b_utility_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
