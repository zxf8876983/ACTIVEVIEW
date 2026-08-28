"""Stage B offline Utility labels and recognition-headroom statistics.

This module consumes already serialized Stage A Episodes and cached estimated
skeleton predictions.  It intentionally has no Habitat or scene-discovery
dependency: Stage A defines the complete Episode boundary.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


UTILITY_TOLERANCE = 1e-5
NEAR_ZERO_TOLERANCE = 1e-6


def file_sha256(path: Any) -> str:
    """Return a deterministic SHA-256 digest for a file path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Expected a finite float, got {value!r}")
    return result


def _entropy_from_log_probs(log_probs: np.ndarray) -> float:
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("log_probs must be a finite one-dimensional array")
    probabilities = np.exp(values)
    return _finite_float(-(probabilities * values).sum())


def _view_diagnostic(viewpoint_id: int, log_probs: np.ndarray, label_id: int) -> Dict[str, Any]:
    values = np.asarray(log_probs, dtype=np.float64)
    if label_id < 0 or label_id >= values.shape[0]:
        raise ValueError(f"label_id {label_id} is outside log-probability dimension {values.shape[0]}")
    predicted = int(np.argmax(values))
    return {
        "viewpoint_id": int(viewpoint_id),
        "logp_true": _finite_float(values[label_id]),
        "predicted_label_id": predicted,
        "correct": bool(predicted == label_id),
        "entropy": _entropy_from_log_probs(values),
    }


def build_utility_record(
    episode: Mapping[str, Any],
    log_probs_by_view: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    """Materialize one supervision-only Stage B record from view predictions."""
    label_id = int(episode["label_id"])
    current = episode["current_view"]
    current_id = int(current["viewpoint_id"])
    if current_id not in log_probs_by_view:
        raise ValueError(f"Missing finite current prediction for viewpoint {current_id}")
    current_diag = _view_diagnostic(current_id, log_probs_by_view[current_id], label_id)

    candidates: List[Dict[str, Any]] = []
    for source in episode["candidate_pool"]:
        viewpoint_id = int(source["viewpoint_id"])
        if viewpoint_id not in log_probs_by_view:
            raise ValueError(f"Missing finite candidate prediction for viewpoint {viewpoint_id}")
        diagnostic = _view_diagnostic(viewpoint_id, log_probs_by_view[viewpoint_id], label_id)
        utility = _finite_float(diagnostic["logp_true"] - current_diag["logp_true"])
        candidate = {
            **diagnostic,
            "geodesic_distance_m": _finite_float(source["geodesic_distance_m"]),
            "utility": utility,
        }
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"Episode {episode.get('episode_id')} has no candidates")

    oracle = min(
        candidates,
        key=lambda item: (-float(item["utility"]), float(item["geodesic_distance_m"]), int(item["viewpoint_id"])),
    )
    max_utility = float(oracle["utility"])
    safe_moves = max_utility > 0.0
    safe_id = int(oracle["viewpoint_id"]) if safe_moves else current_id
    safe_utility = max(0.0, max_utility)
    return {
        "episode_id": str(episode["episode_id"]),
        "record_id": str(episode["record_id"]),
        "policy_split": str(episode["policy_split"]),
        "scene_id": str(episode["scene_id"]),
        "region": str(episode["region"]),
        "label_id": label_id,
        "current": current_diag,
        "candidates": candidates,
        "oracle": {
            "candidate_oracle_viewpoint_id": int(oracle["viewpoint_id"]),
            "candidate_oracle_utility": max_utility,
            "safe_oracle_viewpoint_id": safe_id,
            "safe_oracle_utility": safe_utility,
            "safe_oracle_stays": not safe_moves,
        },
    }


def _classification_metrics(rows: Sequence[Tuple[int, int, float]], num_classes: int) -> Dict[str, Any]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    entropies: List[float] = []
    for target, prediction, entropy in rows:
        confusion[int(target), int(prediction)] += 1
        entropies.append(float(entropy))
    per_class: Dict[str, Dict[str, float | int]] = {}
    f1_values: List[float] = []
    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(class_id)] = {
            "support": int(support),
            "accuracy": float(recall),
            "f1": float(f1),
        }
    total = int(confusion.sum())
    return {
        "n": total,
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "per_class": per_class,
    }


def _distribution(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def _split_metrics(records: Sequence[Mapping[str, Any]], num_classes: int) -> Dict[str, Any]:
    policy_rows: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    region_rows: Dict[str, Dict[str, List[Tuple[int, int, float]]]] = defaultdict(lambda: defaultdict(list))
    max_utilities: List[float] = []
    pair_utilities: List[float] = []
    rescue_count = 0
    degradation_count = 0
    safe_preserved_count = 0
    positive_count = 0
    near_zero_count = 0
    negative_count = 0
    for record in records:
        target = int(record["label_id"])
        current = record["current"]
        candidates = record["candidates"]
        oracle = record["oracle"]
        by_id = {int(item["viewpoint_id"]): item for item in candidates}
        oracle_item = by_id[int(oracle["candidate_oracle_viewpoint_id"])]
        safe_item = current if int(oracle["safe_oracle_viewpoint_id"]) == int(current["viewpoint_id"]) else by_id[int(oracle["safe_oracle_viewpoint_id"])]
        selected = {
            "NoMove": current,
            "CandidateOracle": oracle_item,
            "SafeOracle": safe_item,
        }
        for policy, item in selected.items():
            row = (target, int(item["predicted_label_id"]), float(item["entropy"]))
            policy_rows[policy].append(row)
            region_rows[str(record["region"])][policy].append(row)
        utilities = [float(item["utility"]) for item in candidates]
        max_utility = max(utilities)
        max_utilities.append(max_utility)
        pair_utilities.extend(utilities)
        if max_utility > 0.0:
            positive_count += 1
        elif abs(max_utility) <= NEAR_ZERO_TOLERANCE:
            near_zero_count += 1
        else:
            negative_count += 1
        if not bool(current["correct"]) and bool(safe_item["correct"]):
            rescue_count += 1
        if bool(current["correct"]) and not bool(oracle_item["correct"]):
            degradation_count += 1
        if bool(current["correct"]) and bool(safe_item["correct"]):
            safe_preserved_count += 1

    policies = {
        policy: _classification_metrics(policy_rows.get(policy, []), num_classes)
        for policy in ("NoMove", "CandidateOracle", "SafeOracle")
    }
    no_move_accuracy = policies["NoMove"]["accuracy"]
    no_move_f1 = policies["NoMove"]["macro_f1"]
    for policy in ("CandidateOracle", "SafeOracle"):
        policies[policy]["accuracy_gain_vs_NoMove"] = policies[policy]["accuracy"] - no_move_accuracy
        policies[policy]["macro_f1_gain_vs_NoMove"] = policies[policy]["macro_f1"] - no_move_f1
    current_wrong = sum(1 for record in records if not bool(record["current"]["correct"]))
    return {
        "episode_count": len(records),
        "policies": policies,
        "per_region": {
            region: {policy: _classification_metrics(rows, num_classes) for policy, rows in policy_map.items()}
            for region, policy_map in sorted(region_rows.items())
        },
        "headroom": {
            "max_candidate_utility": _distribution(max_utilities),
            "positive_ratio": positive_count / len(records) if records else 0.0,
            "near_zero_ratio": near_zero_count / len(records) if records else 0.0,
            "negative_ratio": negative_count / len(records) if records else 0.0,
            "positive_headroom_episode_count": positive_count,
            "positive_headroom_episode_ratio": positive_count / len(records) if records else 0.0,
            "candidate_pair_utility": {
                **_distribution(pair_utilities),
                "positive_ratio": sum(value > 0.0 for value in pair_utilities) / len(pair_utilities) if pair_utilities else 0.0,
                "negative_ratio": sum(value < 0.0 for value in pair_utilities) / len(pair_utilities) if pair_utilities else 0.0,
                "near_zero_ratio": sum(abs(value) <= NEAR_ZERO_TOLERANCE for value in pair_utilities) / len(pair_utilities) if pair_utilities else 0.0,
            },
        },
        "rescue": {
            "current_wrong_count": current_wrong,
            "rescue_count": rescue_count,
            "rescue_rate_among_current_wrong": rescue_count / current_wrong if current_wrong else 0.0,
            "degradation_count": degradation_count,
            "degradation_rate": degradation_count / len(records) if records else 0.0,
            "current_correct_safe_correct_count": safe_preserved_count,
        },
    }


def summarize_utility_records(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    categories: Sequence[str],
) -> Dict[str, Any]:
    """Compute split and aggregate recognition/headroom metrics."""
    all_records = [record for split in ("train", "val", "test") for record in records_by_split.get(split, [])]
    return {
        split: _split_metrics(records_by_split.get(split, []), len(categories))
        for split in ("train", "val", "test")
    } | {"all": _split_metrics(all_records, len(categories))}
