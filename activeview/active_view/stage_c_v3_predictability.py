"""Train-to-Val nearest-neighbour predictability audit for Stage C-v3."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_c_v3_teacher import load_jsonl
from activeview.active_view.utility_label_builder import file_sha256


NEAR_ZERO = 1e-6
K_NEIGHBORS = 5
GEOMETRY_DIM = 11
SEMANTIC_DIM = 19


def _utility_lookup(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = load_jsonl(path)
    lookup = {str(row["episode_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate episode_id in {path}")
    return lookup


def _examples(feature_path: Path, utility_path: Path, regret_lookup: Mapping[str, float], labels: Sequence[str]) -> list[Dict[str, Any]]:
    utilities = _utility_lookup(utility_path)
    output: list[Dict[str, Any]] = []
    for feature in load_jsonl(feature_path):
        episode_id = str(feature["episode_id"])
        utility = utilities.get(episode_id)
        if utility is None:
            raise ValueError(f"Missing Stage B utility for {episode_id}")
        by_id = {int(item["viewpoint_id"]): item for item in utility["candidates"]}
        candidate_ids = [int(value) for value in feature["candidate_viewpoint_ids"]]
        geometry = np.asarray(feature["candidate_geometry"], dtype=np.float64)
        if geometry.shape != (len(candidate_ids), GEOMETRY_DIM) or not np.isfinite(geometry).all():
            raise ValueError(f"Invalid candidate geometry in {episode_id}")
        current = np.asarray(feature["current_feature"], dtype=np.float64)
        if current.shape != (275,) or not np.isfinite(current).all():
            raise ValueError(f"Invalid current feature in {episode_id}")
        for index, candidate_id in enumerate(candidate_ids):
            candidate = by_id.get(candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate ID mismatch in {episode_id}")
            label_id = int(feature["label_id"])
            output.append({
                "episode_id": episode_id,
                "record_id": str(feature["record_id"]),
                "label_id": label_id,
                "action_label": labels[label_id] if 0 <= label_id < len(labels) else str(label_id),
                "candidate_id": candidate_id,
                "geometry": geometry[index],
                "observable": np.concatenate([current[256:], geometry[index]]),
                "utility": float(candidate["utility"]),
                "episode_regret": float(regret_lookup.get(episode_id, 0.0)),
            })
    return output


def _normalize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std, np.stack([mean, std])


def _nearest_indices(train: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if len(train) < k:
        raise ValueError(f"Train reference has {len(train)} rows; k={k} is invalid")
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k, algorithm="auto")
        model.fit(train)
        return model.kneighbors(query, return_distance=False)
    except ImportError:
        # Small fallback for installations without scikit-learn.
        result = np.empty((len(query), k), dtype=np.int64)
        chunk = 512
        for start in range(0, len(query), chunk):
            block = query[start : start + chunk]
            distances = np.sum((block[:, None, :] - train[None, :, :]) ** 2, axis=2)
            result[start : start + len(block)] = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90))}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        if len(indices) > 1:
            result[indices] = result[indices].mean()
    return result


def _audit(train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]], key: str, k: int, disagreement_threshold: float) -> dict[str, Any]:
    train_x, val_x, stats = _normalize(
        np.stack([np.asarray(row[key], dtype=np.float64) for row in train]),
        np.stack([np.asarray(row[key], dtype=np.float64) for row in val]),
    )
    train_y = np.asarray([float(row["utility"]) for row in train], dtype=np.float64)
    val_y = np.asarray([float(row["utility"]) for row in val], dtype=np.float64)
    neighbors = _nearest_indices(train_x, val_x, k)
    neighbor_values = train_y[neighbors]
    means = neighbor_values.mean(axis=1)
    stds = neighbor_values.std(axis=1)
    disagreements = neighbor_values.max(axis=1) - neighbor_values.min(axis=1)
    predicted_positive = means > NEAR_ZERO
    true_positive = val_y > NEAR_ZERO
    sign_agreement = predicted_positive == true_positive
    sign_conflict = (neighbor_values > NEAR_ZERO).any(axis=1) & (neighbor_values <= NEAR_ZERO).any(axis=1)
    rows: list[Dict[str, Any]] = []
    for index, row in enumerate(val):
        rows.append({
            **row,
            "neighbor_mean": float(means[index]),
            "neighbor_std": float(stds[index]),
            "absolute_disagreement": float(disagreements[index]),
            "sign_agreement": bool(sign_agreement[index]),
            "sign_conflict": bool(sign_conflict[index]),
            "high_disagreement": bool(disagreements[index] >= disagreement_threshold),
        })
    return {
        "input_key": key,
        "input_dim": int(train_x.shape[1]),
        "k": k,
        "normalization": "Train mean/std only",
        "train_reference_count": len(train),
        "val_query_count": len(val),
        "train_mean": stats[0].tolist(),
        "train_std": stats[1].tolist(),
        "neighbor_utility_std": _distribution(stds),
        "absolute_utility_disagreement": _distribution(disagreements),
        "neighbor_mean_vs_true": {
            "mae": float(np.mean(np.abs(means - val_y))),
            "pearson": _pearson(means, val_y),
            "spearman": _pearson(_rank(means), _rank(val_y)),
        },
        "utility_sign_agreement_rate": float(np.mean(sign_agreement)),
        "same_neighborhood_sign_conflict_rate": float(np.mean(sign_conflict)),
        "high_disagreement_threshold_iqr": float(disagreement_threshold),
        "high_disagreement_rate": float(np.mean(disagreements >= disagreement_threshold)),
        "rows": rows,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "utility_sign_agreement_rate": float(np.mean([row["sign_agreement"] for row in rows])),
        "same_neighborhood_sign_conflict_rate": float(np.mean([row["sign_conflict"] for row in rows])),
        "high_disagreement_rate": float(np.mean([row["high_disagreement"] for row in rows])),
        "neighbor_utility_std": _distribution([float(row["neighbor_std"]) for row in rows]),
        "absolute_utility_disagreement": _distribution([float(row["absolute_disagreement"]) for row in rows]),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    regrets = np.asarray([float(row["episode_regret"]) for row in rows], dtype=np.float64)
    p50, p90 = np.percentile(regrets, [50, 90])
    groups: dict[str, list[Mapping[str, Any]]] = {"low_regret": [], "medium_regret": [], "high_regret": []}
    for row in rows:
        regret = float(row["episode_regret"])
        name = "low_regret" if regret <= p50 else "medium_regret" if regret <= p90 else "high_regret"
        groups[name].append(row)
    output = {name: _group_summary(values) | {"regret_thresholds": {"p50": float(p50), "p90": float(p90)}} for name, values in groups.items()}
    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row["action_label"])].append(row)
    output["action_class"] = {name: _group_summary(values) for name, values in sorted(by_action.items())}
    radius = np.asarray([float(np.asarray(row["geometry"])[9]) for row in rows])
    r1, r2 = np.percentile(radius, [33.333333, 66.666667])
    radius_groups = {"low": [], "medium": [], "high": []}
    for row, value in zip(rows, radius):
        radius_groups["low" if value <= r1 else "medium" if value <= r2 else "high"].append(row)
    output["candidate_radius_bin"] = {name: _group_summary(values) for name, values in radius_groups.items()}
    azimuth_bins = {"0_30": [], "30_60": [], "60_90": [], "90_120": [], "120_150": [], "150_180": []}
    for row in rows:
        geometry = np.asarray(row["geometry"], dtype=np.float64)
        angle = abs(float(np.degrees(np.arctan2(geometry[5], geometry[6]))))
        index = min(int(angle // 30), 5)
        names = list(azimuth_bins)
        azimuth_bins[names[index]].append(row)
    output["candidate_azimuth_bin"] = {name: _group_summary(values) for name, values in azimuth_bins.items()}
    return output


def run_predictability_audit(
    *, feature_root: Path, stage_b_root: Path, v0_predictions: Path, label_mapping: Path, output_path: Path, k: int = K_NEIGHBORS,
) -> dict[str, Any]:
    if k != K_NEIGHBORS:
        raise ValueError("EXP012 fixes k=5; do not sweep k")
    mapping = json.loads(label_mapping.read_text(encoding="utf-8"))
    labels = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    prediction_rows = load_jsonl(v0_predictions)
    predictions = {str(row["episode_id"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("Duplicate episode_id in frozen v0 predictions")
    val_feature_rows = load_jsonl(feature_root / "features/val.jsonl")
    val_episode_ids = {str(row["episode_id"]) for row in val_feature_rows}
    if set(predictions) != val_episode_ids:
        raise ValueError("Frozen v0 Val prediction coverage does not match Val features")
    train = _examples(feature_root / "features/train.jsonl", stage_b_root / "utility_labels/train.jsonl", {}, labels)
    val = _examples(feature_root / "features/val.jsonl", stage_b_root / "utility_labels/val.jsonl", {episode_id: float(row["regret"]) for episode_id, row in predictions.items()}, labels)
    train_utility = np.asarray([float(row["utility"]) for row in train], dtype=np.float64)
    disagreement_threshold = float(np.percentile(train_utility, 75) - np.percentile(train_utility, 25))
    geometry = _audit(train, val, "geometry", k, disagreement_threshold)
    observable = _audit(train, val, "observable", k, disagreement_threshold)
    result = {
        "protocol": "ACTIVEVIEW Stage C-v3 candidate utility predictability audit",
        "diagnostic_only": True, "split": "train_reference_to_val_query", "test_used": False,
        "k": k, "disagreement_threshold": {"source": "Train utility IQR", "value": disagreement_threshold},
        "geometry_only": {key: value for key, value in geometry.items() if key != "rows"},
        "observable_state_plus_geometry": {key: value for key, value in observable.items() if key != "rows"},
        "groups": {"geometry_only": _group_rows(geometry["rows"], labels), "observable_state_plus_geometry": _group_rows(observable["rows"], labels)},
        "provenance": {
            "train_feature_sha256": file_sha256(feature_root / "features/train.jsonl"),
            "val_feature_sha256": file_sha256(feature_root / "features/val.jsonl"),
            "train_utility_sha256": file_sha256(stage_b_root / "utility_labels/train.jsonl"),
            "val_utility_sha256": file_sha256(stage_b_root / "utility_labels/val.jsonl"),
            "v0_val_predictions_sha256": file_sha256(v0_predictions),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
