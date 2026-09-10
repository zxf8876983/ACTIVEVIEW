"""Shared read-only loaders and metrics for reduced12 NBV diagnostics.

The helpers intentionally load only policy Train/Val files.  Future candidate
evidence is exposed to callers as targets/oracle values, never as deployable
input by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
NUM_CLASSES = len(LABELS)
RADIUS_VALUES = (1.5, 2.0, 2.5, 3.0)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _align_cache(
    stage_rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray],
    d_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stage_index = {str(row["episode_id"]): index for index, row in enumerate(stage_rows)}
    output: list[dict[str, Any]] = []
    for row in d_rows:
        episode_id = str(row["episode_id"])
        if episode_id not in stage_index:
            raise ValueError(f"Stage-D episode is absent from Stage-C: {episode_id}")
        index = stage_index[episode_id]
        if int(cache["labels"][index]) != int(row["label_id"]):
            raise ValueError(f"label mismatch for {episode_id}")
        valid = np.flatnonzero(cache["candidate_mask"][index]).astype(int)
        cached_ids = cache["candidate_ids"][index, valid].astype(int).tolist()
        expected = [int(value) for value in stage_rows[index]["candidate_viewpoint_ids"]]
        if cached_ids != expected:
            raise ValueError(f"candidate identity mismatch for {episode_id}")
        item = dict(row)
        item["stage_index"] = index
        item["cache_index"] = index
        output.append(item)
    return output


def load_train_val() -> dict[str, Any]:
    """Load Stage-C/Stage-D rows and true evidence cache for Train and Val."""
    data_root = get_data_root()
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    stage_c_root = policy_root / "stage_c"
    cache_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    stage_c: dict[str, list[dict[str, Any]]] = {}
    stage_d: dict[str, list[dict[str, Any]]] = {}
    caches: dict[str, dict[str, np.ndarray]] = {}
    for split, cache_name in (("train", "train_candidate_true_logp.npz"), ("val", "val_all_candidate_true_logp.npz")):
        stage_c[split] = load_jsonl(stage_c_root / "features" / f"{split}.jsonl")
        stage_d[split] = load_jsonl(policy_root / "stage_d" / "features" / f"{split}.jsonl")
        caches[split] = read_npz(cache_root / cache_name)
        if len(stage_c[split]) != int(caches[split]["labels"].shape[0]):
            raise ValueError(f"{split} Stage-C/cache count mismatch")
        stage_d[split] = _align_cache(stage_c[split], caches[split], stage_d[split])
    predictions = {
        split: load_jsonl(stage_c_root / "predictions" / f"{split}_predictions.jsonl")
        for split in ("train", "val")
    }
    prediction_by_episode = {
        split: {str(row["episode_id"]): row for row in rows}
        for split, rows in predictions.items()
    }
    for split in ("train", "val"):
        for row in stage_d[split]:
            pred = prediction_by_episode[split].get(str(row["episode_id"]))
            if pred is None:
                raise ValueError(f"missing Stage-C prediction for {row['episode_id']}")
            row["stage_c_prediction"] = pred
    summary_path = stage_c_root / "stage_c_feature_summary.json"
    stats_path = stage_c_root / "stage_c_feature_stats.json"
    return {
        "data_root": data_root,
        "policy_root": policy_root,
        "stage_c_rows": stage_c,
        "stage_d_rows": stage_d,
        "caches": caches,
        "stats": json.loads(stats_path.read_text(encoding="utf-8")),
        "feature_summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def candidate_logp(cache: Mapping[str, np.ndarray], row_index: int, viewpoint_id: int) -> np.ndarray:
    ids = cache["candidate_ids"][row_index]
    matches = np.flatnonzero((ids == int(viewpoint_id)) & cache["candidate_mask"][row_index])
    if matches.size != 1:
        raise ValueError(f"viewpoint {viewpoint_id} is not uniquely legal at cache row {row_index}")
    return np.asarray(cache["candidate_logp"][row_index, int(matches[0])], dtype=np.float32)


def legal_ids(cache: Mapping[str, np.ndarray], row_index: int) -> list[int]:
    mask = np.asarray(cache["candidate_mask"][row_index], dtype=bool)
    return cache["candidate_ids"][row_index, mask].astype(int).tolist()


def gt_margin(logp: Sequence[float], label: int) -> float:
    values = np.asarray(logp, dtype=np.float64)
    others = np.delete(values, int(label))
    return float(values[int(label)] - np.max(others))


def viewpoint_coord(viewpoint_id: int) -> tuple[int, int]:
    value = int(viewpoint_id)
    return value // 8, value % 8


def lattice_distance(left: int, right: int) -> int:
    radius_left, az_left = viewpoint_coord(left)
    radius_right, az_right = viewpoint_coord(right)
    azimuth = abs(az_left - az_right)
    azimuth = min(azimuth, 8 - azimuth)
    return abs(radius_left - radius_right) + azimuth


def angular_neighbor(left: int, right: int) -> bool:
    radius_left, az_left = viewpoint_coord(left)
    radius_right, az_right = viewpoint_coord(right)
    return radius_left == radius_right and min(abs(az_left - az_right), 8 - abs(az_left - az_right)) == 1


def radial_neighbor(left: int, right: int) -> bool:
    radius_left, az_left = viewpoint_coord(left)
    radius_right, az_right = viewpoint_coord(right)
    return az_left == az_right and abs(radius_left - radius_right) == 1


def rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(order.size, dtype=np.float64)
    result[order] = np.arange(order.size, dtype=np.float64)
    return result


def correlation(left: Sequence[float], right: Sequence[float], *, spearman: bool = False) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if spearman:
        x, y = rank(x), rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def classification(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=np.int64)
    outputs = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(targets, outputs):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": support, "recall": recall, "f1": f1}
    return {
        "n": int(targets.size),
        "accuracy": float(np.mean(targets == outputs)) if targets.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def terminal_from_actions(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], actions: Sequence[int],
) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    moves = 0
    for row, action in zip(rows, actions):
        cache_index = int(row["cache_index"])
        label = int(row["label_id"])
        labels.append(label)
        if int(action) < 0:
            logp = np.asarray(row["s1_feature"][256:268], dtype=np.float32)
        else:
            logp = candidate_logp(cache, cache_index, int(action))
            moves += 1
        predictions.append(int(np.argmax(logp)))
    result = classification(labels, predictions)
    result["move_rate"] = float(moves / len(rows)) if rows else 0.0
    result["stay_rate"] = 1.0 - result["move_rate"]
    return result


def candidate_score_map(prediction: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(viewpoint): float(score)
        for viewpoint, score in zip(prediction["candidate_viewpoint_ids"], prediction["predicted_utilities"])
    }


def safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


__all__ = [
    "LABELS", "NUM_CLASSES", "RADIUS_VALUES", "angular_neighbor", "candidate_logp",
    "candidate_score_map", "classification", "correlation", "gt_margin", "lattice_distance",
    "legal_ids", "load_train_val", "rank", "radial_neighbor", "terminal_from_actions", "viewpoint_coord",
]
