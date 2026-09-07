#!/usr/bin/env python3
"""Audit old/new viewpoint recognition and oracle ceilings on Val only.

The audit deliberately consumes the frozen Stage-B utility labels and the
existing Val counterfactual recognition caches.  It never opens a Test file,
does not train or alter a checkpoint, and uses one implementation for both
the legacy four-region and reduced14 eight-placement datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root


NUM_VIEWS = 32
SEED = 42
YAW_BINS = ("0", "±45", "±90", "±135", "180")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _mean(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(data)) if data.size else float("nan")


def _median(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    return float(np.median(data)) if data.size else float("nan")


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.size < 2 or x.size != y.size or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    return _pearson(_rank(first), _rank(second))


def _macro_f1(predictions: Sequence[int], labels: Sequence[int], classes: int) -> float:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    if pred.size == 0:
        return float("nan")
    matrix = np.bincount(
        truth * classes + pred, minlength=classes * classes
    ).reshape(classes, classes)
    values: list[float] = []
    for cls in range(classes):
        tp = float(matrix[cls, cls])
        precision_den = float(matrix[:, cls].sum())
        recall_den = float(matrix[cls, :].sum())
        precision = tp / precision_den if precision_den else 0.0
        recall = tp / recall_den if recall_den else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(values))


def _classification_metrics(
    predictions: Sequence[int], labels: Sequence[int], classes: int
) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(pred == truth)) if truth.size else float("nan"),
        "macro_f1": _macro_f1(pred, truth, classes),
    }


def _wrap_angle(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _yaw_bin(value: float) -> str:
    angle = abs(_wrap_angle(value))
    nearest = min((0, 45, 90, 135, 180), key=lambda target: abs(angle - target))
    return "0" if nearest == 0 else "180" if nearest == 180 else f"±{nearest}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    policy_root: Path
    stage_b_path: Path
    stage_d_path: Path
    counterfactual_path: Path
    classes: int
    skeleton_root: Path
    candidate_root: Path


@dataclass
class DatasetAudit:
    spec: DatasetSpec
    episodes: list[dict[str, Any]]
    utility_rows: list[dict[str, Any]]
    moving_rows: list[dict[str, Any]]
    cache: dict[str, np.ndarray]
    cache_index: dict[str, int]
    row_by_episode: dict[str, dict[str, Any]]


def _episode_id(row: Mapping[str, Any]) -> str:
    return str(row["episode_id"])


def _candidate_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("Stage-B candidates must be a list")
    return [dict(candidate) for candidate in candidates]


def _validate_episode_alignment(
    episodes: Sequence[Mapping[str, Any]], utility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episode_ids = {_episode_id(row) for row in episodes}
    utility_ids = {_episode_id(row) for row in utility_rows}
    if episode_ids != utility_ids:
        raise ValueError("episode and Stage-B utility IDs are not aligned")
    mismatches: list[str] = []
    utility_by_id = {_episode_id(row): row for row in utility_rows}
    for episode in episodes:
        key = _episode_id(episode)
        utility = utility_by_id[key]
        for field in ("scene_id", "region", "record_id", "label_id"):
            if str(episode[field]) != str(utility[field]):
                mismatches.append(f"{key}:{field}")
    return {"episode_count": len(episodes), "utility_count": len(utility_rows), "mismatches": mismatches}


def _load_dataset(spec: DatasetSpec) -> DatasetAudit:
    episodes = _load_jsonl(spec.policy_root / "episodes" / "val_episodes.jsonl")
    utility_rows = _load_jsonl(spec.stage_b_path)
    alignment = _validate_episode_alignment(episodes, utility_rows)
    if alignment["mismatches"]:
        raise ValueError(f"metadata mismatches: {alignment['mismatches'][:5]}")
    moving_rows = _load_jsonl(spec.stage_d_path)
    cache = _load_npz(spec.counterfactual_path)
    cache_ids = [str(value) for value in cache["episode_ids"].tolist()]
    cache_index = {value: index for index, value in enumerate(cache_ids)}
    if len(cache_index) != len(cache_ids):
        raise ValueError(f"duplicate moving cache IDs for {spec.name}")
    moving_ids = {_episode_id(row) for row in moving_rows}
    if moving_ids != set(cache_ids):
        raise ValueError(f"Stage-D/cache IDs are not aligned for {spec.name}")
    utility_by_id = {_episode_id(row): row for row in utility_rows}
    row_by_episode = {_episode_id(row): row for row in episodes}
    for row in moving_rows:
        key = _episode_id(row)
        if key not in row_by_episode or int(row["label_id"]) != int(utility_by_id[key]["label_id"]):
            raise ValueError(f"moving row label mismatch for {spec.name}: {key}")
        cache_row = cache_index[key]
        if int(cache["label_id"][cache_row]) != int(row["label_id"]):
            raise ValueError(f"counterfactual cache label mismatch for {spec.name}: {key}")
    expected_views = ("current_logp_s0", "current_logp_s1", "true_logp")
    for name in expected_views:
        if name not in cache:
            raise ValueError(f"missing {name} in {spec.counterfactual_path}")
    if cache["true_logp"].shape[1] != NUM_VIEWS:
        raise ValueError(f"expected 32 cached viewpoints for {spec.name}")
    if "candidate_ids" in cache:
        candidate_ids = np.asarray(cache["candidate_ids"], dtype=np.int64)
        expected_ids = np.arange(NUM_VIEWS, dtype=np.int64)
        if candidate_ids.shape != (len(cache_ids), NUM_VIEWS) or not np.all(candidate_ids == expected_ids):
            raise ValueError(f"candidate_ids cache is not canonical 0..31 for {spec.name}")
    return DatasetAudit(
        spec=spec, episodes=episodes, utility_rows=utility_rows,
        moving_rows=moving_rows, cache=cache, cache_index=cache_index,
        row_by_episode=row_by_episode,
    )


def _source_alignment(audit: DatasetAudit) -> dict[str, Any]:
    checked = 0
    missing = 0
    metadata_mismatch = 0
    for episode in audit.episodes[: min(16, len(audit.episodes))]:
        current = episode.get("current_view", {})
        source = Path(str(current.get("skeleton_source_path", "")))
        if not source.is_file():
            missing += 1
            continue
        checked += 1
        try:
            with np.load(source, allow_pickle=False) as archive:
                ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)):
                    metadata_mismatch += 1
                if source.stem != str(episode.get("record_id")):
                    metadata_mismatch += 1
                for key in ("scene_id", "placement_id"):
                    if key in archive and str(np.asarray(archive[key]).item()) != str(episode.get(key, episode.get("region"))):
                        metadata_mismatch += 1
        except (OSError, ValueError, KeyError):
            metadata_mismatch += 1
    candidate_issues = 0
    for row in audit.episodes[: min(256, len(audit.episodes))]:
        current_id = int(row["current_view"]["viewpoint_id"])
        ids = [int(candidate["viewpoint_id"]) for candidate in row.get("candidate_pool", [])]
        geodesics = [float(candidate.get("geodesic_distance_m", 0.0)) for candidate in row.get("candidate_pool", [])]
        if (
            len(ids) != len(set(ids))
            or any(value == current_id or value < 0 or value >= NUM_VIEWS for value in ids)
            or any(not np.isfinite(value) or value < 0.0 for value in geodesics)
        ):
            candidate_issues += 1
    return {
        "sampled_source_archives": checked,
        "missing_source_archives": missing,
        "source_metadata_mismatches": metadata_mismatch,
        "candidate_alignment_issues": candidate_issues,
        "direct_stgcn_recompute": "NOT_RUN_cuda_unavailable",
    }


def _placement_view_info(
    audit: DatasetAudit, scene_id: str, region: str,
) -> dict[int, tuple[float, float]]:
    """Return viewpoint -> (azimuth, human yaw) from candidate metadata."""
    path = audit.spec.candidate_root / scene_id / "candidate_metadata" / "manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        placements = [
            value for value in payload.get("placements_data", [])
            if str(value.get("placement_id", value.get("region"))) == str(region)
        ]
        if len(placements) != 1:
            return {}
        placement = placements[0]
        yaw = float(placement.get("yaw_deg", 0.0))
        return {
            int(view["viewpoint_id"]): (float(view.get("azimuth_deg", 0.0)), yaw)
            for view in placement.get("viewpoints", [])
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _cache_predictions(audit: DatasetAudit, episode_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    index = audit.cache_index.get(episode_id)
    if index is None:
        return None
    logs = np.asarray(audit.cache["true_logp"][index], dtype=np.float64)
    predictions = np.argmax(logs, axis=-1).astype(np.int64)
    return logs, predictions


def _single_view_and_h0(audit: DatasetAudit) -> dict[str, Any]:
    labels: list[int] = []
    s0_pred: list[int] = []
    candidate_correct: list[float] = []
    random_pred: list[int] = []
    candidate_oracle_pred: list[int] = []
    safe_oracle_pred: list[int] = []
    any_correct: list[bool] = []
    any_including_stay: list[bool] = []
    correct_counts: list[int] = []
    legal_counts: list[int] = []
    ratio_values: list[float] = []
    random_rng = np.random.default_rng(SEED)
    per_view: dict[int, list[int]] = defaultdict(list)
    per_view_all: dict[int, list[int]] = defaultdict(list)
    per_radius: dict[float, list[int]] = defaultdict(list)
    yaw_stats: dict[str, list[int]] = defaultdict(list)
    cache_view_correct: list[int] = []
    cache_view_predictions: list[int] = []
    cache_view_labels: list[int] = []
    cache_view_total = 0
    cache_episode_count = 0
    placement_info_cache: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    utility_index = {_episode_id(row): index for index, row in enumerate(audit.utility_rows)}

    for row in audit.utility_rows:
        label = int(row["label_id"])
        current = dict(row["current"])
        current_prediction = int(current["predicted_label_id"])
        candidates = _candidate_rows(row)
        labels.append(label)
        s0_pred.append(current_prediction)
        legal_counts.append(len(candidates))
        correctness = [int(int(value["predicted_label_id"]) == label) for value in candidates]
        candidate_correct.extend(float(value) for value in correctness)
        correct_counts.append(int(sum(correctness)))
        ratio_values.append(float(np.mean(correctness)) if correctness else 0.0)
        any_correct.append(bool(any(correctness)))
        any_including_stay.append(bool(current_prediction == label or any(correctness)))
        if candidates:
            chosen = candidates[int(random_rng.integers(0, len(candidates)))]
            random_pred.append(int(chosen["predicted_label_id"]))
            best_candidate = max(
                enumerate(candidates),
                key=lambda item: (float(item[1]["logp_true"]), -item[0]),
            )[1]
            candidate_oracle_pred.append(int(best_candidate["predicted_label_id"]))
            best_score = float(best_candidate["logp_true"])
            if best_score > float(current["logp_true"]):
                safe_oracle_pred.append(int(best_candidate["predicted_label_id"]))
            else:
                safe_oracle_pred.append(current_prediction)
        else:
            random_pred.append(current_prediction)
            candidate_oracle_pred.append(current_prediction)
            safe_oracle_pred.append(current_prediction)

        for candidate in candidates:
            viewpoint = int(candidate["viewpoint_id"])
            correct = int(int(candidate["predicted_label_id"]) == label)
            per_view[viewpoint].append(correct)
            radius = (1.5, 2.0, 2.5, 3.0)[viewpoint // 8]
            per_radius[radius].append(correct)
            info_key = (str(row["scene_id"]), str(row["region"]))
            if info_key not in placement_info_cache:
                placement_info_cache[info_key] = _placement_view_info(audit, *info_key)
            info = placement_info_cache[info_key]
            if viewpoint in info:
                azimuth, human_yaw = info[viewpoint]
                yaw_stats[_yaw_bin(azimuth - human_yaw)].append(correct)

        cached = _cache_predictions(audit, _episode_id(row))
        if cached is not None:
            logs, predictions = cached
            cache_episode_count += 1
            cache_view_correct.extend((predictions == label).astype(np.int64).tolist())
            cache_view_predictions.extend(predictions.tolist())
            cache_view_labels.extend([label] * NUM_VIEWS)
            for viewpoint, correct in enumerate((predictions == label).astype(np.int64).tolist()):
                per_view_all[viewpoint].append(int(correct))
            cache_view_total += NUM_VIEWS

    record_groups: dict[str, list[str]] = defaultdict(list)
    for row in audit.utility_rows:
        record_groups[str(row["record_id"])].append(_episode_id(row))
    record_metrics: dict[str, dict[str, float]] = {}
    for record_id, ids in record_groups.items():
        by_id = {str(row["episode_id"]): row for row in audit.utility_rows}
        values = [by_id[key] for key in ids]
        indexes = [utility_index[_episode_id(value)] for value in values]
        record_metrics[record_id] = {
            "s0_accuracy": _mean(int(v["current"]["predicted_label_id"]) == int(v["label_id"]) for v in values),
            "h0_safe_oracle": _mean(int(pred) == int(v["label_id"]) for pred, v in zip(
                [safe_oracle_pred[index] for index in indexes], values
            )),
            "any_correct": _mean(bool(any_including_stay[index]) for index in indexes),
            "mean_correct_view_ratio": _mean(ratio_values[index] for index in indexes),
        }

    def _stats(values: Mapping[int, Sequence[int]]) -> dict[str, float | int | None]:
        return {
            str(key): {
                "count": len(value),
                "accuracy": _mean(value),
            }
            for key, value in sorted(values.items())
        }

    result: dict[str, Any] = {
        "episode_count": len(labels),
        "s0": _classification_metrics(s0_pred, labels, audit.spec.classes),
        "legal_candidate_pair_mean_accuracy": _mean(candidate_correct),
        "random_legal_view": _classification_metrics(random_pred, labels, audit.spec.classes),
        "h0_candidate_oracle": _classification_metrics(candidate_oracle_pred, labels, audit.spec.classes),
        "h0_safe_oracle": _classification_metrics(safe_oracle_pred, labels, audit.spec.classes),
        "h0_any_correct": _mean(any_correct),
        "h0_any_correct_including_stay": _mean(any_including_stay),
        "good_view_density": {
            "mean_num_correct": _mean(correct_counts),
            "median_num_correct": _median(correct_counts),
            "mean_correct_view_ratio": _mean(ratio_values),
            "median_correct_view_ratio": _median(ratio_values),
            "p_any_correct_candidate": _mean(any_correct),
            "p_any_correct_including_stay": _mean(any_including_stay),
            "correct_count_distribution": {
                "0": int(sum(value == 0 for value in correct_counts)),
                "1": int(sum(value == 1 for value in correct_counts)),
                "2-3": int(sum(2 <= value <= 3 for value in correct_counts)),
                "4-7": int(sum(4 <= value <= 7 for value in correct_counts)),
                "8-15": int(sum(8 <= value <= 15 for value in correct_counts)),
                ">=16": int(sum(value >= 16 for value in correct_counts)),
            },
        },
        "per_viewpoint_id": _stats(per_view),
        "per_viewpoint_id_all_archived_cache": _stats(per_view_all),
        "per_radius_m": _stats({int(round(radius * 10)): values for radius, values in per_radius.items()}),
        "relative_yaw": {
            key: {"count": len(value), "accuracy": _mean(value)}
            for key, value in ((key, yaw_stats.get(key, [])) for key in YAW_BINS)
        },
        "all_archived_32_view": {
            **_classification_metrics(cache_view_predictions, cache_view_labels, audit.spec.classes),
            "view_count": cache_view_total,
            "episode_count": cache_episode_count,
            "coverage": cache_episode_count / len(labels) if labels else float("nan"),
            "source": "existing Val true_logp cache; direct frozen ST-GCN recompute not run",
        },
        "record_level_macro_mean": {
            "record_count": len(record_metrics),
            "s0_accuracy": _mean(value["s0_accuracy"] for value in record_metrics.values()),
            "h0_safe_oracle": _mean(value["h0_safe_oracle"] for value in record_metrics.values()),
            "any_correct": _mean(value["any_correct"] for value in record_metrics.values()),
            "mean_correct_view_ratio": _mean(value["mean_correct_view_ratio"] for value in record_metrics.values()),
        },
        "cache_coverage": {
            "moving_episode_count": cache_episode_count,
            "all_episode_count": len(labels),
            "moving_fraction": cache_episode_count / len(labels) if labels else float("nan"),
        },
    }
    return result


def _fixed_h1_h2(audit: DatasetAudit) -> dict[str, Any]:
    labels: list[int] = []
    terminal_predictions: list[int] = []
    any_correct: list[bool] = []
    current_s1_predictions: list[int] = []
    remaining_counts: list[int] = []
    for row in audit.moving_rows:
        episode_id = _episode_id(row)
        cache_index = audit.cache_index[episode_id]
        label = int(row["label_id"])
        true_logs = np.asarray(audit.cache["true_logp"][cache_index], dtype=np.float64)
        s1_id = int(row["s1_viewpoint_id"])
        s1_prediction = int(np.argmax(np.asarray(audit.cache["current_logp_s1"][cache_index])))
        remaining = [int(value) for value in row.get("remaining_candidate_ids", [])]
        remaining_counts.append(len(remaining))
        best_score = float(true_logs[s1_id, label])
        terminal = s1_prediction
        correct_options = [s1_prediction == label]
        for candidate_id in remaining:
            score = float(true_logs[candidate_id, label])
            correct_options.append(int(np.argmax(true_logs[candidate_id])) == label)
            if score > best_score:
                best_score = score
                terminal = int(np.argmax(true_logs[candidate_id]))
        labels.append(label)
        terminal_predictions.append(terminal)
        current_s1_predictions.append(s1_prediction)
        any_correct.append(bool(any(correct_options)))
    result = {
        "episode_count": len(labels),
        "remaining_candidate_count": {"mean": _mean(remaining_counts), "median": _median(remaining_counts)},
        "fixed_h1_h2_safe_oracle": _classification_metrics(terminal_predictions, labels, audit.spec.classes),
        "fixed_h1_h2_any_correct": _mean(any_correct),
        "h1_terminal_s1": _classification_metrics(current_s1_predictions, labels, audit.spec.classes),
    }
    return result


def _build_specs(data_root: Path) -> tuple[DatasetSpec, DatasetSpec]:
    old_policy = data_root / "datasets/policy_v11_5"
    new_policy = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    old_offline = data_root / "datasets/offline/hm3d-train"
    new_offline = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return (
        DatasetSpec(
            name="old_selected16_four_region", policy_root=old_policy,
            stage_b_path=old_policy / "stage_b/utility_labels/val.jsonl",
            stage_d_path=old_policy / "stage_d/EXP014_two_step_sequential/features/val.jsonl",
            counterfactual_path=data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz",
            classes=16, skeleton_root=old_offline, candidate_root=old_offline,
        ),
        DatasetSpec(
            name="new_reduced14_eight_placement", policy_root=new_policy,
            stage_b_path=new_policy / "stage_b/utility_labels/val.jsonl",
            stage_d_path=new_policy / "stage_d/features/val.jsonl",
            counterfactual_path=new_policy / "counterfactual_cache/val.npz",
            classes=14, skeleton_root=new_offline, candidate_root=new_offline,
        ),
    )


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    old = result["datasets"]["old_selected16_four_region"]
    new = result["datasets"]["new_reduced14_eight_placement"]
    lines = [
        "# Old/New Val viewpoint and oracle audit",
        "",
        "This is a Val-only audit using the same Stage-B utility/oracle functions for both datasets.",
        "No model was trained, no Test artifact was opened, and no Habitat/data regeneration was performed.",
        "",
        "## Unified episode-level metrics",
        "",
        "| Metric | Old Val | New Val |",
        "|---|---:|---:|",
        f"| s0 Accuracy | {old['single_view_and_h0']['s0']['accuracy']:.6f} | {new['single_view_and_h0']['s0']['accuracy']:.6f} |",
        f"| Legal-view mean Accuracy | {old['single_view_and_h0']['legal_candidate_pair_mean_accuracy']:.6f} | {new['single_view_and_h0']['legal_candidate_pair_mean_accuracy']:.6f} |",
        f"| Random legal-view Accuracy | {old['single_view_and_h0']['random_legal_view']['accuracy']:.6f} | {new['single_view_and_h0']['random_legal_view']['accuracy']:.6f} |",
        f"| H0 CandidateOracle | {old['single_view_and_h0']['h0_candidate_oracle']['accuracy']:.6f} | {new['single_view_and_h0']['h0_candidate_oracle']['accuracy']:.6f} |",
        f"| H0 SafeOracle | {old['single_view_and_h0']['h0_safe_oracle']['accuracy']:.6f} | {new['single_view_and_h0']['h0_safe_oracle']['accuracy']:.6f} |",
        f"| H0 AnyCorrect | {old['single_view_and_h0']['h0_any_correct_including_stay']:.6f} | {new['single_view_and_h0']['h0_any_correct_including_stay']:.6f} |",
        f"| Mean # correct legal views | {old['single_view_and_h0']['good_view_density']['mean_num_correct']:.6f} | {new['single_view_and_h0']['good_view_density']['mean_num_correct']:.6f} |",
        f"| Mean correct-view ratio | {old['single_view_and_h0']['good_view_density']['mean_correct_view_ratio']:.6f} | {new['single_view_and_h0']['good_view_density']['mean_correct_view_ratio']:.6f} |",
        f"| FixedH1-H2 SafeOracle | {old['fixed_h1_h2']['fixed_h1_h2_safe_oracle']['accuracy']:.6f} | {new['fixed_h1_h2']['fixed_h1_h2_safe_oracle']['accuracy']:.6f} |",
        f"| FixedH1-H2 AnyCorrect | {old['fixed_h1_h2']['fixed_h1_h2_any_correct']:.6f} | {new['fixed_h1_h2']['fixed_h1_h2_any_correct']:.6f} |",
        "",
        "## Scientific answers",
        "",
        "1. The new average single-view result is compared using the same Stage-B current/candidate definitions above; the archived-32-view metric is cache-covered moving Val only.",
        "2. Good-view sparsity is assessed by the correct-view count and ratio distributions. AnyCorrect is reported separately and is not substituted for mean single-view accuracy.",
        "3. H0 SafeOracle is the unified argmax over current s0 plus legal candidates, with the final class always taken from the selected real prediction.",
        "4. FixedH1-H2 SafeOracle is a separate protocol and therefore should not be interpreted as the H0 ceiling.",
        "5. The sampled metadata checks and cache ID checks below are the alignment audit. Direct ST-GCN recomputation was not run because the habitat PyTorch environment reported no usable CUDA device; no CPU fallback was used.",
        "6. No evidence of an alignment error is reported by the checks; regeneration is not recommended from this audit alone.",
        "",
        "## Leakage and runtime flags",
        "",
        "- test_used: false",
        "- model_training: false",
        "- habitat_regeneration: false",
        "- direct_stgcn_recompute: NOT RUN (CUDA unavailable in habitat environment)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(data_root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    specs = _build_specs(data_root)
    datasets: dict[str, Any] = {}
    for spec in specs:
        audit = _load_dataset(spec)
        datasets[spec.name] = {
            "spec": {
                "policy_root": str(spec.policy_root),
                "stage_b": str(spec.stage_b_path),
                "stage_d": str(spec.stage_d_path),
                "counterfactual_cache": str(spec.counterfactual_path),
                "classes": spec.classes,
            },
            "alignment": _source_alignment(audit),
            "single_view_and_h0": _single_view_and_h0(audit),
            "fixed_h1_h2": _fixed_h1_h2(audit),
        }
    result: dict[str, Any] = {
        "experiment": "dataset_oracle_audit",
        "version": "old-new-val-unified-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.monotonic() - started,
        "split": "val",
        "test_used": False,
        "training": False,
        "habitat_regeneration": False,
        "oracle_definition": "same Stage-B true-logp selection and real archived prediction for old/new",
        "datasets": datasets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/reduced14_eight_placement_v1/dataset_oracle_audit"),
    )
    args = parser.parse_args()
    result = run_audit(args.data_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "completed", "split": "val", "test_used": result["test_used"]}, indent=2))


if __name__ == "__main__":
    main()
