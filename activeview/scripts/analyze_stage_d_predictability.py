#!/usr/bin/env python3
"""Run the Val-only EXP028 oracle-action predictability audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, build_stage_d_trajectories
from activeview.active_view.stage_d_predictability import (
    ACTION_NAMES, MARGIN_BINS, margin_bin_index, neighbor_agreement, neighbor_entropy,
    oracle_action_index, oracle_margin, quantized_context_key,
)
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, observation_keys_from_feature_rows
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    expected = str(split).lower()
    invalid = {"<missing>" if row.get("policy_split") is None else str(row["policy_split"]).lower() for row in rows if str(row.get("policy_split", "")).lower() != expected}
    if invalid:
        raise ValueError(f"{name} must explicitly contain only {expected}: {sorted(invalid)}")


def _index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in result:
            raise ValueError(f"Duplicate {name} episode_id: {episode_id}")
        result[episode_id] = row
    return result


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _load_spatial_cache(path: Path, keys: Sequence[RGBObservationKey]) -> tuple[np.ndarray, dict[tuple[str, str, str, int], int]]:
    embeddings = np.load(path / "embeddings.npy", mmap_mode="r")
    manifest = [json.loads(line) for line in (path / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"])): i for i, row in enumerate(manifest)}
    expected = {key.tuple for key in keys}
    if embeddings.ndim != 3 or embeddings.shape[1:] != (16, 768) or embeddings.dtype != np.float16:
        raise ValueError(f"Invalid EXP025 spatial cache shape/dtype: {embeddings.shape} {embeddings.dtype}")
    if set(index) != expected or embeddings.shape[0] != len(manifest):
        raise ValueError("EXP025 cache keys do not exactly match visited s0/s1 observations")
    return embeddings, index


def _rgb_pooled(rows: Sequence[Mapping[str, Any]], embeddings: np.ndarray, index: Mapping[tuple[str, str, str, int], int]) -> np.ndarray:
    output = np.empty((len(rows), 2, 768), dtype=np.float32)
    for i, row in enumerate(rows):
        common = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        for position, viewpoint_id in enumerate((int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]))):
            key = (*common, viewpoint_id)
            if key not in index:
                raise ValueError(f"Missing visited RGB cache key: {key}")
            output[i, position] = np.asarray(embeddings[index[key]], dtype=np.float32).mean(axis=0)
    return output


def _observable_vectors(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, np.ndarray], rgb: np.ndarray, rgb_mean: np.ndarray, rgb_std: np.ndarray) -> np.ndarray:
    vectors = np.empty((len(rows), 275 + 275 + 19 + 11 + 11 + 768 + 768), dtype=np.float32)
    for i, row in enumerate(rows):
        current = np.asarray(stats["current_mean"], dtype=np.float32)
        current_std = np.asarray(stats["current_std"], dtype=np.float32)
        delta_mean = np.asarray(stats["delta_mean"], dtype=np.float32)
        delta_std = np.asarray(stats["delta_std"], dtype=np.float32)
        geometry_mean = np.asarray(stats["geometry_mean"], dtype=np.float32)
        geometry_std = np.asarray(stats["geometry_std"], dtype=np.float32)
        s0 = (np.asarray(row["s0_feature"], dtype=np.float32) - current) / current_std
        s1 = (np.asarray(row["s1_feature"], dtype=np.float32) - current) / current_std
        delta = (np.asarray(row["delta_semantic"], dtype=np.float32) - delta_mean) / delta_std
        geometry = np.asarray(row["second_step_candidate_geometry"], dtype=np.float32)
        padded = np.broadcast_to(geometry_mean, (2, 11)).copy()
        padded[: len(geometry)] = geometry
        padded = (padded - geometry_mean) / geometry_std
        rgb_pair = (rgb[i] - rgb_mean) / rgb_std
        vectors[i] = np.concatenate((s0, s1, delta, padded.reshape(-1), rgb_pair[0], rgb_pair[1]))
    if not np.isfinite(vectors).all():
        raise ValueError("Observable vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Observable vector has zero norm")
    return vectors / norms


def _exp027_action(row: Mapping[str, Any]) -> int:
    if bool(row["predicted_stays"]):
        return 0
    selected = row.get("predicted_candidate_viewpoint_id")
    ids = [int(value) for value in row["remaining_candidate_ids"]]
    if selected is None or int(selected) not in ids:
        raise ValueError(f"EXP027 selected candidate is not in remaining IDs: {row['episode_id']}")
    return ids.index(int(selected)) + 1


def _margin_table(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], trajectories: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    buckets: list[list[Mapping[str, Any]]] = [[] for _ in MARGIN_BINS]
    for row in rows:
        buckets[margin_bin_index(float(oracle_margin(row["second_step_utility_targets"])["margin_1"]))].append(row)
    output = []
    for index, bucket in enumerate(buckets):
        labels = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in bucket], dtype=np.int64)
        predicted = np.asarray([_exp027_action(predictions[str(row["episode_id"])]) for row in bucket], dtype=np.int64)
        oracle_move = labels > 0; predicted_move = predicted > 0
        candidate_mask = oracle_move & predicted_move
        candidate_hit = float(np.mean(predicted[candidate_mask] == labels[candidate_mask])) if np.any(candidate_mask) else None
        selected_utilities = []
        regrets = []
        if trajectories is not None:
            for row in bucket:
                trajectory = trajectories[str(row["episode_id"])]
                selected_utilities.append(float(trajectory["selected_true_utility"]))
                regrets.append(float(trajectory["regret"]))
        lower, upper = MARGIN_BINS[index]
        output.append({
            "bin": f"[{lower}, {upper if np.isfinite(upper) else '+inf'})",
            "lower": lower, "upper": None if not np.isfinite(upper) else upper,
            "count": len(bucket),
            "oracle_distribution": {name: int(np.sum(labels == action)) for action, name in enumerate(ACTION_NAMES)},
            "exp027_three_way_accuracy": float(np.mean(predicted == labels)) if len(bucket) else None,
            "exp027_binary_move_stay_accuracy": float(np.mean(predicted_move == oracle_move)) if len(bucket) else None,
            "exp027_both_move_candidate_hit": candidate_hit,
            "selected_action_mean_true_utility": float(np.mean(selected_utilities)) if selected_utilities else None,
            "trajectory_mean_regret": float(np.mean(regrets)) if regrets else None,
        })
    return output


def _high_margin(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], trajectories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for threshold in (0.25, 0.5, 1.0):
        selected = [row for row in rows if float(oracle_margin(row["second_step_utility_targets"])["margin_1"]) >= threshold]
        labels = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in selected], dtype=np.int64)
        predicted = np.asarray([_exp027_action(predictions[str(row["episode_id"])]) for row in selected], dtype=np.int64)
        move = labels > 0; pred_move = predicted > 0
        both = move & pred_move
        output[f"S{(0.25, 0.5, 1.0).index(threshold) + 1}"] = {
            "margin_threshold": threshold, "count": len(selected),
            "three_way_accuracy": float(np.mean(labels == predicted)) if len(selected) else None,
            "binary_move_stay_accuracy": float(np.mean(move == pred_move)) if len(selected) else None,
            "candidate_hit": float(np.mean(labels[both] == predicted[both])) if np.any(both) else None,
            "selected_true_utility": float(np.mean([trajectories[str(row["episode_id"])] ["selected_true_utility"] for row in selected])) if selected else None,
        }
    return output


def _group_audit(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[quantized_context_key(row)].append(row)
    valid = [items for items in groups.values() if len(items) >= 5]
    by_region: dict[str, list[list[Mapping[str, Any]]]] = defaultdict(list)
    for items in valid:
        by_region[str(items[0]["region"])].append(items)
    def stats(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        pairs = action_pairs = binary_pairs = candidate_switches = move_pairs = 0
        for group in items:
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    if group[left]["record_id"] == group[right]["record_id"]:
                        continue
                    pairs += 1
                    a = oracle_action_index(group[left]["second_step_utility_targets"]); b = oracle_action_index(group[right]["second_step_utility_targets"])
                    action_pairs += int(a != b); binary_pairs += int((a > 0) != (b > 0))
                    if a > 0 and b > 0:
                        move_pairs += 1; candidate_switches += int(a != b)
        return {"group_count": len(items), "cross_motion_pair_count": pairs, "oracle_action_switch_rate": action_pairs / pairs if pairs else None, "binary_gate_switch_rate": binary_pairs / pairs if pairs else None, "candidate_move_pair_count": move_pairs, "candidate_switch_rate_move_only": candidate_switches / move_pairs if move_pairs else None}
    return {"all": stats(valid), "per_region": {region: stats(items) for region, items in sorted(by_region.items())}}


def _region_breakdown(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], nn_labels: np.ndarray, nn_indices: np.ndarray, trajectories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for region in ("bedroom", "living_room", "kitchen", "dining_area"):
        indices = [i for i, row in enumerate(rows) if str(row["region"]) == region]
        if not indices:
            continue
        labels = np.asarray([oracle_action_index(rows[i]["second_step_utility_targets"]) for i in indices], dtype=np.int64)
        predicted = np.asarray([_exp027_action(predictions[str(rows[i]["episode_id"])]) for i in indices], dtype=np.int64)
        entropies = [neighbor_entropy(nn_labels[i, :25]) for i in indices]
        majority = np.asarray([max(range(3), key=lambda a: (int(np.sum(nn_labels[i, :25] == a)), -a)) for i in indices])
        harmful = []
        for i in indices:
            tr = trajectories[str(rows[i]["episode_id"])]
            if tr["moves"] >= 2 and float(tr["selected_true_utility"]) <= 0:
                harmful.append(float(tr["selected_true_utility"]))
        output[region] = {"episode_count": len(indices), "oracle_distribution": {name: int(np.sum(labels == a)) for a, name in enumerate(ACTION_NAMES)}, "oracle_margin_mean": float(np.mean([oracle_margin(rows[i]["second_step_utility_targets"])["margin_1"] for i in indices])), "exp027_accuracy": float(np.mean(predicted == labels)), "nn25_entropy_mean": float(np.mean([item["three_way"] for item in entropies])), "nn25_majority_accuracy": float(np.mean(majority == labels)), "harmful_move_rate": float(len(harmful) / len(indices)), "harmful_move_count": len(harmful)}
    return output


def _action_class_breakdown(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], trajectories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels = [oracle_action_index(row["second_step_utility_targets"]) for row in rows]
    predicted = [_exp027_action(predictions[str(row["episode_id"])]) for row in rows]
    result: dict[str, Any] = {}
    stay = [i for i, action in enumerate(labels) if action == 0]
    false_moves = [i for i in stay if predicted[i] > 0]
    result["oracle_stay"] = {"count": len(stay), "correct_stay": len(stay) - len(false_moves), "false_move_rate": len(false_moves) / len(stay) if stay else 0.0, "false_move_mean_regret": float(np.mean([float(trajectories[str(rows[i]["episode_id"])]["regret"]) for i in false_moves])) if false_moves else 0.0}
    move = [i for i, action in enumerate(labels) if action > 0]
    missed = [i for i in move if predicted[i] == 0]
    result["oracle_move"] = {"count": len(move), "move_recall": (len(move) - len(missed)) / len(move) if move else 0.0, "false_stay_rate": len(missed) / len(move) if move else 0.0}
    for action, name in ((1, "oracle_p2"), (2, "oracle_p3")):
        subset = [i for i, value in enumerate(labels) if value == action]
        result[name] = {"count": len(subset), "candidate_hit": float(np.mean([predicted[i] == action for i in subset])) if subset else 0.0, "false_stay_rate": float(np.mean([predicted[i] == 0 for i in subset])) if subset else 0.0}
    return result


def _missed_by_best_utility(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stratify EXP027 missed beneficial moves by fixed oracle utility bins."""
    bins = ((0.0, 0.05), (0.05, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf")))
    output = []
    for lower, upper in bins:
        values = []
        for row in rows:
            utilities = [float(value) for value in row["second_step_utility_targets"]]
            if _exp027_action(predictions[str(row["episode_id"])]) == 0 and oracle_action_index(utilities) > 0:
                best = max(utilities)
                if lower <= best < upper:
                    values.append(best)
        output.append({"bin": f"[{lower}, {upper if np.isfinite(upper) else '+inf'})", "count": len(values), "total_missed_positive_utility": float(sum(values)), "mean_missed_positive_utility": float(np.mean(values)) if values else None})
    return output


def analyze(*, cache_root: Path, stage_b_root: Path, v0_train_predictions: Path, v0_val_predictions: Path, exp027_runtime: Path, spatial_cache: Path, label_mapping: Path, output: Path) -> dict[str, Any]:
    summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    train_features = load_jsonl(Path(summary["feature_files"]["train"])); val_features = load_jsonl(Path(summary["feature_files"]["val"]))
    stage_b_train = load_jsonl(stage_b_root / "utility_labels" / "train.jsonl"); stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_train = load_jsonl(v0_train_predictions); v0_val = load_jsonl(v0_val_predictions)
    exp027_train = load_jsonl(exp027_runtime / "train_predictions.jsonl"); exp027_val = load_jsonl(exp027_runtime / "val_predictions.jsonl")
    for rows, split, name in ((train_features, "train", "Stage D Train"), (val_features, "val", "Stage D Val"), (stage_b_train, "train", "Stage B Train"), (stage_b_val, "val", "Stage B Val"), (v0_train, "train", "v0 Train"), (v0_val, "val", "v0 Val"), (exp027_train, "train", "EXP027 Train"), (exp027_val, "val", "EXP027 Val")):
        _assert_split(rows, split, name)
    train_ids = {str(row["episode_id"]) for row in train_features}; val_ids = {str(row["episode_id"]) for row in val_features}
    if train_ids & val_ids:
        raise ValueError("Train/Val feature episode overlap")
    for features, v0, name in ((train_features, v0_train, "Train"), (val_features, v0_val, "Val")):
        eligible_v0_ids = {str(row["episode_id"]) for row in v0 if not bool(row["predicted_stays"])}
        if eligible_v0_ids != {str(row["episode_id"]) for row in features}:
            raise ValueError(f"{name} Stage-D/v0 episode alignment mismatch")
    train_pred = _index(exp027_train, "EXP027 Train prediction"); val_pred = _index(exp027_val, "EXP027 Val prediction")
    if set(train_pred) != train_ids or set(val_pred) != val_ids:
        raise ValueError("EXP027 prediction episode alignment mismatch")
    train_keys, _ = observation_keys_from_feature_rows(train_features); val_keys, _ = observation_keys_from_feature_rows(val_features)
    if {key.tuple for key in train_keys} & {key.tuple for key in val_keys}:
        raise ValueError("Train/Val RGB observation keys overlap")
    embeddings, rgb_index = _load_spatial_cache(spatial_cache, sorted(set(train_keys) | set(val_keys), key=lambda key: key.tuple))
    train_rgb = _rgb_pooled(train_features, embeddings, rgb_index); val_rgb = _rgb_pooled(val_features, embeddings, rgb_index)
    unique_train_rgb = np.unique(train_rgb.reshape(-1, 768), axis=0); rgb_mean = unique_train_rgb.mean(axis=0); rgb_std = unique_train_rgb.std(axis=0); rgb_std[rgb_std < 1e-6] = 1.0
    train_vectors = _observable_vectors(train_features, stats, train_rgb, rgb_mean, rgb_std); val_vectors = _observable_vectors(val_features, stats, val_rgb, rgb_mean, rgb_std)
    train_labels = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in train_features], dtype=np.int64); val_labels = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in val_features], dtype=np.int64)
    # The canonical oracle is independently reconstructed and checked against the frozen helper.
    fixed = build_fixed_first_oracle(stage_b_val, v0_val, val_features); fixed_index = _index(fixed, "Fixed-first oracle")
    for row in val_features:
        oracle = oracle_action_index(row["second_step_utility_targets"]); trajectory = fixed_index[str(row["episode_id"])]
        direct = 0 if int(trajectory["moves"]) == 1 else [int(value) for value in row["remaining_candidate_ids"]].index(int(trajectory["selected_viewpoint_id"])) + 1
        if oracle != direct:
            raise ValueError(f"Fixed-first oracle mismatch for {row['episode_id']}")
    exp027_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_features, exp027_val); trajectory_index = _index(exp027_trajectories, "EXP027 trajectory")
    import faiss  # type: ignore
    nn_index = faiss.IndexFlatIP(train_vectors.shape[1]); nn_index.add(np.asarray(train_vectors, dtype=np.float32)); similarities, neighbor_indices = nn_index.search(np.asarray(val_vectors, dtype=np.float32), 25)
    train_neighbor_labels = train_labels[neighbor_indices]
    nn_metrics = {str(k): neighbor_agreement(train_labels, neighbor_indices, val_labels, k) for k in (1, 5, 10, 25)}
    entropies = [neighbor_entropy(row.tolist()) for row in train_neighbor_labels]
    margin_table = _margin_table(val_features, val_pred, trajectory_index)
    train_margin_table = _margin_table(train_features, train_pred, None)
    for bin_index, row in enumerate(margin_table):
        indices = [i for i, feature in enumerate(val_features) if margin_bin_index(float(oracle_margin(feature["second_step_utility_targets"])["margin_1"])) == bin_index]
        if indices:
            labels_for_bin = val_labels[indices]; neighbors_for_bin = train_neighbor_labels[indices]
            majority = np.asarray([max(range(3), key=lambda action: (int(np.sum(values == action)), -action)) for values in neighbors_for_bin], dtype=np.int64)
            entropy_for_bin = [neighbor_entropy(values.tolist()) for values in neighbors_for_bin]
            row.update({"nn25_three_way_accuracy": float(np.mean(majority == labels_for_bin)), "nn25_binary_accuracy": float(np.mean((majority > 0) == (labels_for_bin > 0))), "nn25_three_way_entropy": float(np.mean([value["three_way"] for value in entropy_for_bin])), "nn25_binary_entropy": float(np.mean([value["binary"] for value in entropy_for_bin]))})
    all_margins = np.asarray([float(oracle_margin(row["second_step_utility_targets"])["margin_1"]) for row in val_features])
    high = _high_margin(val_features, val_pred, trajectory_index)
    local_consistency = {"three_way_mean": float(np.mean([max(np.bincount(row, minlength=3) / 25.0) for row in train_neighbor_labels])), "three_way_median": float(np.median([max(np.bincount(row, minlength=3) / 25.0) for row in train_neighbor_labels])), "binary_mean": float(np.mean([max(np.bincount((row > 0).astype(np.int64), minlength=2) / 25.0) for row in train_neighbor_labels])), "binary_median": float(np.median([max(np.bincount((row > 0).astype(np.int64), minlength=2) / 25.0) for row in train_neighbor_labels])), "three_way_ge_0.8": float(np.mean([max(np.bincount(row, minlength=3) / 25.0) >= 0.8 for row in train_neighbor_labels])), "three_way_ge_0.9": float(np.mean([max(np.bincount(row, minlength=3) / 25.0) >= 0.9 for row in train_neighbor_labels]))}
    harmful = []; missed = []; harmful_by_margin = defaultdict(lambda: {"count": 0, "total_negative_utility": 0.0}); missed_by_margin = defaultdict(lambda: {"count": 0, "total_positive_utility": 0.0})
    for row in val_features:
        pred = _exp027_action(val_pred[str(row["episode_id"])]); utilities = [float(v) for v in row["second_step_utility_targets"]]; selected = 0.0 if pred == 0 else utilities[pred - 1]; oracle = oracle_action_index(utilities); bucket = margin_bin_index(float(oracle_margin(utilities)["margin_1"]))
        if pred > 0 and selected <= 0: harmful.append(selected); harmful_by_margin[bucket]["count"] += 1; harmful_by_margin[bucket]["total_negative_utility"] += -selected
        if pred == 0 and oracle > 0: missed.append(max(utilities)); missed_by_margin[bucket]["count"] += 1; missed_by_margin[bucket]["total_positive_utility"] += max(utilities)
    summary = {
        "experiment_id": "EXP028", "experiment_name": "oracle_action_predictability_representation_sufficiency_audit", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": False, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False,
        "eligible_episode_counts": {"train": len(train_features), "val": len(val_features)}, "oracle_distribution": {"train": {name: int(np.sum(train_labels == i)) for i, name in enumerate(ACTION_NAMES)}, "val": {name: int(np.sum(val_labels == i)) for i, name in enumerate(ACTION_NAMES)}},
        "margin_distribution": {"val_mean": float(np.mean(all_margins)), "val_median": float(np.median(all_margins)), "val_p25": float(np.percentile(all_margins, 25)), "val_p75": float(np.percentile(all_margins, 75)), "val_p90": float(np.percentile(all_margins, 90))},
        "margin_bins": {"train": train_margin_table, "val": margin_table}, "high_margin_subsets": high, "nearest_neighbor": {"metric": "cosine", "index_split": "train", "agreement": nn_metrics, "neighbor_label_exclusion_self": True},
        "neighborhood_entropy": {"three_way_mean": float(np.mean([e["three_way"] for e in entropies])), "three_way_median": float(np.median([e["three_way"] for e in entropies])), "three_way_p25": float(np.percentile([e["three_way"] for e in entropies], 25)), "three_way_p75": float(np.percentile([e["three_way"] for e in entropies], 75)), "three_way_p90": float(np.percentile([e["three_way"] for e in entropies], 90)), "binary_mean": float(np.mean([e["binary"] for e in entropies])), "binary_median": float(np.median([e["binary"] for e in entropies]))},
        "local_consistency": local_consistency, "same_context_cross_motion": {"train": _group_audit(train_features, train_pred), "val": _group_audit(val_features, val_pred)}, "region_breakdown": _region_breakdown(val_features, val_pred, train_neighbor_labels, neighbor_indices, trajectory_index), "action_class_breakdown": _action_class_breakdown(val_features, val_pred, trajectory_index),
        "harmful_move_vs_margin": {str(k): value for k, value in sorted(harmful_by_margin.items())}, "missed_beneficial_move_vs_margin": {str(k): value for k, value in sorted(missed_by_margin.items())}, "missed_beneficial_move_vs_best_utility": _missed_by_best_utility(val_features, val_pred), "harmful_move": {"count": len(harmful), "rate": len(harmful) / len(val_features), "mean_selected_utility": float(np.mean(harmful)) if harmful else None, "total_negative_utility": float(-sum(harmful))}, "missed_beneficial_move": {"count": len(missed), "rate": len(missed) / len(val_features), "mean_oracle_utility": float(np.mean(missed)) if missed else None, "total_missed_positive_utility": float(sum(missed))},
        "optional_probe": {"status": "SKIPPED", "reason": "Nearest-neighbor audit directly answers the registered diagnostic without adding a trained probe"},
        "representation": {"blocks": ["normalized_s0", "normalized_s1", "normalized_delta", "normalized_p2_geometry", "normalized_p3_geometry", "train-z-scored_rgb_spatial_s0_mean", "train-z-scored_rgb_spatial_s1_mean"], "dimension": int(train_vectors.shape[1]), "normalization_stats_split": "train", "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_skeleton_used": False, "true_utility_used_as_model_input": False, "val_used_for_feature_normalization": False, "val_used_for_neighbor_index": False},
        "provenance": {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "exp027_train_predictions_sha256": file_sha256(exp027_runtime / "train_predictions.jsonl"), "exp027_val_predictions_sha256": file_sha256(exp027_runtime / "val_predictions.jsonl"), "spatial_manifest_sha256": file_sha256(spatial_cache / "manifest.jsonl")},
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"); return summary


def _analysis_markdown(result: Mapping[str, Any]) -> str:
    rows = result["margin_bins"]["val"]
    lines = ["# EXP028 — Oracle Action Predictability / Representation Sufficiency Audit", "", "Val-only diagnostic; no policy was trained and Test was not read.", "", "## Table A — Margin bins", "", "| margin | count | EXP027 3-way acc | binary acc | candidate hit | NN25 entropy |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['bin']} | {row['count']} | {row['exp027_three_way_accuracy']} | {row['exp027_binary_move_stay_accuracy']} | {row['nn25_three_way_accuracy']} | {row['nn25_three_way_entropy']} |")
    lines += ["", "## Table B — NN agreement", "", "| k | 3-way accuracy | binary accuracy |", "|---:|---:|---:|"]
    for k, row in result["nearest_neighbor"]["agreement"].items():
        lines.append(f"| {k} | {row['three_way_accuracy']:.6f} | {row['binary_accuracy']:.6f} |")
    lines += ["", "## Table C — High-margin subsets", "", "| subset | count | 3-way acc | binary acc | candidate hit |", "|---|---:|---:|---:|---:|"]
    for name, row in result["high_margin_subsets"].items():
        lines.append(f"| {name} (≥{row['margin_threshold']}) | {row['count']} | {row['three_way_accuracy']} | {row['binary_move_stay_accuracy']} | {row['candidate_hit']} |")
    lines += ["", "## Local consistency", "", json.dumps(result["local_consistency"], indent=2), "", "## Action-class breakdown", "", json.dumps(result["action_class_breakdown"], indent=2), "", "## Scientific interpretation", "", "EXP028 is an analysis-only audit. The registered Case A/B/C label remains INCONCLUSIVE until the observed margin-stratified accuracy, NN agreement and entropy are reviewed together. These results assess predictability under the frozen legal representation and held-out-motion protocol; they do not establish that future viewpoints are intrinsically unpredictable.", "", "## Leakage flags", "", "- future_candidate_rgb_used=false", "- future_candidate_depth_used=false", "- future_candidate_skeleton_used=false", "- true_utility_used_as_model_input=false", "- val_used_for_feature_normalization=false", "- val_used_for_neighbor_index=false", "- test_used=false"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root(); parser = argparse.ArgumentParser(description=__doc__)
    cache = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"; stage_b = data_root / "datasets/policy_v11_5/stage_b"
    parser.add_argument("--cache-root", type=Path, default=cache); parser.add_argument("--stage-b-root", type=Path, default=stage_b)
    parser.add_argument("--v0-train-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl"); parser.add_argument("--v0-val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    parser.add_argument("--exp027-runtime", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP027_spatial_rgb_oracle_behavior_cloning")); parser.add_argument("--spatial-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4")); parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP028_oracle_action_predictability/result.json"); parser.add_argument("--analysis", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP028_oracle_action_predictability/analysis.md"); return parser


def main() -> None:
    args = build_parser().parse_args(); values = vars(args); analysis = values.pop("analysis"); result = analyze(**values); analysis.write_text(_analysis_markdown(result), encoding="utf-8"); print(json.dumps({"experiment_id": "EXP028", "status": result["status"], "test_used": False, "training_performed": False, "eligible_episode_counts": result["eligible_episode_counts"], "nearest_neighbor": result["nearest_neighbor"]["agreement"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
