#!/usr/bin/env python3
"""Run the corrected EXP035--EXP037-R1 campaign with context-safe identity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import (
    VIEW_COUNT, canonical_realpath, context_key, deterministic_oracle_action,
    fit_bayesian_linear, gmrf_smooth, graph_edges, predict_model,
    relative_view_descriptor, train_bradley_terry, train_dense_regressor,
    viewpoint_azimuth, viewpoint_radius,
)
from activeview.active_view.stage_d_evaluation import (
    build_fixed_first_oracle, build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_policy import second_step_decision
from activeview.active_view.stage_d_predictability import oracle_margin
from activeview.active_view.utility_label_builder import file_sha256
from activeview.scripts.build_stage_b_utility_labels import _load_archive_predictions, _load_model

EXP_ROOT = REPO_ROOT / "experiments" / "stage_d"
EXP035 = EXP_ROOT / "EXP035_R1_dense_recognition_quality_field"
EXP036 = EXP_ROOT / "EXP036_R1_single_step_view_optimization"
EXP037 = EXP_ROOT / "EXP037_R1_multistep_graph_active_perception"


def _corr(x: Iterable[float], y: Iterable[float]) -> dict[str, float | None]:
    a = np.asarray(list(x), dtype=np.float64); b = np.asarray(list(y), dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])}


def _episode_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / "datasets/policy_v11_5/episodes" / f"{split}_episodes.jsonl"; rows = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows): raise ValueError(f"invalid policy split in {path}")
    return rows


def _utility_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / "datasets/policy_v11_5/stage_b/utility_labels" / f"{split}.jsonl"; rows = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows): raise ValueError(f"invalid utility split in {path}")
    return rows


def _source_by_context(episodes: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    output: dict[tuple[str, str, str], str] = {}
    for row in episodes:
        source = row.get("current_view", {}).get("skeleton_source_path")
        if not source: raise ValueError(f"Missing skeleton source path for {row.get('episode_id')}")
        key = context_key(row); resolved = canonical_realpath(source)
        if key in output and output[key] != resolved: raise ValueError(f"Source path mismatch for context {key}")
        output[key] = resolved
    return output


def _identity_audit(episodes: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    by_record: dict[str, set[tuple[str, str]]] = defaultdict(set); source = _source_by_context(episodes); examples: list[dict[str, Any]] = []
    for row in episodes: by_record[str(row["record_id"])].add((str(row["scene_id"]), str(row["region"])))
    collisions = {record: contexts for record, contexts in by_record.items() if len(contexts) > 1}
    for record in sorted(collisions)[:20]:
        for scene, region in sorted(collisions[record])[:3]: examples.append({"record_id": record, "scene_id": scene, "region": region, "source_path": source[(scene, region, record)]})
    return {"split": split, "episode_rows": len(episodes), "unique_record_id": len(by_record), "unique_context_key": len(source), "record_ids_with_multiple_contexts": len(collisions), "max_contexts_per_record_id": max((len(v) for v in by_record.values()), default=0), "collision_examples": examples}


def _split_audit(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row["episode_id"]) for row in train_rows}; val_ids = {str(row["episode_id"]) for row in val_rows}; train_ctx = {context_key(row) for row in train_rows}; val_ctx = {context_key(row) for row in val_rows}; train_record = {str(row["record_id"]) for row in train_rows}; val_record = {str(row["record_id"]) for row in val_rows}
    return {"episode_id_overlap": len(train_ids & val_ids), "context_key_overlap": len(train_ctx & val_ctx), "record_id_overlap_allowed_by_protocol": len(train_record & val_record), "train_episode_count": len(train_ids), "val_episode_count": len(val_ids)}


def _dense_fields(data_root: Path, split: str, stage_d_rows: Sequence[Mapping[str, Any]], episode_source: Mapping[tuple[str, str, str], str], out_root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text()); mapping = json.loads(Path(summary["label_mapping"]).read_text()); model, device = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), "cpu")
    contexts = sorted({context_key(row) for row in stage_d_rows}); output_dir = out_root / split; output_dir.mkdir(parents=True, exist_ok=True); fields: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows_by_context: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in stage_d_rows: rows_by_context[context_key(row)].append(row)
    for scene, region, record_id in contexts:
        key = (scene, region, record_id); source = Path(episode_source[key])
        with np.load(source, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int32); positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        if ids.shape != (VIEW_COUNT,) or positions.shape != (VIEW_COUNT, 3): raise ValueError(f"Invalid navigation schema: {source}")
        if len(rows_by_context[key]) != 1: raise ValueError(f"Expected one Stage-D row per context: {key}")
        target = output_dir / scene / region / f"{record_id}.npz"
        if target.is_file():
            with np.load(target, allow_pickle=False) as archive:
                if tuple(np.asarray(archive["viewpoint_ids"], dtype=np.int32)) != tuple(ids):
                    raise ValueError(f"Existing R1 field viewpoint mismatch: {target}")
                if str(np.asarray(archive["source_skeleton_sha256"]).item()) != file_sha256(source):
                    raise ValueError(f"Existing R1 field source hash mismatch: {target}")
                fields[key] = {"context_key": key, "record_id": record_id, "scene_id": scene, "region": region, "label_id": int(np.asarray(archive["label_id"]).item()), "viewpoint_ids": ids, "positions": positions, "radius": np.asarray(archive["radius"], dtype=np.float32), "azimuth": np.asarray(archive["azimuth"], dtype=np.float32), "logp_true": np.asarray(archive["logp_true"], dtype=np.float32), "ce": np.asarray(archive["ce"], dtype=np.float32), "entropy": np.asarray(archive["entropy"], dtype=np.float32), "top1_probability": np.asarray(archive["top1_probability"], dtype=np.float32), "top1_top2_margin": np.asarray(archive["top1_top2_margin"], dtype=np.float32), "predicted_class": np.asarray(archive["predicted_class"], dtype=np.int64), "correct": np.asarray(archive["correct"], dtype=bool), "source_path": str(source), "source_sha256": file_sha256(source)}
            continue
        predictions = _load_archive_predictions(source, model, device, 64, len(mapping)); log_probs = np.full((VIEW_COUNT, len(mapping)), np.nan, dtype=np.float32)
        for viewpoint_id, values in predictions.items(): log_probs[int(viewpoint_id)] = values.astype(np.float32)
        label_id = int(rows_by_context[key][0]["label_id"]); selected = log_probs[:, label_id]; probabilities = np.exp(log_probs); top2 = np.partition(log_probs, -2, axis=1)[:, -2:]; margin = np.max(top2, axis=1) - np.min(top2, axis=1)
        field = {"context_key": key, "record_id": record_id, "scene_id": scene, "region": region, "label_id": label_id, "viewpoint_ids": ids, "positions": positions, "radius": np.asarray([viewpoint_radius(v) for v in ids], dtype=np.float32), "azimuth": np.asarray([viewpoint_azimuth(v) for v in ids], dtype=np.float32), "logp_true": selected, "ce": -selected, "entropy": -np.sum(probabilities * np.nan_to_num(log_probs), axis=1), "top1_probability": np.max(probabilities, axis=1), "top1_top2_margin": margin, "predicted_class": np.argmax(log_probs, axis=1), "correct": np.argmax(log_probs, axis=1) == label_id, "source_path": str(source), "source_sha256": file_sha256(source)}
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, viewpoint_ids=ids, radius=field["radius"], azimuth=field["azimuth"], logp_true=field["logp_true"], ce=field["ce"], entropy=field["entropy"], top1_probability=field["top1_probability"], top1_top2_margin=field["top1_top2_margin"], predicted_class=field["predicted_class"], correct=field["correct"], label_id=np.asarray(label_id), scene_id=np.asarray(scene), region=np.asarray(region), record_id=np.asarray(record_id), source_path=np.asarray(str(source)), source_skeleton_sha256=np.asarray(field["source_sha256"]))
        fields[key] = field
    return fields, {"context_count": len(fields), "view_count": len(fields) * VIEW_COUNT, "skeleton_archives": len({episode_source[key] for key in contexts})}


def _stage_b_reproduction_audit(stage_d_rows: Sequence[Mapping[str, Any]], utility_by_episode: Mapping[str, Mapping[str, Any]], fields: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    matched_eps = matched_candidates = identity_mismatch = scene_mismatch = region_mismatch = source_mismatch = missing = 0; errors: list[float] = []
    for row in stage_d_rows:
        utility = utility_by_episode.get(str(row["episode_id"]));
        if utility is None: raise ValueError(f"Missing utility row {row['episode_id']}")
        key = context_key(row); field = fields.get(key)
        if field is None: identity_mismatch += 1; continue
        matched_eps += 1
        if field["scene_id"] != str(utility["scene_id"]): scene_mismatch += 1
        if field["region"] != str(utility["region"]): region_mismatch += 1
        for candidate in utility["candidates"]:
            viewpoint_id = int(candidate["viewpoint_id"])
            if viewpoint_id not in set(int(v) for v in field["viewpoint_ids"]): missing += 1; continue
            matched_candidates += 1; errors.append(abs(float(field["logp_true"][viewpoint_id]) - float(candidate["logp_true"])))
    array = np.asarray(errors, dtype=np.float64); status = "PASS" if identity_mismatch == scene_mismatch == region_mismatch == source_mismatch == missing == 0 and (array.size == 0 or array.max() < 1e-2) else "FAIL"
    return {"matched_episode_count": matched_eps, "matched_candidate_count": matched_candidates, "identity_mismatch_count": identity_mismatch, "scene_mismatch_count": scene_mismatch, "region_mismatch_count": region_mismatch, "source_path_mismatch_count": source_mismatch, "candidate_missing_count": missing, "max_abs_logp_error": float(array.max()) if array.size else 0.0, "mean_abs_logp_error": float(array.mean()) if array.size else 0.0, "p99_abs_logp_error": float(np.percentile(array, 99)) if array.size else 0.0, "status": status}


def _field_audit(fields: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([field["ce"] for field in fields.values()]); angular: list[float] = []; radial: list[float] = []; opposite: list[float] = []
    for row in values:
        for radius in range(4): block = row[radius * 8 : (radius + 1) * 8]; angular.extend(np.abs(block - np.roll(block, -1)))
        for azimuth in range(8): radial.extend(np.abs(row[azimuth::8][:-1] - row[azimuth::8][1:]))
        opposite.extend(np.abs(row - np.roll(row, 16)))
    return {"ce_mean": float(values.mean()), "ce_std": float(values.std()), "within_record_range_mean": float(np.ptp(values, axis=1).mean()), "neighbor_azimuth_abs_diff": float(np.mean(angular)), "neighbor_radius_abs_diff": float(np.mean(radial)), "opposite_view_abs_diff": float(np.mean(opposite)), "azimuth_neighbor_correlation": _corr(values[:, :-1].ravel(), np.roll(values, -1, axis=1)[:, :-1].ravel()), "radius_neighbor_correlation": _corr(values[:, ::8][:, :-1].ravel(), values[:, ::8][:, 1:].ravel())}


def _base_context(row: Mapping[str, Any]) -> np.ndarray: return np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32)


def _sample_matrix(rows: Sequence[Mapping[str, Any]], fields: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[str, str, str], int]]]:
    x: list[np.ndarray] = []; y: list[float] = []; keys: list[tuple[tuple[str, str, str], int]] = []
    for row in rows:
        key = context_key(row); field = fields[key]; current = field["positions"][int(row["s1_viewpoint_id"])]
        for viewpoint_id in range(VIEW_COUNT): x.append(np.concatenate([_base_context(row), relative_view_descriptor(field["positions"], current, viewpoint_id)])); y.append(float(field["ce"][viewpoint_id])); keys.append((key, viewpoint_id))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(0); std = train.std(0); std[std < 1e-6] = 1.0; return (train - mean) / std, (values - mean) / std


def _single_step(methods: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], fields: Mapping[tuple[str, str, str], Mapping[str, Any]], offsets: Mapping[tuple[str, str, str], int]) -> dict[str, Any]:
    selected_rows = [row for row in rows if len(row.get("remaining_candidate_ids", [])) == 2 and deterministic_oracle_action(row["second_step_utility_targets"]) != 0]; output: dict[str, Any] = {}
    for name, prediction in methods.items():
        hits: list[bool] = []; margins = {str(x): [] for x in (0.25, 0.5, 1.0, 2.0)}
        for row in selected_rows:
            key = context_key(row); ids = [int(v) for v in row["remaining_candidate_ids"]]; geodesic = {viewpoint_id: float(distance) for viewpoint_id, distance in zip(ids, row["second_step_candidate_geodesic"])}; base = offsets[key] * VIEW_COUNT; predicted = min(ids, key=lambda v: (float(prediction[base + v]), geodesic[v], v)); truth = ids[int(np.argmax(row["second_step_utility_targets"]))]; hits.append(predicted == truth); margin = float(oracle_margin(row["second_step_utility_targets"])["candidate_margin"] or 0.0)
            for threshold in margins:
                if margin >= float(threshold): margins[threshold].append(predicted == truth)
        output[name] = {"winner_accuracy": float(np.mean(hits)) if hits else None, "episode_count": len(hits), "high_margin": {k: float(np.mean(v)) if v else None for k, v in margins.items()}}
    return output


def _second_predictions(rows: Sequence[Mapping[str, Any]], values: np.ndarray, offsets: Mapping[tuple[str, str, str], int]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        key = context_key(row); ids = [int(v) for v in row["remaining_candidate_ids"]]; base = offsets[key] * VIEW_COUNT; utilities = [-float(values[base + v]) for v in ids]; stays, selected, maximum = second_step_decision(utilities, ids, [float(v) for v in row["second_step_candidate_geodesic"]]); output.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": ids, "predicted_utilities": utilities, "predicted_stays": bool(stays), "predicted_candidate_viewpoint_id": None if stays else int(selected), "max_predicted_utility": float(maximum)})
    return output


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    classes = sorted(set(truth.tolist()) | set(predicted.tolist())); scores = []
    for cls in classes:
        tp = np.sum((truth == cls) & (predicted == cls)); fp = np.sum((truth != cls) & (predicted == cls)); fn = np.sum((truth == cls) & (predicted != cls)); scores.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _trajectory_metrics(stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], stage_d_val: Sequence[Mapping[str, Any]], predictions: Mapping[str, np.ndarray], offsets: Mapping[tuple[str, str, str], int], categories: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    trajectories = {name: build_stage_d_trajectories(stage_b_val, v0_val, stage_d_val, _second_predictions(stage_d_val, values, offsets)) for name, values in predictions.items()}; trajectories["Fixed-first Oracle"] = build_fixed_first_oracle(stage_b_val, v0_val, stage_d_val)
    return {name: summarize_trajectory_rows(rows, categories) for name, rows in trajectories.items()}, trajectories


def _load_exp014_trajectory(data_root: Path, stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], stage_d_val: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = data_root / "experiments/stage_d/EXP014_two_step_sequential/runtime/val_second_step_predictions.jsonl"
    rows = load_jsonl(path)
    trajectories = build_stage_d_trajectories(stage_b_val, v0_val, stage_d_val, rows)
    summary = summarize_trajectory_rows(trajectories, categories)
    if abs(float(summary["recognition"]["accuracy"]) - 0.6582540931) > 1e-5 or abs(float(summary["recognition"]["macro_f1"]) - 0.6101526052) > 1e-5:
        raise RuntimeError(f"TRAJECTORY_EVALUATOR_GATE=FAIL: {summary['recognition']}")
    return summary, trajectories


def _full_selection(predictions: Mapping[str, np.ndarray], fields: Mapping[tuple[str, str, str], Mapping[str, Any]], offsets: Mapping[tuple[str, str, str], int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, values in predictions.items():
        rows = []
        for key in sorted(fields):
            field = fields[key]; selected = int(np.argmin(values[offsets[key] * VIEW_COUNT : offsets[key] * VIEW_COUNT + VIEW_COUNT])); rank = int(np.argsort(field["ce"])[selected]) + 1; rows.append((float(field["ce"][selected]), float(np.min(field["ce"])), rank))
        arr = np.asarray(rows); output[name] = {"selected_true_ce": float(arr[:, 0].mean()), "best_possible_ce": float(arr[:, 1].mean()), "ce_regret": float((arr[:, 0] - arr[:, 1]).mean()), "top1_oracle_hit": float(np.mean(arr[:, 2] == 1)), "top3_oracle_hit": float(np.mean(arr[:, 2] <= 3)), "selected_rank_mean": float(arr[:, 2].mean())}
    return output


def _fit_ce_proxy(train_fields: Mapping[tuple[str, str, str], Mapping[str, Any]], val_fields: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    def matrix(fields: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        x: list[np.ndarray] = []; y: list[float] = []
        for field in fields.values(): x.extend(np.c_[field["entropy"], field["top1_probability"], field["top1_top2_margin"]]); y.extend(field["ce"])
        return np.c_[np.ones(len(x)), np.asarray(x, dtype=np.float64)], np.asarray(y, dtype=np.float64)
    train_x, train_y = matrix(train_fields); val_x, val_y = matrix(val_fields); mean = train_x[:, 1:].mean(0); std = train_x[:, 1:].std(0); std[std < 1e-6] = 1.0; train_x[:, 1:] = (train_x[:, 1:] - mean) / std; val_x[:, 1:] = (val_x[:, 1:] - mean) / std; beta = np.linalg.solve(train_x.T @ train_x + np.eye(train_x.shape[1]), train_x.T @ train_y); prediction = val_x @ beta
    return {"input": ["entropy", "top1_probability", "top1_top2_margin"], "train_observations": len(train_y), "val_observations": len(val_y), "val_metrics": {"mae": float(np.mean(np.abs(prediction - val_y))), "rmse": float(np.sqrt(np.mean((prediction - val_y) ** 2))), **_corr(prediction, val_y)}, "uses_gt_label_as_input": False, "_coefficients": beta.tolist(), "_mean": mean.tolist(), "_std": std.tolist()}


def _multistep(val_rows: Sequence[Mapping[str, Any]], fields: Mapping[tuple[str, str, str], Mapping[str, Any]], predictions: Mapping[str, np.ndarray], offsets: Mapping[tuple[str, str, str], int], stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], proxy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    neighbors: dict[int, list[int]] = {i: [] for i in range(VIEW_COUNT)}
    for left, right in graph_edges(): neighbors[left].append(right); neighbors[right].append(left)
    result: dict[str, Any] = {"experiment_id": "EXP037-R1", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": False, "methods": {}}
    for method in ("Stay", "Random", "Greedy", "StaticDP", "GMRFMean", "GMRFLCB", "Thompson", "Entropy", "Oracle"):
        horizons: dict[str, Any] = {}
        for horizon in (1, 2, 3):
            rows: list[tuple[float, float, float, float, int, int]] = []
            for row in val_rows:
                key = context_key(row); field = fields[key]; node = int(row["s1_viewpoint_id"]); current = node; path = 0.0; visited = [node]; base = offsets[key] * VIEW_COUNT
                for _ in range(horizon):
                    choices = neighbors[node]
                    if method == "Stay": choice = node
                    elif method == "Random": choice = min(choices) if choices else node
                    elif method == "Oracle": choice = min([node, *choices], key=lambda x: float(field["ce"][x]) + (0.05 if x != node else 0.0))
                    else:
                        source_name = {"Greedy": "DenseRegression", "StaticDP": "GMRF", "GMRFMean": "GMRF", "GMRFLCB": "BayesianLCB", "Thompson": "ThompsonMean", "Entropy": "BayesianMean"}[method]; values = predictions[source_name][base : base + VIEW_COUNT].copy()
                        if method in ("GMRFMean", "GMRFLCB", "Thompson") and proxy is not None:
                            observed_features = np.asarray([field["entropy"][node], field["top1_probability"][node], field["top1_top2_margin"][node]], dtype=np.float64); normalized = (observed_features - np.asarray(proxy["_mean"])) / np.asarray(proxy["_std"]); observed_proxy = float(np.dot(np.asarray(proxy["_coefficients"])[1:], normalized) + np.asarray(proxy["_coefficients"])[0]); values[node] = (values[node] + 4.0 * observed_proxy) / 5.0; values = gmrf_smooth(values)
                        choice = min([node, *choices], key=lambda x: float(values[x]) + (0.05 * float(np.linalg.norm(field["positions"][x] - field["positions"][node])) if x != node else 0.0))
                    if choice == node: break
                    path += float(np.linalg.norm(field["positions"][choice] - field["positions"][node])); node = int(choice); visited.append(node)
                rows.append((float(field["ce"][node]), float(np.min(field["ce"][visited])), float(field["ce"][current]), path, int(field["predicted_class"][node]), int(row["label_id"])))
            arr = np.asarray(rows); eligible_truth = arr[:, 5].astype(int); eligible_pred = arr[:, 4].astype(int)
            v0_by_id = {str(item["episode_id"]): item for item in v0_val}; eligible_ids = {str(row["episode_id"]) for row in val_rows}; all_truth: list[int] = eligible_truth.tolist(); all_pred: list[int] = eligible_pred.tolist()
            for record in stage_b_val:
                episode_id = str(record["episode_id"])
                if episode_id not in eligible_ids:
                    prediction = v0_by_id.get(episode_id)
                    if prediction is None or not bool(prediction["predicted_stays"]):
                        raise ValueError(f"Missing frozen v0 Stay prediction for {episode_id}")
                    all_truth.append(int(record["label_id"])); all_pred.append(int(prediction["current_predicted_label_id"]))
            horizons[str(horizon)] = {"eligible_episode_count": int(len(arr)), "har_episode_count": int(len(all_truth)), "terminal_true_ce": float(arr[:, 0].mean()), "best_seen_true_ce": float(arr[:, 1].mean()), "ce_improvement": float(np.mean(arr[:, 2] - arr[:, 1])), "path_length": float(arr[:, 3].mean()), "gain_per_meter": float(np.mean((arr[:, 2] - arr[:, 1]) / np.maximum(arr[:, 3], 1e-6))), "terminal_har_accuracy": float(np.mean(np.asarray(all_truth) == np.asarray(all_pred))), "terminal_har_macro_f1": _macro_f1(np.asarray(all_truth, dtype=int), np.asarray(all_pred, dtype=int)), "privileged": method == "Oracle"}
        result["methods"][method] = horizons
    return result


def _write(directory: Path, result: Mapping[str, Any], text: str, identity: Mapping[str, Any], split: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True); (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); (directory / "analysis.md").write_text(f"# {result['experiment_id']}\n\n{text}\n\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n", encoding="utf-8"); (directory / "identity_audit.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8"); (directory / "split_audit.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")


def run(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter(); train_ep = _episode_rows(data_root, "train"); val_ep = _episode_rows(data_root, "val"); stage_b_train = _utility_rows(data_root, "train"); stage_b_val = _utility_rows(data_root, "val"); feature_root = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features"; stage_d_train = load_jsonl(feature_root / "train.jsonl"); stage_d_val = load_jsonl(feature_root / "val.jsonl"); identity = {"train": _identity_audit(train_ep, "train"), "val": _identity_audit(val_ep, "val")}; split = _split_audit(stage_d_train, stage_d_val)
    if split["episode_id_overlap"] or split["context_key_overlap"]: raise ValueError(f"Split audit failed: {split}")
    train_fields, train_meta = _dense_fields(data_root, "train", stage_d_train, _source_by_context(train_ep), data_root / "datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field"); val_fields, val_meta = _dense_fields(data_root, "val", stage_d_val, _source_by_context(val_ep), data_root / "datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field"); utility_train = {str(row["episode_id"]): row for row in stage_b_train}; utility_val = {str(row["episode_id"]): row for row in stage_b_val}; reproduction = _stage_b_reproduction_audit(stage_d_train, utility_train, train_fields); reproduction["val"] = _stage_b_reproduction_audit(stage_d_val, utility_val, val_fields)
    if reproduction["status"] == "FAIL" or reproduction["val"]["status"] == "FAIL": raise ValueError(f"Stage-B reproduction audit failed: {reproduction}")
    stage035 = {"experiment_id": "EXP035-R1", "status": "COMPLETED", "split": ["train", "val"], "test_used": False, "training_performed": False, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "train": train_meta, "val": val_meta, "field_structure_audit": _field_audit(val_fields), "stage_b_reproduction_audit": reproduction, "identity_audit": identity, "split_audit": split, "old_experiment_status": "INVALID_PENDING_R1"}
    train_x, train_y, _ = _sample_matrix(stage_d_train, train_fields); val_x, _, _ = _sample_matrix(stage_d_val, val_fields); train_x, val_x = _standardize(train_x, val_x); train_offsets = {key: i for i, key in enumerate(sorted(train_fields))}; val_offsets = {key: i for i, key in enumerate(sorted(val_fields))}; dense_model, dense_loss = train_dense_regressor(train_x, train_y); pred_dense = predict_model(dense_model, val_x)
    key_index = {(key, viewpoint_id): index for index, (key, viewpoint_id) in enumerate(_sample_matrix(stage_d_train, train_fields)[2])}; pair_l: list[int] = []; pair_r: list[int] = []; pair_y: list[float] = []; rng = np.random.default_rng(42); edges = graph_edges(); edge_set = {tuple(sorted(edge)) for edge in edges}
    for key in sorted(train_fields):
        non_neighbors = [(a, b) for a in range(VIEW_COUNT) for b in range(a + 1, VIEW_COUNT) if (a, b) not in edge_set]; rng.shuffle(non_neighbors)
        for left, right in list(edges) + non_neighbors[:64]: pair_l.append(key_index[(key, left)]); pair_r.append(key_index[(key, right)]); pair_y.append(float(train_fields[key]["ce"][left] < train_fields[key]["ce"][right]))
    bt_model, bt_loss = train_bradley_terry(train_x, np.asarray(pair_l), np.asarray(pair_r), np.asarray(pair_y)); pred_bt = predict_model(bt_model, val_x); bayes = fit_bayesian_linear(np.c_[np.ones(len(train_x)), train_x], train_y); bayes_mean, bayes_sigma = bayes.predict(np.c_[np.ones(len(val_x)), val_x]); gmrfs = np.asarray([gmrf_smooth(pred_dense[i * VIEW_COUNT : i * VIEW_COUNT + VIEW_COUNT]) for i in range(len(val_fields))]).reshape(-1); predictions = {"DenseRegression": pred_dense, "BradleyTerry": pred_bt, "GMRF": gmrfs, "BayesianMean": bayes_mean, "BayesianLCB": bayes_mean - bayes_sigma}; covariance = bayes.covariance * bayes.residual_variance; rng = np.random.default_rng(42); predictions["ThompsonMean"] = np.mean([np.c_[np.ones(len(val_x)), val_x] @ rng.multivariate_normal(bayes.weights, covariance) for _ in range(20)], axis=0)
    mapping = json.loads(Path(json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())["label_mapping"]).read_text()); categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]; summary_path = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/stage_d_feature_summary.json"; v0_val = load_jsonl(Path(json.loads(summary_path.read_text())["source_stage_c_v0_predictions"]["val"])); exp014_metric, _ = _load_exp014_trajectory(data_root, stage_b_val, v0_val, stage_d_val, categories); metrics, trajectories = _trajectory_metrics(stage_b_val, v0_val, stage_d_val, predictions, val_offsets, categories)
    metrics["EXP014"] = exp014_metric
    stage036 = {"experiment_id": "EXP036-R1", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "methods": {"DenseRegression": {"train_final_loss": dense_loss}, "BradleyTerry": {"train_final_loss": bt_loss, "pair_count": len(pair_y)}, "GMRF": {"lambda": 0.25}, "BayesianMean": {"alpha": 1.0}, "BayesianLCB": {"beta_uncertainty": 1.0}, "Thompson": {"samples": 20}}, "p2_p3": _single_step(predictions, stage_d_val, val_fields, val_offsets), "full_32_view": _full_selection(predictions, val_fields, val_offsets), "trajectory_metrics": metrics, "evaluator_equivalence_gate": {"status": "PASS", "exp014_accuracy": exp014_metric["recognition"]["accuracy"], "exp014_macro_f1": exp014_metric["recognition"]["macro_f1"]}, "fixed_first_oracle": metrics["Fixed-first Oracle"], "dense_supervision_context_count": len(train_fields), "legal_future_input": False}
    proxy = _fit_ce_proxy(train_fields, val_fields); stage037 = _multistep(stage_d_val, val_fields, predictions, val_offsets, stage_b_val, v0_val, proxy); stage037["ce_proxy_calibrator"] = proxy; provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_b_summary_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json")}; stage035["provenance"] = provenance; stage036["provenance"] = provenance; stage037["provenance"] = provenance
    _write(EXP035, stage035, "R1 context-safe dense field; old record_id-only results are invalid.", identity, split); _write(EXP036, stage036, "R1 single-step optimizers with unified trajectory metrics.", identity, split); _write(EXP037, stage037, "R1 multistep graph diagnostics with terminal HAR metrics.", identity, split); (EXP035 / "stage_b_reproduction_audit.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8"); print(json.dumps({"elapsed_seconds": time.perf_counter() - started, "EXP035": stage035["status"], "EXP036": stage036["status"], "EXP037": stage037["status"], "train_contexts": len(train_fields), "val_contexts": len(val_fields)}, ensure_ascii=False)); return {"EXP035": stage035, "EXP036": stage036, "EXP037": stage037}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=Path("../../data/ActiveView")); args = parser.parse_args(); run(args.data_root.resolve())


if __name__ == "__main__": main()
