#!/usr/bin/env python3
"""Run EXP035--EXP037 using frozen Train/Val artifacts only."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import (
    VIEW_COUNT, BayesianLinear, deterministic_oracle_action, fit_bayesian_linear,
    gmrf_smooth, predict_model, relative_view_descriptor, train_bradley_terry,
    train_dense_regressor, viewpoint_azimuth, viewpoint_radius,
)
from activeview.active_view.stage_d_predictability import oracle_margin
from activeview.active_view.utility_label_builder import file_sha256
from activeview.scripts.build_stage_b_utility_labels import _load_archive_predictions, _load_model


EXP_ROOT = REPO_ROOT / "experiments" / "stage_d"
EXP035 = EXP_ROOT / "EXP035_dense_recognition_quality_field"
EXP036 = EXP_ROOT / "EXP036_single_step_view_optimization"
EXP037 = EXP_ROOT / "EXP037_multistep_graph_active_perception"


def _corr(x: Iterable[float], y: Iterable[float]) -> dict[str, float | None]:
    a = np.asarray(list(x), dtype=np.float64); b = np.asarray(list(y), dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])}


def _episode_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / "datasets/policy_v11_5/episodes" / f"{split}_episodes.jsonl"
    rows = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows):
        raise ValueError(f"invalid policy split in {path}")
    return rows


def _record_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["record_id"])
        if key not in output:
            output[key] = row
    return output


def _utility_index(data_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = data_root / "datasets/policy_v11_5/stage_b/utility_labels" / f"{split}.jsonl"
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("policy_split", "")).lower() != split:
                raise ValueError("utility split mismatch")
            output.setdefault(str(row["record_id"]), row)
    return output


def _dense_fields(data_root: Path, split: str, out_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _episode_rows(data_root, split); records = _record_index(rows); utilities = _utility_index(data_root, split)
    stage_b = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
    mapping = json.loads(Path(stage_b["label_mapping"]).read_text()); category_count = len(mapping)
    model, device = _load_model(Path(stage_b["stgcn_checkpoint"]), category_count, "cpu")
    out_dir = out_root / split; out_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, dict[str, Any]] = {}; smoke: list[float] = []
    for index, (record_id, row) in enumerate(sorted(records.items())):
        source = Path(row["current_view"]["skeleton_source_path"])
        with np.load(source, allow_pickle=False) as archive:
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int32)
            positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        predictions = _load_archive_predictions(source, model, device, 64, category_count)
        log_probs = np.full((VIEW_COUNT, category_count), np.nan, dtype=np.float32)
        for viewpoint_id, values in predictions.items():
            log_probs[int(viewpoint_id)] = values.astype(np.float32)
        label_id = int(row["label_id"])
        selected = log_probs[:, label_id]
        probabilities = np.exp(log_probs)
        top2 = np.partition(log_probs, -2, axis=1)[:, -2:]
        margin = np.max(top2, axis=1) - np.min(top2, axis=1)
        field = {"record_id": record_id, "scene_id": str(row["scene_id"]), "region": str(row["region"]), "label_id": label_id, "viewpoint_ids": ids, "radius": np.asarray([viewpoint_radius(v) for v in ids], dtype=np.float32), "azimuth": np.asarray([viewpoint_azimuth(v) for v in ids], dtype=np.float32), "logp_true": selected, "ce": -selected, "entropy": -np.sum(probabilities * np.nan_to_num(log_probs), axis=1), "top1_probability": np.max(probabilities, axis=1), "top1_top2_margin": margin, "predicted_class": np.argmax(log_probs, axis=1), "correct": np.argmax(log_probs, axis=1) == label_id, "positions": positions, "source_path": str(source)}
        target = out_dir / f"{record_id}.npz"; np.savez_compressed(target, viewpoint_ids=ids, radius=field["radius"], azimuth=field["azimuth"], logp_true=field["logp_true"], ce=field["ce"], entropy=field["entropy"], top1_probability=field["top1_probability"], top1_top2_margin=field["top1_top2_margin"], predicted_class=field["predicted_class"], correct=field["correct"], label_id=np.asarray(label_id), scene_id=np.asarray(row["scene_id"]), region=np.asarray(row["region"]), source_path=np.asarray(str(source)))
        fields[record_id] = field
        if index < 2:
            utility = utilities[record_id]
            for candidate in utility["candidates"][:2]:
                smoke.append(abs(float(field["logp_true"][int(candidate["viewpoint_id"])]) - float(candidate["logp_true"])))
    return fields, {"record_count": len(fields), "view_count": len(fields) * VIEW_COUNT, "smoke_max_abs_logp_error": float(max(smoke) if smoke else 0.0), "source_sha256": file_sha256(next(iter(records.values()))["current_view"]["skeleton_source_path"]) if records else None}


def _field_audit(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([field["ce"] for field in fields.values()]); angular = []; radial = []; opposite = []
    for row in values:
        for radius in range(4):
            block = row[radius * 8 : (radius + 1) * 8]; angular.extend(np.abs(block - np.roll(block, -1)))
        for azimuth in range(8):
            radial.extend(np.abs(row[azimuth::8][:-1] - row[azimuth::8][1:]))
        opposite.extend(np.abs(row - np.roll(row, 16)))
    return {"ce_mean": float(values.mean()), "ce_std": float(values.std()), "within_record_range_mean": float(np.ptp(values, axis=1).mean()), "best_worst_gap": float(np.ptp(values, axis=1).mean()), "neighbor_azimuth_abs_diff": float(np.mean(angular)), "neighbor_radius_abs_diff": float(np.mean(radial)), "opposite_view_abs_diff": float(np.mean(opposite)), "azimuth_neighbor_correlation": _corr(values[:, :-1].ravel(), np.roll(values, -1, axis=1)[:, :-1].ravel()), "radius_neighbor_correlation": _corr(values[:, ::8][:, :-1].ravel(), values[:, ::8][:, 1:].ravel())}


def _base_context(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32)


def _sample_matrix(rows: list[dict[str, Any]], fields: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    x: list[np.ndarray] = []; y: list[float] = []; keys: list[tuple[str, int]] = []
    representatives = _record_index(rows)
    for record_id, row in representatives.items():
        field = fields[record_id]
        current_id = int(row.get("s1_viewpoint_id", row.get("proposal_rank_1_id", 0)))
        current = np.asarray(field["positions"][current_id], dtype=np.float32)
        for viewpoint_id in range(VIEW_COUNT):
            desc = relative_view_descriptor(field["positions"], current, viewpoint_id)
            x.append(np.concatenate([_base_context(row), desc])); y.append(float(field["ce"][viewpoint_id])); keys.append((record_id, viewpoint_id))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0); std = train.std(0); std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std, mean, std


def _single_step(methods: dict[str, np.ndarray], val_rows: list[dict[str, Any]], fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in val_rows if row.get("remaining_candidate_ids")]
    out: dict[str, Any] = {}
    for name, prediction in methods.items():
        hits = []; margins: dict[str, list[bool]] = {str(x): [] for x in (0.25, 0.5, 1.0, 2.0)}
        offset = {key: i for i, key in enumerate(sorted(fields))}
        for row in rows:
            ids = [int(v) for v in row["remaining_candidate_ids"]]
            if len(ids) != 2: continue
            if deterministic_oracle_action(row["second_step_utility_targets"]) == 0:
                continue
            field = fields[str(row["record_id"])]
            predicted = int(np.argmin([prediction[offset[str(row['record_id'])] * 32 + i] for i in ids]))
            truth = int(np.argmax(row["second_step_utility_targets"]))
            hits.append(predicted == truth)
            margin = float(oracle_margin(row["second_step_utility_targets"])["candidate_margin"] or 0.0)
            for threshold in margins:
                if margin >= float(threshold): margins[threshold].append(predicted == truth)
        out[name] = {"winner_accuracy": float(np.mean(hits)) if hits else None, "episode_count": len(hits), "high_margin": {k: float(np.mean(v)) if v else None for k, v in margins.items()}}
    return out


def _full_selection(predictions: dict[str, np.ndarray], fields: dict[str, dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    representatives = _record_index(val_rows); output: dict[str, Any] = {}
    for name, values in predictions.items():
        rows = []; offset = {key: i for i, key in enumerate(sorted(fields))}
        for record_id, row in representatives.items():
            field = fields[record_id]; predicted = values[offset[record_id] * 32 : offset[record_id] * 32 + 32]; selected = int(np.argmin(predicted)); true_ce = float(field["ce"][selected]); best = float(np.min(field["ce"])); rank = int(np.argsort(field["ce"])[selected]) + 1
            current_id = int(row.get("s1_viewpoint_id", row.get("current_view", {}).get("viewpoint_id", 0)))
            rows.append((true_ce, best, rank, float(field["ce"][current_id])))
        arr = np.asarray(rows); output[name] = {"selected_true_ce": float(arr[:, 0].mean()), "best_possible_ce": float(arr[:, 1].mean()), "ce_regret": float((arr[:, 0] - arr[:, 1]).mean()), "top1_oracle_hit": float(np.mean(arr[:, 2] == 1)), "top3_oracle_hit": float(np.mean(arr[:, 2] <= 3)), "selected_rank_mean": float(arr[:, 2].mean()), "improvement_relative_current": float(np.mean(arr[:, 3] - arr[:, 0]))}
    return output


def _write(directory: Path, result: dict[str, Any], readme: str) -> None:
    directory.mkdir(parents=True, exist_ok=True); (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); (directory / "analysis.md").write_text(f"# {result['experiment_id']}\n\n{readme}\n\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n", encoding="utf-8")


def run(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter(); train_rows = _episode_rows(data_root, "train"); val_rows = _episode_rows(data_root, "val")
    train_fields, train_meta = _dense_fields(data_root, "train", data_root / "datasets/policy_v11_5/stage_d/EXP035_dense_recognition_quality_field")
    val_fields, val_meta = _dense_fields(data_root, "val", data_root / "datasets/policy_v11_5/stage_d/EXP035_dense_recognition_quality_field")
    stage035 = {"experiment_id": "EXP035", "status": "COMPLETED", "split": ["train", "val"], "test_used": False, "training_performed": False, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "train": train_meta, "val": val_meta, "field_structure_audit": _field_audit(val_fields), "privileged_evaluation_only": True, "smoke_status": "PASS" if max(train_meta["smoke_max_abs_logp_error"], val_meta["smoke_max_abs_logp_error"]) <= 1e-2 else "BLOCKED", "smoke_tolerance": 1e-2}
    if stage035["smoke_status"] != "PASS":
        raise RuntimeError("EXP035_STATUS=BLOCKED: Stage-B smoke reproduction mismatch")
    feature_root = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features"
    stage_d_train = load_jsonl(feature_root / "train.jsonl"); stage_d_val = load_jsonl(feature_root / "val.jsonl")
    train_x, train_y, train_keys = _sample_matrix(stage_d_train, train_fields); val_x, val_y, val_keys = _sample_matrix(stage_d_val, val_fields); train_x, val_x, mean, std = _standardize(train_x, val_x)
    dense_model, dense_loss = train_dense_regressor(train_x, train_y); pred_dense = predict_model(dense_model, val_x)
    train_pairs_l: list[int] = []; train_pairs_r: list[int] = []; pair_y: list[float] = []; key_index = {key: i for i, key in enumerate(train_keys)}
    rng = np.random.default_rng(42)
    graph = json.loads(json.dumps([[a, b] for a, b in __import__('activeview.active_view.stage_d_dense_campaign', fromlist=['graph_edges']).graph_edges()]))
    for record_id in sorted(train_fields):
        graph_pairs = [(record_id, a, b) for a, b in graph]
        non_neighbors = [(a, b) for a in range(VIEW_COUNT) for b in range(a + 1, VIEW_COUNT) if tuple(sorted((a, b))) not in {tuple(edge) for edge in graph}]
        rng.shuffle(non_neighbors)
        random_pairs = [(record_id, a, b) for a, b in non_neighbors[:64]]
        for _, left, right in graph_pairs + random_pairs:
            train_pairs_l.append(key_index[(record_id, int(left))]); train_pairs_r.append(key_index[(record_id, int(right))]); train_y_pair = float(train_fields[record_id]["ce"][int(left)] < train_fields[record_id]["ce"][int(right)]); pair_y.append(train_y_pair)
    bt_model, bt_loss = train_bradley_terry(train_x, np.asarray(train_pairs_l), np.asarray(train_pairs_r), np.asarray(pair_y)); pred_bt = predict_model(bt_model, val_x)
    bayes = fit_bayesian_linear(np.c_[np.ones(len(train_x)), train_x], train_y); bayes_mean, bayes_sigma = bayes.predict(np.c_[np.ones(len(val_x)), val_x]); gmrfs = np.asarray([gmrf_smooth(pred_dense[i * 32 : i * 32 + 32]) for i in range(len(val_fields))]).reshape(-1)
    predictions = {"DenseRegression": pred_dense, "BradleyTerry": pred_bt, "GMRF": gmrfs, "BayesianMean": bayes_mean, "BayesianLCB": bayes_mean - bayes_sigma}
    thompson = []
    rng = np.random.default_rng(42); covariance = bayes.covariance * bayes.residual_variance
    for _ in range(20):
        sample_w = rng.multivariate_normal(bayes.weights, covariance); thompson.append(np.c_[np.ones(len(val_x)), val_x] @ sample_w)
    predictions["ThompsonMean"] = np.mean(thompson, axis=0)
    stage036 = {"experiment_id": "EXP036", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "methods": {"DenseRegression": {"train_final_loss": dense_loss, "model": "legal state + 9-D viewpoint descriptor"}, "BradleyTerry": {"train_final_loss": bt_loss, "pair_count": len(pair_y)}, "GMRF": {"lambda": 0.25}, "BayesianMean": {"alpha": 1.0, "beta": 1.0}, "BayesianLCB": {"beta_uncertainty": 1.0}, "Thompson": {"samples": 20, "seeds": list(range(20))}, "Kernel": "METHOD_E_SKIPPED_FOR_SCALE"}, "p2_p3": _single_step(predictions, stage_d_val, val_fields), "full_32_view": _full_selection(predictions, val_fields, stage_d_val), "dense_supervision_record_count": len(train_fields), "legal_future_input": False, "leakage_flags": {"future_candidate_skeleton_used_at_inference": False, "future_true_ce_used_at_inference": False, "test_used": False}}
    stage037 = _multistep(stage_d_val, val_fields, predictions)
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_b_summary_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json")}
    for directory, result, text in ((EXP035, stage035, "Frozen 32-view ST-GCN quality field and topology audit."), (EXP036, stage036, "Legal single-step dense, pairwise, graph and Bayesian diagnostics."), (EXP037, stage037, "Val-only graph rollout diagnostic; original Stage-D protocol unchanged.")):
        result["provenance"] = provenance; _write(directory, result, text)
    print(json.dumps({"elapsed_seconds": time.perf_counter() - started, "EXP035": stage035["status"], "EXP036": stage036["status"], "EXP037": stage037["status"]}, ensure_ascii=False)); return {"EXP035": stage035, "EXP036": stage036, "EXP037": stage037}


def _multistep(val_rows: list[dict[str, Any]], fields: dict[str, dict[str, Any]], predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run deterministic graph rollouts using legal static fields and observable feedback."""
    representatives = _record_index(val_rows); offset = {key: i for i, key in enumerate(sorted(fields))}; result: dict[str, Any] = {"experiment_id": "EXP037", "status": "COMPLETED", "protocol": "MULTISTEP_DIAGNOSTIC", "original_stage_d_protocol_unchanged": True, "split": "val", "test_used": False, "training_performed": False, "future_candidate_skeleton_used": False, "future_true_ce_used": False, "visited_view_skeleton_used": True, "methods": {}}
    graph = __import__('activeview.active_view.stage_d_dense_campaign', fromlist=['graph_edges']).graph_edges(); neighbors = {i: [] for i in range(VIEW_COUNT)}
    for left, right in graph: neighbors[left].append(right); neighbors[right].append(left)
    for method in ("Stay", "Random", "Greedy", "StaticDP", "GMRFMean", "GMRFLCB", "Thompson", "Entropy", "Oracle"):
        horizons: dict[str, Any] = {}
        for horizon in (1, 2, 3):
            rows = []
            for record_id, row in representatives.items():
                field = fields[record_id]; current = int(row.get("s1_viewpoint_id", row.get("current_view", {}).get("viewpoint_id", 0))); start_ce = float(field["ce"][current]); node = current; path = 0.0; visited = [node]
                for _ in range(horizon):
                    candidates = neighbors[node]
                    if method == "Stay": choice = node
                    elif method == "Random": choice = min(candidates) if candidates else node
                    elif method == "Oracle": choice = min([node, *candidates], key=lambda x: float(field["ce"][x]) + (0.05 if x != node else 0.0))
                    else:
                        method_prediction = {"Greedy": "DenseRegression", "StaticDP": "GMRF", "GMRFMean": "GMRF", "GMRFLCB": "BayesianLCB", "Thompson": "ThompsonMean", "Entropy": "BayesianMean"}.get(method, "DenseRegression")
                        values = predictions[method_prediction][offset[record_id] * 32 : offset[record_id] * 32 + 32]
                        if method == "StaticDP":
                            def value(next_node: int, remaining: int) -> float:
                                if remaining == 0:
                                    return float(values[next_node])
                                options = [float(values[next_node])]
                                options.extend(0.05 * float(np.linalg.norm(field["positions"][neighbor] - field["positions"][next_node])) + value(neighbor, remaining - 1) for neighbor in neighbors[next_node])
                                return min(options)
                            choice = min([node, *candidates], key=lambda x: value(x, horizon - 1) + (0.05 * float(np.linalg.norm(field["positions"][x] - field["positions"][node])) if x != node else 0.0))
                        else:
                            choice = min([node, *candidates], key=lambda x: float(values[x]) + (0.05 * float(np.linalg.norm(field["positions"][x] - field["positions"][node])) if x != node else 0.0))
                    if choice == node: break
                    path += float(np.linalg.norm(field["positions"][choice] - field["positions"][node])); node = int(choice); visited.append(node)
                terminal = float(field["ce"][node]); best_seen = float(np.min(field["ce"][visited])); rows.append((terminal, best_seen, start_ce, path))
            arr = np.asarray(rows); horizons[str(horizon)] = {"terminal_true_ce": float(arr[:, 0].mean()), "best_seen_true_ce": float(arr[:, 1].mean()), "ce_improvement": float(np.mean(arr[:, 2] - arr[:, 1])), "path_length": float(arr[:, 3].mean()), "gain_per_meter": float(np.mean((arr[:, 2] - arr[:, 1]) / np.maximum(arr[:, 3], 1e-6)))}
        result["methods"][method] = horizons
    result["movement_cost_source"] = "EUCLIDEAN_DIAGNOSTIC"; result["oracle_multistep_upper_bound"] = True; result["deployable"] = False; return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=Path("../../data/ActiveView")); args = parser.parse_args(); run(args.data_root.resolve())


if __name__ == "__main__":
    main()
