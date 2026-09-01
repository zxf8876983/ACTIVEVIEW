#!/usr/bin/env python3
"""Run the EXP038--EXP040 observability and belief-space campaign (Train/Val only)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import tempfile
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_belief import (
    action_diagnostics,
    belief_from_log_probs,
    fuse_beliefs,
    masked_binary_loss,
    masked_regression_loss,
    oracle_action,
    top_k_belief,
)
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import (
    VIEW_COUNT,
    canonical_realpath,
    context_key,
    graph_edges,
    relative_view_descriptor,
)
from activeview.active_view.stage_d_evaluation import (
    build_fixed_first_oracle,
    build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_policy import second_step_decision, semantic_delta
from activeview.active_view.utility_label_builder import file_sha256
from activeview.scripts.build_stage_b_utility_labels import _load_model, _load_archive_predictions
from activeview.perception.skeleton_definition import get_skeleton_definition


EXP_ROOT = REPO_ROOT / "experiments" / "stage_d"
EXP038 = EXP_ROOT / "EXP038_oracle_observability_ladder"
EXP039 = EXP_ROOT / "EXP039_belief_conditioned_view_risk"
EXP040 = EXP_ROOT / "EXP040_belief_space_active_har"
MARGIN_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Context:
    row: Mapping[str, Any]
    field: Mapping[str, Any]
    source_path: Path


def _split_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features" / f"{split}.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        if str(row.get("policy_split", "")).lower() != split:
            raise ValueError(f"{path} contains a row without explicit {split} policy_split")
    if split == "test":
        raise ValueError("Test is permanently locked")
    return rows


def _utility_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise ValueError("Test is permanently locked")
    path = data_root / "datasets/policy_v11_5/stage_b/utility_labels" / f"{split}.jsonl"
    rows = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows):
        raise ValueError(f"invalid Stage-B split in {path}")
    return rows


def _load_contexts(data_root: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Context]:
    root = data_root / "datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field" / split
    output: dict[tuple[str, str, str], Context] = {}
    for index, row in enumerate(rows, start=1):
        key = context_key(row)
        if key in output:
            raise ValueError(f"duplicate context key {key}")
        source = root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            field = {name: np.asarray(archive[name]) for name in ("ce", "predicted_class", "label_id")}
            field["source_path"] = str(np.asarray(archive["source_path"]).item())
            field["viewpoint_ids"] = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        with np.load(Path(field["source_path"]), allow_pickle=False) as skeleton_archive:
            field["positions"] = np.asarray(skeleton_archive["viewpoint_agent_positions"], dtype=np.float32)
        if field["ce"].shape != (VIEW_COUNT,) or field["positions"].shape != (VIEW_COUNT, 3):
            raise ValueError(f"invalid dense field schema: {source}")
        output[key] = Context(row=row, field=field, source_path=Path(field["source_path"]))
        if index % 2000 == 0:
            print(f"Loaded {split} dense contexts: {index}/{len(rows)}", flush=True)
    return output


def _base(row: Mapping[str, Any], current_feature: Sequence[float] | None = None) -> np.ndarray:
    s0 = np.asarray(row["s0_feature"], dtype=np.float32)
    s1 = np.asarray(row["s1_feature"] if current_feature is None else current_feature, dtype=np.float32)
    delta = semantic_delta(s0, s1)
    # Keep the pilot state compact while retaining the complete 16-D beliefs,
    # entropy, margin and pose confidence from both visited views plus delta.
    # This is a legal observable subset; no future quality is included.
    return np.concatenate([s0[256:], s1[256:], delta]).astype(np.float32)


def _view_ids(row: Mapping[str, Any]) -> list[int]:
    ids = [int(row["s1_viewpoint_id"])] + [int(item) for item in row["remaining_candidate_ids"]]
    return list(dict.fromkeys(ids))


def _samples(contexts: Mapping[tuple[str, str, str], Context], include_label: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[tuple[str, str, str], int]]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    labels: list[int] = []
    keys: list[tuple[tuple[str, str, str], int]] = []
    for key in sorted(contexts):
        item = contexts[key]; row = item.row; field = item.field; current = field["positions"][int(row["s1_viewpoint_id"])]
        base = _base(row)
        for viewpoint_id in _view_ids(row):
            descriptor = relative_view_descriptor(field["positions"], current, viewpoint_id)
            values = [base, descriptor]
            if include_label:
                one_hot = np.zeros(16, dtype=np.float32); one_hot[int(row["label_id"])] = 1.0; values.append(one_hot)
            features.append(np.concatenate(values).astype(np.float32)); targets.append(float(field["ce"][viewpoint_id])); labels.append(int(row["label_id"])); keys.append((key, viewpoint_id))
    return np.asarray(features), np.asarray(targets, dtype=np.float32), np.asarray(labels, dtype=np.int64), keys


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0); std = train.std(axis=0); std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _train_scalar(train_x: np.ndarray, train_y: np.ndarray, *, epochs: int = 20, batch_size: int = 1024) -> tuple[Any, float]:
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); criterion = nn.SmoothL1Loss(); x = torch.from_numpy(train_x); y = torch.from_numpy(train_y).reshape(-1, 1); final = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(x)); total = 0.0
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]; loss = criterion(model(x[idx]), y[idx]); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(idx)
        final = total / len(x)
    return model, final


def _predict_scalar(model: Any, values: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(values.astype(np.float32))).reshape(-1).numpy()


def _train_heads(train_x: np.ndarray, train_y: np.ndarray, labels: np.ndarray, *, binary: bool = False, epochs: int = 20) -> tuple[Any, float]:
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); x = torch.from_numpy(train_x); y = torch.from_numpy(train_y); cls = torch.from_numpy(labels); final = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(x)); total = 0.0
        for start in range(0, len(x), 1024):
            idx = order[start:start + 1024]; prediction = model(x[idx]);
            loss = masked_binary_loss(prediction, y[idx], cls[idx]) if binary else masked_regression_loss(prediction, y[idx], cls[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(idx)
        final = total / len(x)
    return model, final


def _predict_context(model: Any, contexts: Mapping[tuple[str, str, str], Context], stats: tuple[np.ndarray, np.ndarray], include_label: bool, *, heads: bool = False) -> dict[tuple[str, str, str], dict[int, np.ndarray | float]]:
    mean, std = stats; output: dict[tuple[str, str, str], dict[int, np.ndarray | float]] = {}; raw_rows: list[np.ndarray] = []; row_keys: list[tuple[tuple[str, str, str], int]] = []
    for key in sorted(contexts):
        item = contexts[key]; row = item.row; current = item.field["positions"][int(row["s1_viewpoint_id"])]
        for viewpoint_id in _view_ids(row):
            values = [_base(row), relative_view_descriptor(item.field["positions"], current, viewpoint_id)]
            if include_label:
                one_hot = np.zeros(16, dtype=np.float32); one_hot[int(row["label_id"])] = 1.0; values.append(one_hot)
            raw_rows.append(np.concatenate(values).astype(np.float32)); row_keys.append((key, viewpoint_id))
    prediction = model_prediction(model, (np.asarray(raw_rows) - mean) / std, heads=heads)
    for index, (key, viewpoint_id) in enumerate(row_keys):
        output.setdefault(key, {})[viewpoint_id] = prediction[index]
    return output


def model_prediction(model: Any, values: np.ndarray, *, heads: bool) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        result = model(torch.from_numpy(values.astype(np.float32))).numpy()
    return result if heads else result.reshape(-1)


def _predict_all_contexts(model: Any, contexts: Mapping[tuple[str, str, str], Context], stats: tuple[np.ndarray, np.ndarray], *, heads: bool) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    """Batch frozen-model predictions for all 32 graph viewpoints per context."""
    mean, std = stats; result: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}; keys = sorted(contexts)
    for start in range(0, len(keys), 128):
        chunk = keys[start:start + 128]; raw: list[np.ndarray] = []; locations: list[tuple[tuple[str, str, str], int]] = []
        for key in chunk:
            item = contexts[key]; row = item.row; current = item.field["positions"][int(row["s1_viewpoint_id"])]
            base = _base(row)
            for viewpoint_id in range(VIEW_COUNT):
                raw.append(np.concatenate([base, relative_view_descriptor(item.field["positions"], current, viewpoint_id)]).astype(np.float32)); locations.append((key, viewpoint_id))
        prediction = model_prediction(model, (np.asarray(raw) - mean) / std, heads=heads)
        for index, (key, viewpoint_id) in enumerate(locations): result.setdefault(key, {})[viewpoint_id] = np.asarray(prediction[index])
    return result


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target; rx = np.argsort(np.argsort(prediction)); ry = np.argsort(np.argsort(target))
    return {"n": int(len(target)), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson": float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) and np.std(target) else None, "spearman": float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else None}


def _candidate_diagnostics(contexts: Mapping[tuple[str, str, str], Context], predictions: Mapping[tuple[str, str, str], Mapping[int, float]]) -> dict[str, Any]:
    hits: list[bool] = []; margins = {str(v): [] for v in MARGIN_THRESHOLDS}
    for key, item in contexts.items():
        row = item.row; ids = [int(v) for v in row["remaining_candidate_ids"]]
        if len(ids) != 2:
            continue
        current = float(item.field["ce"][int(row["s1_viewpoint_id"])])
        true_u = [current - float(item.field["ce"][candidate]) for candidate in ids]
        if oracle_action(true_u) == 0:
            continue
        predicted = min(ids, key=lambda v: (float(predictions[key][v]), v))
        truth = ids[int(np.argmax(true_u))]; hits.append(predicted == truth)
        margin = abs(true_u[0] - true_u[1])
        for threshold in margins:
            if margin >= float(threshold): margins[threshold].append(predicted == truth)
    return {"winner_accuracy": float(np.mean(hits)) if hits else None, "episode_count": len(hits), "high_margin": {k: {"count": len(v), "accuracy": float(np.mean(v)) if v else None} for k, v in margins.items()}}


def _second_predictions(rows: Sequence[Mapping[str, Any]], contexts: Mapping[tuple[str, str, str], Context], predictions: Mapping[tuple[str, str, str], Mapping[int, float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        key = context_key(row); item = contexts[key]; current_id = int(row["s1_viewpoint_id"]); current = float(predictions[key][current_id]); ids = [int(v) for v in row["remaining_candidate_ids"]]; utilities = [current - float(predictions[key][candidate]) for candidate in ids]; geodesics = [float(v) for v in row["second_step_candidate_geodesic"]]; stays, selected, maximum = second_step_decision(utilities, ids, geodesics); output.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": ids, "predicted_utilities": utilities, "predicted_stays": bool(stays), "predicted_candidate_viewpoint_id": None if stays else int(selected), "max_predicted_utility": float(maximum)})
    return output


def _trajectory_summary(stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], stage_d_val: Sequence[Mapping[str, Any]], categories: Sequence[str], contexts: Mapping[tuple[str, str, str], Context], predictions: Mapping[tuple[str, str, str], Mapping[int, float]]) -> dict[str, Any]:
    rows = _second_predictions(stage_d_val, contexts, predictions)
    trajectories = build_stage_d_trajectories(stage_b_val, v0_val, stage_d_val, rows); oracle = build_fixed_first_oracle(stage_b_val, v0_val, stage_d_val)
    return {"learned": summarize_trajectory_rows(trajectories, categories), "fixed_first_oracle": summarize_trajectory_rows(oracle, categories)}


def _class_prior(contexts: Mapping[tuple[str, str, str], Context]) -> np.ndarray:
    sums = np.zeros((16, VIEW_COUNT), dtype=np.float64); counts = np.zeros(16, dtype=np.float64)
    for item in contexts.values():
        label = int(item.row["label_id"]); sums[label] += item.field["ce"]; counts[label] += 1.0
    return sums / np.maximum(counts[:, None], 1.0)


def _exp038(data_root: Path, train: Mapping[tuple[str, str, str], Context], val: Mapping[tuple[str, str, str], Context], stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> dict[str, Any]:
    train_x, train_y, train_labels, _ = _samples(train, False); val_x, val_y, val_labels, val_keys = _samples(val, False); train_x, val_x, mean, std = _standardize(train_x, val_x)
    l0, loss0 = _train_scalar(train_x, train_y); pred0 = _predict_context(l0, val, (mean, std), False)
    train_lx, _, _, _ = _samples(train, True); val_lx, _, _, _ = _samples(val, True); train_lx, val_lx, lmean, lstd = _standardize(train_lx, val_lx); l1, loss1 = _train_scalar(train_lx, train_y); pred1 = _predict_context(l1, val, (lmean, lstd), True)
    target = np.asarray([float(val[key].field["ce"][viewpoint_id]) for key, viewpoint_id in val_keys], dtype=np.float32); flat0 = np.asarray([float(pred0[key][viewpoint_id]) for key, viewpoint_id in val_keys]); flat1 = np.asarray([float(pred1[key][viewpoint_id]) for key, viewpoint_id in val_keys])
    prior = _class_prior(train)
    prior_pred = {key: {viewpoint_id: float(prior[int(item.row["label_id"]), viewpoint_id]) for viewpoint_id in _view_ids(item.row)} for key, item in val.items()}
    v0_summary = summarize_trajectory_rows(build_stage_d_trajectories(stage_b_val, v0_val, [], []), categories) if False else None
    trajectories: dict[str, Any] = {}
    for name, pred in (("L0_LEGAL", pred0), ("L1_GT_LABEL", pred1), ("CLASS_VIEW_PRIOR_ORACLE_LABEL", prior_pred)):
        summary = _trajectory_summary(stage_b_val, v0_val, list(item.row for item in val.values()), categories, val, pred)
        trajectories[name] = summary["learned"]
    blocked = {"status": "GT_MOTION_STATE_BLOCKED", "reason": "No canonical GT-to-source joint/coordinate mapping is exposed; the experiment does not guess a mapping."}
    return {"experiment_id": "EXP038", "status": "COMPLETED", "split": ["train", "val"], "test_used": False, "training_performed": True, "variants": {"L0_LEGAL": {"train_final_loss": loss0, "candidate_metrics": _metrics(flat0, target), "winner": _candidate_diagnostics(val, pred0), "trajectory": trajectories["L0_LEGAL"], "privileged": False}, "L1_GT_LABEL": {"train_final_loss": loss1, "candidate_metrics": _metrics(flat1, target), "winner": _candidate_diagnostics(val, pred1), "trajectory": trajectories["L1_GT_LABEL"], "privileged": True, "gt_activity_label_used_at_inference": True}, "L2_GT_MOTION_STATE": blocked, "L3_GT_LABEL_MOTION": {"status": "BLOCKED_DEPENDS_ON_L2"}, "CLASS_VIEW_PRIOR_ORACLE_LABEL": {"winner": _candidate_diagnostics(val, prior_pred), "trajectory": trajectories["CLASS_VIEW_PRIOR_ORACLE_LABEL"], "privileged": True}}, "observability_gap": {"label_delta_accuracy": None, "motion_delta_accuracy": None, "joint_delta_accuracy": None, "motion_status": blocked["status"]}, "leakage_flags": {"gt_label_variant_privileged": True, "gt_motion_variant_privileged": True, "future_candidate_quality_used_as_input": False, "test_used": False}}


def _head_predictions(model: Any, contexts: Mapping[tuple[str, str, str], Context], stats: tuple[np.ndarray, np.ndarray], binary: bool = False) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    return {key: {viewpoint_id: np.asarray(value) for viewpoint_id, value in values.items()} for key, values in _predict_context(model, contexts, stats, False, heads=True).items()}


def _belief(row: Mapping[str, Any]) -> np.ndarray:
    return belief_from_log_probs(np.asarray(row["s1_feature"], dtype=np.float32)[256:272])


def _exp039(data_root: Path, train: Mapping[tuple[str, str, str], Context], val: Mapping[tuple[str, str, str], Context], stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> tuple[dict[str, Any], Any, Any, tuple[np.ndarray, np.ndarray]]:
    train_x, train_y, train_labels, _ = _samples(train, False); val_x, val_y, val_labels, val_keys = _samples(val, False); train_x, val_x, mean, std = _standardize(train_x, val_x)
    ce_model, ce_loss = _train_heads(train_x, train_y, train_labels); correct_target = np.asarray([float(val[key].field["predicted_class"][viewpoint_id] == int(val[key].row["label_id"])) for key, viewpoint_id in val_keys], dtype=np.float32); correct_model, correct_loss = _train_heads(train_x, np.asarray([float(train[key].field["predicted_class"][viewpoint_id] == int(train[key].row["label_id"])) for key, viewpoint_id in _samples(train, False)[3]], dtype=np.float32), train_labels, binary=True)
    ce = _head_predictions(ce_model, val, (mean, std)); correctness = _head_predictions(correct_model, val, (mean, std), binary=True)
    method_predictions: dict[str, dict[tuple[str, str, str], dict[int, float]]] = {}
    for method in ("MAP_CLASS_RISK", "BELIEF_EXPECTED_RISK", "TOP3_BELIEF_RISK", "BELIEF_EXPECTED_CORRECTNESS", "GT_LABEL_HEAD_UPPER_BOUND"):
        per_context: dict[tuple[str, str, str], dict[int, float]] = {}
        for key, item in val.items():
            row = item.row; belief = _belief(row); belief = top_k_belief(belief, 3) if method == "TOP3_BELIEF_RISK" else belief; label = int(row["label_id"]); map_id = int(np.argmax(belief)); per_context[key] = {}
            for viewpoint_id in _view_ids(row):
                vector = ce[key][viewpoint_id] if method != "BELIEF_EXPECTED_CORRECTNESS" else correctness[key][viewpoint_id]
                if method == "MAP_CLASS_RISK": score = float(vector[map_id])
                elif method == "GT_LABEL_HEAD_UPPER_BOUND": score = float(vector[label])
                else: score = float(np.dot(belief, vector))
                per_context[key][viewpoint_id] = score
        method_predictions[method] = per_context
    trajectories = {}
    for name, pred in method_predictions.items(): trajectories[name] = _trajectory_summary(stage_b_val, v0_val, list(item.row for item in val.values()), categories, val, pred)["learned"]
    target = np.asarray([float(val[key].field["ce"][viewpoint_id]) for key, viewpoint_id in val_keys], dtype=np.float32); flat = np.asarray([float(ce[key][viewpoint_id][int(val[key].row["label_id"])]) for key, viewpoint_id in val_keys])
    result = {"experiment_id": "EXP039", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "model": {"ce_head": {"final_loss": ce_loss, "masked_target": True}, "correctness_head": {"final_loss": correct_loss, "masked_target": True}}, "current_belief_audit": _belief_audit(val), "methods": {name: {"trajectory": trajectories[name], "winner": _candidate_diagnostics(val, pred)} for name, pred in method_predictions.items()}, "gt_label_head_upper_bound": True, "leakage_flags": {"gt_label_used_in_legal_inference": False, "gt_motion_used_in_inference": False, "true_ce_evaluator_only": True, "future_candidate_perception_used_as_input": False, "test_used": False}, "candidate_ce_metrics_for_true_head": _metrics(flat, target)}
    return result, ce_model, correct_model, (mean, std)


def _belief_audit(contexts: Mapping[tuple[str, str, str], Context]) -> dict[str, Any]:
    top1 = top2 = top3 = top5 = 0; entropy: list[float] = []
    for item in contexts.values():
        belief = _belief(item.row); label = int(item.row["label_id"]); order = np.argsort(-belief, kind="mergesort"); top1 += int(order[0] == label); top2 += int(label in order[:2]); top3 += int(label in order[:3]); top5 += int(label in order[:5]); entropy.append(float(-np.sum(belief * np.log(np.maximum(belief, 1e-12)))))
    n = len(contexts)
    return {"count": n, "top1_accuracy": top1 / n, "top2_coverage": top2 / n, "top3_coverage": top3 / n, "top5_coverage": top5 / n, "mean_entropy": float(np.mean(entropy))}


def _view_belief(path: Path, viewpoint_id: int, model: Any, cache: dict[str, dict[int, np.ndarray]], category_count: int) -> np.ndarray:
    """Acquire an exact frozen ST-GCN belief only after a viewpoint is visited."""
    source = str(path)
    if source not in cache:
        logs = _load_archive_predictions(path, model, torch_device(model), 64, category_count)
        cache[source] = {int(view): belief_from_log_probs(logp) for view, logp in logs.items()}
        if len(cache) % 500 == 0:
            print(f"EXP040 frozen ST-GCN observation cache: {len(cache)} source archives", flush=True)
    if int(viewpoint_id) not in cache[source]:
        raise ValueError(f"viewpoint {viewpoint_id} has no finite frozen ST-GCN output in {path}")
    return cache[source][int(viewpoint_id)]


def torch_device(model: Any) -> Any:
    import torch
    return next(model.parameters()).device


_OBS_MODEL: Any = None


def _observation_worker_init(checkpoint: str, category_count: int) -> None:
    global _OBS_MODEL
    import torch
    torch.set_num_threads(1)
    _OBS_MODEL, _ = _load_model(Path(checkpoint), category_count, "cpu")


def _observation_worker(paths: Sequence[str], category_count: int) -> dict[str, dict[int, np.ndarray]]:
    """Batch frozen ST-GCN inference over archives to amortize model calls."""
    import torch
    output: dict[str, dict[int, np.ndarray]] = {}
    device = torch_device(_OBS_MODEL)
    for start in range(0, len(paths), 16):
        group = paths[start:start + 16]; batches: list[np.ndarray] = []; ids: list[np.ndarray] = []
        for path in group:
            with np.load(path, allow_pickle=False) as archive:
                skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
                viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            finite = np.isfinite(skeleton).all(axis=(1, 2, 3)); valid = np.flatnonzero(finite)
            batches.append(skeleton[valid]); ids.append(viewpoint_ids[valid])
        joined = np.concatenate(batches, axis=0)
        with torch.inference_mode():
            logits = torch.log_softmax(_OBS_MODEL(torch.from_numpy(joined).to(device).unsqueeze(-1)), dim=-1).cpu().numpy()
        offset = 0
        for path, view_ids, batch in zip(group, ids, batches):
            count = len(batch); values = logits[offset:offset + count]; offset += count
            output[path] = {int(view): belief_from_log_probs(logp) for view, logp in zip(view_ids.tolist(), values)}
    return output


def _observation_process(paths: Sequence[str], checkpoint: str, category_count: int, output_path: str) -> None:
    """Process entry point; write results to a temporary file to avoid IPC pickling."""
    _observation_worker_init(checkpoint, category_count)
    with open(output_path, "wb") as handle:
        pickle.dump(_observation_worker(paths, category_count), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _exp040(data_root: Path, val: Mapping[tuple[str, str, str], Context], stage_b_val: Sequence[Mapping[str, Any]], v0_val: Sequence[Mapping[str, Any]], categories: Sequence[str], ce_model: Any, correct_model: Any, stats: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text()); mapping = json.loads(Path(summary["label_mapping"]).read_text()); stgcn, device = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), "cpu");
    global _OBS_MODEL
    _OBS_MODEL = stgcn
    neighbors: dict[int, list[int]] = {i: [] for i in range(VIEW_COUNT)}
    for left, right in graph_edges(): neighbors[left].append(right); neighbors[right].append(left)
    static_ce = _predict_all_contexts(ce_model, val, stats, heads=True); static_correctness = _predict_all_contexts(correct_model, val, stats, heads=True)
    method_names = ("Stay", "Random", "BELIEF_GREEDY_LATEST", "BELIEF_GREEDY_MEAN", "BELIEF_GREEDY_GEOMETRIC", "BELIEF_CORRECTNESS_GREEDY", "GT_LABEL_BELIEF_UPPER_BOUND", "TRUE_CE_GRAPH_ORACLE")
    rng = np.random.default_rng(42); observation_cache: dict[str, dict[int, np.ndarray]] = {}; methods: dict[str, dict[str, Any]] = {}
    # Acquire frozen ST-GCN beliefs once per source archive using four CPU workers.
    # The cache is still consulted only after a viewpoint is selected, so no
    # unvisited prediction enters a planner input.
    source_paths = sorted({str(item.source_path) for item in val.values()})
    if source_paths:
        chunks = [source_paths[index::4] for index in range(4)]
        # Spawn four workers and exchange results through temporary files so
        # large belief dictionaries are not serialized through a pipe.
        ctx = mp.get_context("spawn")
        temp_paths = [str(Path(tempfile.mkstemp(prefix="activeview_exp040_", suffix=".pkl")[1])) for _ in range(4)]
        processes = [ctx.Process(target=_observation_process, args=(chunk, str(summary["stgcn_checkpoint"]), len(mapping), temp_path)) for chunk, temp_path in zip(chunks, temp_paths)]
        for process in processes: process.start()
        for process in processes: process.join()
        try:
            for process, temp_path in zip(processes, temp_paths):
                if process.exitcode != 0:
                    raise RuntimeError(f"EXP040 observation worker failed with exit code {process.exitcode}")
                with open(temp_path, "rb") as handle:
                    observation_cache.update(pickle.load(handle))
        finally:
            for temp_path in temp_paths:
                Path(temp_path).unlink(missing_ok=True)
        print(f"EXP040 frozen ST-GCN observation cache: {len(observation_cache)} source archives", flush=True)
    for method in method_names:
        horizons: dict[str, Any] = {}
        for horizon in (1, 2, 3):
            terminal_rows: list[dict[str, Any]] = []
            for key, item in val.items():
                row = item.row; field = item.field; node = int(row["s1_viewpoint_id"]); current_feature = np.asarray(row["s1_feature"], dtype=np.float32); visited_beliefs = [_belief(row)]; visited = [node]; path_length = 0.0; move_count = 0
                for _ in range(horizon):
                    choices = [node] + sorted(neighbors[node]); belief_mode = "latest" if method.endswith("LATEST") else "mean" if method.endswith("MEAN") else "geometric" if method.endswith("GEOMETRIC") else "latest"; belief = fuse_beliefs(visited_beliefs, belief_mode)
                    scores: list[float] = []
                    for candidate in choices:
                        if method == "TRUE_CE_GRAPH_ORACLE": score = float(field["ce"][candidate])
                        else:
                            vector = (static_correctness if method == "BELIEF_CORRECTNESS_GREEDY" else static_ce)[key][candidate]
                            if method == "BELIEF_CORRECTNESS_GREEDY": score = -float(np.dot(belief, vector))
                            elif method == "GT_LABEL_BELIEF_UPPER_BOUND": score = float(vector[int(row["label_id"])])
                            else: score = float(np.dot(belief, vector))
                        scores.append(score)
                    if method == "Random":
                        action = int(rng.integers(len(choices)))
                    elif method == "Stay":
                        action = 0
                    elif method == "BELIEF_CORRECTNESS_GREEDY":
                        action = int(np.argmin(np.asarray(scores, dtype=np.float64)))
                    else:
                        action = int(np.argmin(np.asarray(scores, dtype=np.float64)))
                    selected = int(choices[action])
                    if selected == node: break
                    path_length += float(np.linalg.norm(field["positions"][selected] - field["positions"][node])); move_count += 1; node = selected; visited.append(node)
                    belief_new = observation_cache[str(item.source_path)][node]; visited_beliefs.append(belief_new)
                terminal_belief = fuse_beliefs(visited_beliefs, "latest")
                terminal_rows.append({"label_id": int(row["label_id"]), "predicted_label_id": int(np.argmax(terminal_belief)), "fused_label_id": int(np.argmax(fuse_beliefs(visited_beliefs, "mean"))), "terminal_ce": float(field["ce"][node]), "best_ce": float(np.min(field["ce"][visited])), "path": path_length, "moves": move_count})
            ordered_items = list(val.values()); index_by_episode = {str(item.row["episode_id"]): index for index, item in enumerate(ordered_items)}; v0_by_id = {str(r["episode_id"]): r for r in v0_val}; eligible = set(index_by_episode); truth = []; terminal = []; fused = []
            for record in stage_b_val:
                episode_id = str(record["episode_id"])
                if episode_id in eligible:
                    item_row = terminal_rows[index_by_episode[episode_id]]; truth.append(int(record["label_id"])); terminal.append(int(item_row["predicted_label_id"])); fused.append(int(item_row["fused_label_id"]))
                else:
                    pred = v0_by_id[episode_id]; truth.append(int(record["label_id"])); terminal.append(int(pred["current_predicted_label_id"])); fused.append(int(pred["current_predicted_label_id"]))
            terminal_array = np.asarray(terminal); truth_array = np.asarray(truth); fused_array = np.asarray(fused); terminal_acc = float(np.mean(terminal_array == truth_array)); fused_acc = float(np.mean(fused_array == truth_array)); terminal_f1 = _macro_f1(truth_array, terminal_array); fused_f1 = _macro_f1(truth_array, fused_array); horizons[str(horizon)] = {"terminal_har_accuracy": terminal_acc, "terminal_har_macro_f1": terminal_f1, "fused_har_accuracy": fused_acc, "fused_har_macro_f1": fused_f1, "average_moves": float(np.mean([r["moves"] for r in terminal_rows])), "move_rate": float(np.mean([r["moves"] > 0 for r in terminal_rows])), "path_length": float(np.mean([r["path"] for r in terminal_rows])), "terminal_true_ce": float(np.mean([r["terminal_ce"] for r in terminal_rows])), "best_visited_true_ce": float(np.mean([r["best_ce"] for r in terminal_rows])), "har_episode_count": len(truth)}
        methods[method] = horizons
    return {"experiment_id": "EXP040", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": False, "methods": methods, "state_scoring": "candidate heads use the initial legal Stage-D state; beliefs update only after visited transitions", "belief_update_rules": {"latest": "last visited q", "mean": "arithmetic mean then normalize", "geometric": "mean log posterior, eps=1e-8"}, "leakage_flags": {"unvisited_viewpoint_output_used": False, "visited_viewpoint_output_after_transition_only": True, "gt_label_planner_input": True, "true_ce_planner_input_for_privileged_oracle": True, "true_ce_evaluator_only_for_legal_methods": True, "test_used": False}}


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    classes = sorted(set(truth.tolist()) | set(predicted.tolist())); values = []
    for cls in classes:
        tp = np.sum((truth == cls) & (predicted == cls)); fp = np.sum((truth != cls) & (predicted == cls)); fn = np.sum((truth == cls) & (predicted != cls)); values.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(values)) if values else 0.0


def _write(directory: Path, result: Mapping[str, Any], title: str, extra: Mapping[str, Any] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True); (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); (directory / "analysis.md").write_text(f"# {title}\n\nTrain/Val-only analysis. Test was not read.\n\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n", encoding="utf-8")
    if extra is not None:
        (directory / next(iter(extra))).write_text(json.dumps(next(iter(extra.values())), indent=2, ensure_ascii=False), encoding="utf-8")


def run(data_root: Path, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, Any]:
    started = time.perf_counter(); train_rows = _split_rows(data_root, "train"); val_rows = _split_rows(data_root, "val"); stage_b_val = _utility_rows(data_root, "val");
    if train_limit is not None: train_rows = train_rows[:train_limit]
    if val_limit is not None: val_rows = val_rows[:val_limit]; stage_b_val = [row for row in stage_b_val if str(row["episode_id"]) in {str(item["episode_id"]) for item in val_rows}]
    train_contexts = _load_contexts(data_root, "train", train_rows); val_contexts = _load_contexts(data_root, "val", val_rows); summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text()); mapping = json.loads(Path(summary["label_mapping"]).read_text()); categories = [name for name, _ in sorted(mapping.items(), key=lambda pair: int(pair[1]))]; v0_path = data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"; v0_all = load_jsonl(v0_path); v0_val = v0_all if val_limit is None else [row for row in v0_all if str(row["episode_id"]) in {str(item["episode_id"]) for item in val_rows}]
    print("EXP038: training/evaluation", flush=True)
    exp038 = _exp038(data_root, train_contexts, val_contexts, stage_b_val, v0_val, categories)
    print("EXP038: complete; EXP039: training/evaluation", flush=True)
    exp039, ce_model, correct_model, stats = _exp039(data_root, train_contexts, val_contexts, stage_b_val, v0_val, categories)
    print("EXP039: complete; EXP040: belief-space evaluation", flush=True)
    exp040 = _exp040(data_root, val_contexts, stage_b_val, v0_val, categories, ce_model, correct_model, stats)
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_feature_summary_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/stage_d_feature_summary.json"), "dense_field_root": str((data_root / "datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field").resolve()), "test_used": False}
    for result in (exp038, exp039, exp040): result["provenance"] = provenance
    _write(EXP038, exp038, "EXP038 — Privileged Oracle Observability Ladder", {"observability_gap.json": exp038["observability_gap"]}); _write(EXP039, exp039, "EXP039 — Deployable Belief-Conditioned View Risk", {"belief_audit.json": exp039["current_belief_audit"]}); _write(EXP040, exp040, "EXP040 — Sequential Belief-Space Active HAR", {"belief_update_audit.json": exp040["belief_update_rules"]})
    print(json.dumps({"elapsed_seconds": time.perf_counter() - started, "train_contexts": len(train_contexts), "val_contexts": len(val_contexts), "EXP038": exp038["status"], "EXP039": exp039["status"], "EXP040": exp040["status"]}, ensure_ascii=False)); return {"EXP038": exp038, "EXP039": exp039, "EXP040": exp040}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=Path("../../data/ActiveView")); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); args = parser.parse_args(); run(args.data_root.resolve(), args.train_limit, args.val_limit)


if __name__ == "__main__":
    main()
