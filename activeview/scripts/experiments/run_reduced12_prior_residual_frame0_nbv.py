#!/usr/bin/env python3
"""Audit a Train-derived static viewpoint prior plus causal Frame-0 residual.

The selector uses only current Frame-0 DINO, legal candidate geometry and a
Train-derived viewpoint prior.  Future candidate recognizer outputs are used
only for Train supervision, validation diagnostics and terminal evaluation.
Policy Test is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_adaptive_head_frame0_nbv_audit import (
    DIAGNOSTIC_REL,
    MAX_OPTIONS,
    OLD_HEAD_REL,
    POLICY_REL,
    SELECTOR_REL,
    _device,
    _head_logits,
    _load_head,
    _load_raw_options,
    _log_softmax,
    _option_geometry,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import load_rows
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (
    TASK_TAU,
    VIS_AUX_WEIGHT,
    VIS_TAU,
    TaskUtilityPredictor,
    _load_visibility_targets,
    _predict_task,
    _task_targets,
    _utility_loss,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    EVAL_BATCH,
    _load_frame0_dino,
    _method_metrics,
    _select,
    _terminal,
)

NUM_CLASSES = len(LABELS)
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 1024
WEIGHT_DECAY = 1e-4
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/prior_residual_frame0_nbv"
RUNTIME_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/prior_residual_frame0_nbv")


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for record in sorted(groups):
        values = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(values, OBS_PER_RECORD, replace=len(values) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _viewpoint_prior(margins: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> tuple[dict[int, float], dict[int, int]]:
    sums = np.zeros(32, dtype=np.float64)
    counts = np.zeros(32, dtype=np.int64)
    for row_ids, row_values, row_mask in zip(ids, margins, mask):
        for slot in np.flatnonzero(row_mask):
            viewpoint = int(row_ids[slot])
            sums[viewpoint] += float(row_values[slot])
            counts[viewpoint] += 1
    valid = counts > 0
    overall = float(sums[valid].sum() / counts[valid].sum()) if valid.any() else 0.0
    prior = {viewpoint: float(sums[viewpoint] / counts[viewpoint]) if counts[viewpoint] else overall for viewpoint in range(32)}
    return prior, {viewpoint: int(counts[viewpoint]) for viewpoint in range(32)}


def _prior_scores(ids: np.ndarray, mask: np.ndarray, prior: Mapping[int, float]) -> np.ndarray:
    scores = np.full(ids.shape, -1e9, dtype=np.float32)
    for index in range(ids.shape[0]):
        active = np.flatnonzero(mask[index])
        scores[index, active] = [float(prior[int(viewpoint)]) for viewpoint in ids[index, active]]
    return scores


def _train_branch(
    name: str,
    model: TaskUtilityPredictor,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    train_geometry: np.ndarray,
    val_geometry: np.ndarray,
    train_tokens: np.ndarray | None,
    val_tokens: np.ndarray | None,
    train_target: np.ndarray,
    val_target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
    train_visibility: np.ndarray | None = None,
    val_visibility: np.ndarray | None = None,
) -> TaskUtilityPredictor:
    model = model.to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sampled = _sample_rows(train_rows, rng)
        train_losses: list[float] = []
        for start in range(0, len(sampled), TRAIN_BATCH):
            index = sampled[start : start + TRAIN_BATCH]
            geometry = torch.from_numpy(train_geometry[index]).to(device, non_blocking=True)
            target = torch.from_numpy(train_target[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[index]).to(device, non_blocking=True)
            tokens = None if train_tokens is None else torch.from_numpy(np.asarray(train_tokens[index], dtype=np.float32)).to(device, non_blocking=True)
            predicted, predicted_visibility = model(geometry, tokens)
            total = _utility_loss(predicted, target, mask, TASK_TAU)[0]
            if predicted_visibility is not None and train_visibility is not None:
                visibility = torch.from_numpy(train_visibility[index]).to(device, non_blocking=True)
                total = total + VIS_AUX_WEIGHT * _utility_loss(predicted_visibility, visibility, mask, VIS_TAU)[0]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_rows), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                geometry = torch.from_numpy(val_geometry[sl]).to(device, non_blocking=True)
                target = torch.from_numpy(val_target[sl]).to(device, non_blocking=True)
                mask = torch.from_numpy(val_mask[sl]).to(device, non_blocking=True)
                tokens = None if val_tokens is None else torch.from_numpy(np.asarray(val_tokens[sl], dtype=np.float32)).to(device, non_blocking=True)
                predicted, predicted_visibility = model(geometry, tokens)
                total = _utility_loss(predicted, target, mask, TASK_TAU)[0]
                if predicted_visibility is not None and val_visibility is not None:
                    visibility = torch.from_numpy(val_visibility[sl]).to(device, non_blocking=True)
                    total = total + VIS_AUX_WEIGHT * _utility_loss(predicted_visibility, visibility, mask, VIS_TAU)[0]
                val_losses.append(float(total.cpu()))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_total_loss": float(np.mean(val_losses)),
            "observations": int(len(sampled)),
            "records": int(len({str(row["record_id"]) for row in train_rows})),
        }
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train={record['train_loss']:.6f} val={record['val_total_loss']:.6f}", flush=True)
        if record["val_total_loss"] < best_loss:
            best_loss = float(record["val_total_loss"])
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "branch": name, "epoch": epoch, "seed": SEED}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    summary = {
        "branch": name,
        "seed": SEED,
        "epochs": EPOCHS,
        "observations_per_record": OBS_PER_RECORD,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_loss,
        "checkpoint_selection": "minimum Moving Val residual objective with fixed visibility auxiliary when enabled",
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval()


def _subset_metric(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    result = _method_metrics(name, rows, labels, predictions, actions)
    result["n"] = int(len(rows))
    return result


def _ranking_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    candidate_mask = np.asarray(mask, dtype=bool).copy()
    candidate_mask[:, 0] = False
    valid_pred, valid_target = predicted[candidate_mask], target[candidate_mask]
    within: list[float] = []
    for index in range(predicted.shape[0]):
        active = np.flatnonzero(candidate_mask[index])
        if active.size >= 2:
            within.append(correlation(predicted[index, active], target[index, active], spearman=True))
    return {
        "candidate_count": int(valid_pred.size),
        "candidate_level_spearman": correlation(valid_pred, valid_target, spearman=True),
        "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
        "within_context_spearman_median": float(np.median(within)) if within else 0.0,
        "within_context_count": int(len(within)),
        "mae": float(np.mean(np.abs(valid_pred - valid_target))) if valid_pred.size else 0.0,
        "mean_abs_predicted_residual": float(np.mean(np.abs(valid_pred))) if valid_pred.size else 0.0,
        "std_predicted_residual": float(np.std(valid_pred)) if valid_pred.size else 0.0,
        "mean_abs_true_residual": float(np.mean(np.abs(valid_target))) if valid_target.size else 0.0,
        "std_true_residual": float(np.std(valid_target)) if valid_target.size else 0.0,
    }


def _switch_analysis(
    prior_actions: Sequence[int], residual_actions: Sequence[int], rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray, prior_predictions: np.ndarray, residual_predictions: np.ndarray,
) -> dict[str, Any]:
    switched = np.asarray(prior_actions, dtype=np.int64) != np.asarray(residual_actions, dtype=np.int64)
    prior_correct = prior_predictions[switched] == labels[switched]
    residual_correct = residual_predictions[switched] == labels[switched]
    indices = np.flatnonzero(switched)
    subset_rows = [rows[int(index)] for index in indices]
    return {
        "count": int(switched.sum()),
        "rate": float(switched.mean()) if switched.size else 0.0,
        "prior_selected_on_switched": _subset_metric("prior", subset_rows, labels[switched], prior_predictions[switched], np.asarray(prior_actions)[switched].tolist()) if switched.any() else {},
        "residual_selected_on_switched": _subset_metric("residual", subset_rows, labels[switched], residual_predictions[switched], np.asarray(residual_actions)[switched].tolist()) if switched.any() else {},
        "prior_wrong_residual_correct": int(np.sum(~prior_correct & residual_correct)),
        "prior_correct_residual_wrong": int(np.sum(prior_correct & ~residual_correct)),
        "both_correct": int(np.sum(prior_correct & residual_correct)),
        "both_wrong": int(np.sum(~prior_correct & ~residual_correct)),
    }


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    static = methods["StaticViewPrior + old adaptive"]
    instance = methods["Adaptive-aware selector + old adaptive"]
    residual_names = ["Prior+GeometryResidual", "Prior+RGBResidual λ=0.5", "Prior+RGBResidual λ=1.0"]
    best_name = max(residual_names, key=lambda name: float(methods[name]["accuracy"]))
    best = methods[best_name]
    static_gain = 100.0 * (best["accuracy"] - static["accuracy"])
    instance_gain = 100.0 * (best["accuracy"] - instance["accuracy"])
    rgb_drop = result["shuffle_metrics"]["accuracy_drop_pp"]
    if static_gain >= 1.5 and best["accuracy"] >= 0.54 and rgb_drop >= 1.0:
        decision = "STRONG KEEP"
    elif static_gain >= 1.0 and best["accuracy"] >= 0.535 and best["macro_f1"] >= static["macro_f1"]:
        decision = "KEEP"
    elif static_gain >= 0.5 and instance_gain < 0.5:
        decision = "WEAK KEEP"
    else:
        decision = "KILL PRIOR+RESIDUAL FRAME0 NBV"
    switched = result["switch_analysis"]["rgb_lambda_1"]
    lines = [
        "# Static View Prior + Frame0 Residual NBV Audit", "",
        "Protocol: current Frame0 RGB → exactly one action from current/Stay + Stage-A legal candidate_pool → selected real O1 alone → frozen ST-GCN + old adaptive head.",
        "Training split: Policy Train only; evaluation/model selection split: Moving Val; Policy Test used: false.",
        "The static prior is Q(v)=mean Train GT-Margin under the old adaptive recognizer. The residual target is R(x,v)=U(x,v)-Q(v).",
        "",
        "## Main Moving-Val results", "",
        "| Method | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|",
    ]
    order = ["Random + old adaptive", "Historical selector + old adaptive", "StaticViewPrior + old adaptive", "Adaptive-aware selector + old adaptive", *residual_names, "GT-Margin Oracle", "GT-TrueLogP Oracle"]
    for name in order:
        metric = methods.get(name)
        if metric:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend([
        "", "## Requested comparisons", "",
        f"Best residual branch: **{best_name}**; gain over StaticViewPrior = {static_gain:+.3f}pp Accuracy / {100.0 * (best['macro_f1'] - static['macro_f1']):+.3f}pp Macro-F1; gain over InstanceOnly = {instance_gain:+.3f}pp Accuracy / {100.0 * (best['macro_f1'] - instance['macro_f1']):+.3f}pp Macro-F1.",
        f"Prior vs residual (RGB λ=1): agreement={switched['agreement']:.6f}, switches={switched['switch_count']} ({switched['switch_rate']:.6f}); switched prior/residual Accuracy={switched['prior_switched_accuracy']:.6f}/{switched['residual_switched_accuracy']:.6f}.",
        f"On switched contexts, residual gain is {switched['residual_switched_accuracy'] - switched['prior_switched_accuracy']:+.6f} Accuracy ({100.0 * (switched['residual_switched_accuracy'] - switched['prior_switched_accuracy']):+.3f}pp).",
        f"Switched transitions: prior-wrong/residual-correct={switched['analysis']['prior_wrong_residual_correct']}, prior-correct/residual-wrong={switched['analysis']['prior_correct_residual_wrong']}.",
        f"RGB residual ranking: candidate Spearman={result['residual_ranking_metrics']['rgb_lambda_1']['candidate_level_spearman']:.6f}; within-context mean/median={result['residual_ranking_metrics']['rgb_lambda_1']['within_context_spearman_mean']:.6f}/{result['residual_ranking_metrics']['rgb_lambda_1']['within_context_spearman_median']:.6f}.",
        f"RGB shuffle: normal={result['shuffle_metrics']['normal']['accuracy']:.6f}, shuffled={result['shuffle_metrics']['rgb_shuffled']['accuracy']:.6f}, drop={rgb_drop:+.3f}pp.",
        f"Residual-zero reproduces prior exactly: actions_equal={result['residual_zero']['actions_equal']}, Accuracy={result['residual_zero']['metrics']['accuracy']:.6f}.",
        "",
        "## High-occlusion subset", "",
        f"Definition: bottom tertile of current Frame0 Stay SceneVisibility ({result['occlusion_metrics']['count']} contexts).",
        "| Selector | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ])
    for name, metric in result["occlusion_metrics"]["methods"].items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend([
        "", "## Oracle and interpretation", "",
        f"GT-Margin Oracle Accuracy/F1={methods['GT-Margin Oracle']['accuracy']:.6f}/{methods['GT-Margin Oracle']['macro_f1']:.6f}; AnyCorrect coverage on the full legal action set={result['oracle_metrics']['any_correct_action_set']['rate']:.6f} (candidate-only={result['oracle_metrics']['any_correct_candidates']['rate']:.6f}).",
        f"Frame0 RGB provides instance-specific NBV beyond the prior only if residual selection changes actions and improves selected real-O1 recognition. The observed λ=1 switch rate is {switched['switch_rate']:.6f}, with net rescue={switched['analysis']['prior_wrong_residual_correct'] - switched['analysis']['prior_correct_residual_wrong']}.",
        f"Decision: **{decision}**. This is a strict one-step diagnostic; no O0+O1 fusion, B3/B4, continuous navigation or Test evaluation was used.",
        f"Largest per-class F1 gains of the best residual branch over StaticViewPrior: {', '.join(f'{name} {gain:+.3f}pp' for name, gain in result['benefiting_classes'])}.",
        "",
        "```text",
        "policy_test_used=false",
        "training_split=Policy Train",
        "evaluation_split=Moving Val",
        "future_candidate_observation_used_for_selector=false",
        "future_candidate_recognizer_output_used_for_selector=false",
        "gt_action_used_for_selector=false",
        "terminal=selected real O1 alone",
        "```",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    started = time.monotonic()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    train_cache = _load_raw_options(data_root, train_rows, "train")
    val_cache = _load_raw_options(data_root, val_rows, "val")
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    old_path = data_root / OLD_HEAD_REL
    selector_path = data_root / SELECTOR_REL
    for path in (old_path, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_head = _load_head(old_path, device)
    train_logp = _log_softmax(_head_logits(train_cache, old_head, device))
    val_logp = _log_softmax(_head_logits(val_cache, old_head, device))
    train_true, train_margin = _task_targets(train_logp, labels_train, train_cache["mask"])
    val_true, val_margin = _task_targets(val_logp, labels_val, val_cache["mask"])
    prior, prior_counts = _viewpoint_prior(train_margin, train_cache["ids"], train_cache["mask"])
    train_q = _prior_scores(train_cache["ids"], train_cache["mask"], prior)
    val_q = _prior_scores(val_cache["ids"], val_cache["mask"], prior)
    train_residual = train_margin - train_q
    val_residual = val_margin - val_q
    stats = json.loads((data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geometry, _, train_geom_mask = _option_geometry(train_rows, stats)
    val_geometry, _, val_geom_mask = _option_geometry(val_rows, stats)
    if not np.array_equal(train_geom_mask, train_cache["mask"]) or not np.array_equal(val_geom_mask, val_cache["mask"]):
        raise ValueError("geometry/action mask mismatch")
    train_tokens, train_dino_meta = _load_frame0_dino(data_root, train_rows, "train", device)
    val_tokens, val_dino_meta = _load_frame0_dino(data_root, val_rows, "val", device)
    train_visibility = _load_visibility_targets(data_root, train_rows, "train")
    val_visibility = _load_visibility_targets(data_root, val_rows, "val")
    runtime = data_root / RUNTIME_REL
    output_root.mkdir(parents=True, exist_ok=True)
    rgb_model = _train_branch(
        "RGBResidual", TaskUtilityPredictor(True, True), train_rows, val_rows, train_geometry, val_geometry,
        train_tokens, val_tokens, train_residual, val_residual, train_cache["mask"], val_cache["mask"], device,
        runtime / "rgb_residual.pth", output_root / "rgb_residual_training.json",
        train_visibility["scores"].astype(np.float32), val_visibility["scores"].astype(np.float32),
    )
    geometry_model = _train_branch(
        "GeometryResidual", TaskUtilityPredictor(False, False), train_rows, val_rows, train_geometry, val_geometry,
        None, None, train_residual, val_residual, train_cache["mask"], val_cache["mask"], device,
        runtime / "geometry_residual.pth", output_root / "geometry_residual_training.json",
    )
    rgb_residual, _ = _predict_task(rgb_model, val_geometry, val_tokens, device)
    geometry_residual, _ = _predict_task(geometry_model, val_geometry, None, device)
    selector_path = data_root / SELECTOR_REL
    historical = TaskUtilityPredictor(True, True).to(device)
    historical.load_state_dict(torch.load(selector_path, map_location=device, weights_only=False)["state_dict"])
    adaptive_path = data_root / RUNTIME_REL.parent / "adaptive_head_frame0_nbv_audit/selector_adapt_old.pth"
    adaptive = TaskUtilityPredictor(True, True).to(device)
    adaptive.load_state_dict(torch.load(adaptive_path, map_location=device, weights_only=False)["state_dict"])
    historical_scores, _ = _predict_task(historical.eval(), val_geometry, val_tokens, device)
    adaptive_scores, _ = _predict_task(adaptive.eval(), val_geometry, val_tokens, device)
    score_sets: dict[str, np.ndarray] = {
        "StaticViewPrior + old adaptive": val_q,
        "Historical selector + old adaptive": historical_scores,
        "Adaptive-aware selector + old adaptive": adaptive_scores,
        "Prior+GeometryResidual": val_q + geometry_residual,
        "Prior+RGBResidual λ=0.5": val_q + 0.5 * rgb_residual,
        "Prior+RGBResidual λ=1.0": val_q + rgb_residual,
    }
    rng = np.random.default_rng(SEED)
    random_scores = np.zeros_like(val_q)
    for index, row in enumerate(val_rows):
        action_count = 1 + len(row["candidate_ids"])
        random_scores[index, :action_count] = rng.random(action_count)
    score_sets["Random + old adaptive"] = random_scores
    score_sets["GT-Margin Oracle"] = val_margin
    score_sets["GT-TrueLogP Oracle"] = val_true
    predictions: dict[str, np.ndarray] = {}
    actions: dict[str, list[int]] = {}
    methods: dict[str, Any] = {}
    for name, scores in score_sets.items():
        selected = _select(val_rows, scores, val_cache["mask"])
        actions[name] = selected
        predictions[name] = _terminal(val_logp, val_cache["ids"], val_cache["mask"], selected, labels_val)
        methods[name] = _method_metrics(name, val_rows, labels_val, predictions[name], selected)
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    actions["Stay"] = stay_actions
    predictions["Stay"] = _terminal(val_logp, val_cache["ids"], val_cache["mask"], stay_actions, labels_val)
    methods["Stay"] = _method_metrics("Stay", val_rows, labels_val, predictions["Stay"], stay_actions)
    best_residual = max((name for name in ("Prior+GeometryResidual", "Prior+RGBResidual λ=0.5", "Prior+RGBResidual λ=1.0")), key=lambda name: methods[name]["accuracy"])
    prior_actions = actions["StaticViewPrior + old adaptive"]
    residual_actions = actions[best_residual]
    switch_by_lambda: dict[str, Any] = {}
    for label, method_name in (("geometry", "Prior+GeometryResidual"), ("rgb_lambda_05", "Prior+RGBResidual λ=0.5"), ("rgb_lambda_1", "Prior+RGBResidual λ=1.0")):
        switch = np.asarray(prior_actions) != np.asarray(actions[method_name])
        analysis = _switch_analysis(prior_actions, actions[method_name], val_rows, labels_val, predictions["StaticViewPrior + old adaptive"], predictions[method_name])
        switch_by_lambda[label] = {
            "agreement": float(1.0 - switch.mean()),
            "switch_count": int(switch.sum()),
            "switch_rate": float(switch.mean()),
            "prior_switched_accuracy": analysis["prior_selected_on_switched"].get("accuracy", 0.0),
            "residual_switched_accuracy": analysis["residual_selected_on_switched"].get("accuracy", 0.0),
            "analysis": analysis,
        }
    permutation = np.random.default_rng(SEED).permutation(len(val_rows))
    shuffled_residual, _ = _predict_task(rgb_model, val_geometry, val_tokens[permutation], device)
    shuffled_scores = val_q + shuffled_residual
    shuffled_actions = _select(val_rows, shuffled_scores, val_cache["mask"])
    shuffled_predictions = _terminal(val_logp, val_cache["ids"], val_cache["mask"], shuffled_actions, labels_val)
    shuffle_metrics = {
        "normal": methods["Prior+RGBResidual λ=1.0"],
        "rgb_shuffled": _method_metrics("Prior+RGBResidual λ=1.0 RGB-shuffled", val_rows, labels_val, shuffled_predictions, shuffled_actions),
        "accuracy_drop_pp": 100.0 * (methods["Prior+RGBResidual λ=1.0"]["accuracy"] - float(np.mean(shuffled_predictions == labels_val))),
        "permutation_seed": SEED,
    }
    zero_actions = _select(val_rows, val_q, val_cache["mask"])
    zero_predictions = _terminal(val_logp, val_cache["ids"], val_cache["mask"], zero_actions, labels_val)
    residual_zero = {"actions_equal": bool(zero_actions == prior_actions), "metrics": _method_metrics("Residual-zero", val_rows, labels_val, zero_predictions, zero_actions)}
    visibility = np.asarray(val_visibility["scores"][:, 0], dtype=np.float64)
    high_mask = visibility < np.quantile(visibility, 1.0 / 3.0)
    high_indices = np.flatnonzero(high_mask)
    high_names = ["StaticViewPrior + old adaptive", "Adaptive-aware selector + old adaptive", "Prior+GeometryResidual", "Prior+RGBResidual λ=1.0", "GT-Margin Oracle"]
    high_methods: dict[str, Any] = {}
    for name in high_names:
        high_rows = [val_rows[int(index)] for index in high_indices]
        high_methods[name] = _subset_metric(name, high_rows, labels_val[high_mask], predictions[name][high_mask], [actions[name][int(index)] for index in high_indices])
    candidate_any = np.any((np.argmax(val_logp, axis=-1) == labels_val[:, None])[:, 1:] & val_cache["mask"][:, 1:], axis=1)
    action_any = np.any((np.argmax(val_logp, axis=-1) == labels_val[:, None]) & val_cache["mask"], axis=1)
    oracle_metrics = {
        "any_correct_action_set": {"count": int(action_any.sum()), "rate": float(action_any.mean())},
        "any_correct_candidates": {"count": int(candidate_any.sum()), "rate": float(candidate_any.mean())},
        "gt_margin_oracle": methods["GT-Margin Oracle"],
        "gt_true_logp_oracle": methods["GT-TrueLogP Oracle"],
    }
    per_class = {name: methods[name]["per_class"] for name in methods}
    residual_ranking = {
        "geometry": _ranking_metrics(geometry_residual, val_residual, val_cache["mask"]),
        "rgb_lambda_05": _ranking_metrics(rgb_residual, val_residual, val_cache["mask"]),
        "rgb_lambda_1": _ranking_metrics(rgb_residual, val_residual, val_cache["mask"]),
    }
    class_gains = []
    for label_name in LABELS:
        static_f1 = float(methods["StaticViewPrior + old adaptive"]["per_class"][label_name]["f1"])
        residual_f1 = float(methods[best_residual]["per_class"][label_name]["f1"])
        class_gains.append((label_name, 100.0 * (residual_f1 - static_f1)))
    class_gains.sort(key=lambda item: (-item[1], item[0]))
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_PRIOR_RESIDUAL_FRAME0_NBV_AUDIT",
        "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(row['record_id']) for row in train_rows}), "val_records": len({str(row['record_id']) for row in val_rows})},
        "labels": list(LABELS),
        "methods": methods,
        "best_prior_residual": best_residual,
        "benefiting_classes": [[name, float(gain)] for name, gain in class_gains[:4]],
        "static_view_prior": {"definition": "Q(v)=mean Train GT-Margin under frozen old adaptive head", "values": {str(key): value for key, value in prior.items()}, "counts": {str(key): value for key, value in prior_counts.items()}, "smoothing": "global mean only for an unseen viewpoint", "train_only": True},
        "residual_ranking_metrics": residual_ranking,
        "switch_analysis": switch_by_lambda,
        "shuffle_metrics": shuffle_metrics,
        "residual_zero": residual_zero,
        "occlusion_metrics": {"definition": "bottom tertile of current Frame0 Stay SceneVisibility", "count": int(high_mask.sum()), "methods": high_methods},
        "oracle_metrics": oracle_metrics,
        "per_class_metrics": per_class,
        "protocol": {"action_set": "current/Stay + Stage-A legal candidate_pool", "terminal": "selected real O1 alone through frozen ST-GCN + old adaptive head", "selector_input": "current Frame0 DINO tokens + candidate geometry + Train-derived Q(v)", "residual_target": "U(x,v)-Q(v), U=old-adaptive GT-Margin", "fixed_lambdas": [0.5, 1.0], "test_used": False},
        "leakage_audit": {"policy_test_used": False, "future_candidate_observation_used_for_selector": False, "future_candidate_recognizer_output_used_for_selector": False, "gt_action_used_for_selector": False, "future_candidate_recognizer_output_used_for_train_supervision_only": True, "terminal_selected_real_o1_alone": True},
        "artifacts": {"old_adaptive_head": str(old_path.resolve()), "historical_selector": str(selector_path.resolve()), "train_options": str((data_root / DIAGNOSTIC_REL / "train_options.npz").resolve()), "val_options": str((data_root / DIAGNOSTIC_REL / "val_options.npz").resolve()), "frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta},
        "checkpoint_sha256": {"old_adaptive_head": _sha256(old_path), "historical_selector": _sha256(selector_path)},
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "epochs": EPOCHS, "elapsed_seconds": time.monotonic() - started},
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "fixed_lambdas": [0.5, 1.0], "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "static_view_prior.json").write_text(json.dumps(result["static_view_prior"], indent=2) + "\n", encoding="utf-8")
    (output_root / "residual_training_metrics.json").write_text(json.dumps({"RGBResidual": json.loads((output_root / "rgb_residual_training.json").read_text()), "GeometryResidual": json.loads((output_root / "geometry_residual_training.json").read_text())}, indent=2) + "\n", encoding="utf-8")
    (output_root / "selector_metrics.json").write_text(json.dumps(methods, indent=2) + "\n", encoding="utf-8")
    (output_root / "residual_ranking_metrics.json").write_text(json.dumps(residual_ranking, indent=2) + "\n", encoding="utf-8")
    (output_root / "switch_analysis.json").write_text(json.dumps(switch_by_lambda, indent=2) + "\n", encoding="utf-8")
    (output_root / "shuffle_metrics.json").write_text(json.dumps(shuffle_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "occlusion_metrics.json").write_text(json.dumps(result["occlusion_metrics"], indent=2) + "\n", encoding="utf-8")
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2) + "\n", encoding="utf-8")
    (output_root / "oracle_metrics.json").write_text(json.dumps(oracle_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(result["leakage_audit"], indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()), "best_prior_residual": result["best_prior_residual"], "accuracy": result["methods"][result["best_prior_residual"]]["accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
