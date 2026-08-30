#!/usr/bin/env python3
"""Train and evaluate EXP024 with visited RGB DINOv2 context only."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_contextual_gate import build_contextual_gate_rows
from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import validate_exp016_episode_alignment
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_executed_gate import build_executed_candidate_oracle_gate_trajectories
from activeview.active_view.stage_d_gate_calibration import binary_average_precision, binary_roc_auc, gate_metrics
from activeview.active_view.stage_d_rgb_context import (
    DINO_EMBED_DIM,
    DINO_MODEL_NAME,
    EXP024_INPUT_DIM,
    RGBContextUtilityRegressor,
    RGBObservationKey,
    build_or_load_rgb_cache,
    embedding_index,
    load_dinov2,
    observation_keys_from_feature_rows,
)
from activeview.active_view.stage_d_utility_gate import build_utility_gate_rows
from activeview.scripts.train_stage_d_contextual_gate import _latent_features, _load_ranker
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    invalid = {"<missing>" if row.get("policy_split") is None else str(row["policy_split"]).lower() for row in rows if str(row.get("policy_split", "")).lower() != split}
    if invalid:
        raise ValueError(f"{name} must explicitly contain only {split} rows: {sorted(invalid)}")


def _attach_rgb_keys(examples: list[dict[str, Any]], feature_rows: Sequence[Mapping[str, Any]]) -> tuple[list[RGBObservationKey], set[RGBObservationKey]]:
    _, episode_keys = observation_keys_from_feature_rows(feature_rows)
    requested: set[RGBObservationKey] = set()
    for example in examples:
        keys = episode_keys.get(str(example["episode_id"]))
        if keys is None:
            raise ValueError(f"Missing RGB keys for {example['episode_id']}")
        example["rgb_s0_key"], example["rgb_s1_key"] = keys
        requested.update(keys)
    return list(requested), requested


def _rgb_matrix(
    examples: Sequence[Mapping[str, Any]],
    role: str,
    embeddings: np.ndarray,
    index: Mapping[tuple[str, str, str, int], int],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for example in examples:
        key = example[f"rgb_{role}_key"]
        row_index = index.get(key.tuple)
        if row_index is None:
            raise ValueError(f"Missing cached RGB embedding for {key.tuple}")
        values.append(np.asarray(embeddings[row_index], dtype=np.float32))
    matrix = np.stack(values).astype(np.float32)
    if matrix.shape != (len(examples), DINO_EMBED_DIM) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid RGB feature matrix for {role}")
    return matrix


def _contextual_features(
    ranker: nn.Module,
    examples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latent = _latent_features(ranker, examples, device, BATCH_SIZE)
    if latent.shape[1] != 129:
        raise ValueError("Frozen EXP014 contextual feature extraction must return 129 columns")
    return latent[:, :128].astype(np.float32), latent[:, 128:129].astype(np.float32)


def _train_regressor(
    model: RGBContextUtilityRegressor,
    contextual: np.ndarray,
    predicted: np.ndarray,
    rgb_s0: np.ndarray,
    rgb_s1: np.ndarray,
    targets: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[list[float], float]:
    dataset = TensorDataset(
        torch.from_numpy(contextual), torch.from_numpy(predicted),
        torch.from_numpy(rgb_s0), torch.from_numpy(rgb_s1), torch.from_numpy(targets),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()
    history: list[float] = []
    model.train()
    for _epoch in range(EPOCHS):
        total = 0.0
        count = 0
        for batch_context, batch_predicted, batch_s0, batch_s1, batch_target in loader:
            output = model(batch_context.to(device), batch_predicted.to(device), batch_s0.to(device), batch_s1.to(device))
            loss = criterion(output, batch_target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = int(batch_target.size(0))
            total += float(loss.detach().cpu()) * size
            count += size
        history.append(total / max(count, 1))
    return history, float(history[-1])


def _predict(
    model: RGBContextUtilityRegressor,
    contextual: np.ndarray,
    predicted: np.ndarray,
    rgb_s0: np.ndarray,
    rgb_s1: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(contextual), BATCH_SIZE):
            end = start + BATCH_SIZE
            values = model(
                torch.from_numpy(contextual[start:end]).to(device),
                torch.from_numpy(predicted[start:end]).to(device),
                torch.from_numpy(rgb_s0[start:end]).to(device),
                torch.from_numpy(rgb_s1[start:end]).to(device),
            )
            outputs.append(values.cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)
    if result.shape != (len(contextual),) or not np.isfinite(result).all():
        raise ValueError("EXP024 predictions must be finite")
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _correlations(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    def corr(left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return None
        return float(np.corrcoef(left, right)[0, 1])
    return {"pearson": corr(predicted, target), "spearman": corr(_rank(predicted), _rank(target))}


def _prediction_rows(examples: Sequence[Mapping[str, Any]], predicted: np.ndarray, originals: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, value in zip(examples, predicted):
        move = float(value) > 0.0
        row = dict(originals[str(example["episode_id"])])
        row.update({"predicted_stays": not move, "predicted_candidate_viewpoint_id": int(example["candidate_id"]) if move else None, "predicted_exec_utility": float(value), "utility_gate_target": float(example["true_utility"])})
        rows.append(row)
    return rows


def _metric_row(name: str, summary: Mapping[str, Any], gate: str, candidate: str) -> dict[str, Any]:
    regret = summary["decision_regret"]
    movement = summary["movement"]
    return {
        "variant": name,
        "gate": gate,
        "candidate": candidate,
        "accuracy": float(summary["recognition"]["accuracy"]),
        "macro_f1": float(summary["recognition"]["macro_f1"]),
        "mean_regret": float(regret["mean"]),
        "median_regret": float(regret["median"]),
        "p90_regret": float(regret["p90"]),
        "headroom_capture": float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]),
        "average_moves": float(movement["average_moves"]),
        "mean_geodesic_cost_m": float(movement["trajectory_geodesic_cost_m"]["mean"]),
    }


def _action_change(exp014: Mapping[str, Mapping[str, Any]], exp024: Mapping[str, Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {"stay_to_move": [], "move_to_stay": [], "both_stay": [], "both_move": []}
    mismatches = 0
    for example in examples:
        episode_id = str(example["episode_id"])
        old = exp014[episode_id]
        new = exp024[episode_id]
        old_move = not bool(old["predicted_stays"])
        new_move = not bool(new["predicted_stays"])
        if old_move and new_move:
            key = "both_move"
            mismatches += int(int(old["predicted_candidate_viewpoint_id"]) != int(new["predicted_candidate_viewpoint_id"]))
        elif not old_move and not new_move:
            key = "both_stay"
        elif not old_move:
            key = "stay_to_move"
        else:
            key = "move_to_stay"
        groups[key].append(float(example["true_utility"]))
    result: dict[str, Any] = {"candidate_identity_mismatch_count": mismatches}
    for key, values in groups.items():
        array = np.asarray(values, dtype=np.float64)
        result[key] = {
            "count": int(array.size),
            "positive_count": int(np.sum(array > 0.0)),
            "nonpositive_count": int(np.sum(array <= 0.0)),
            "mean_true_utility": float(array.mean()) if array.size else 0.0,
            "median_true_utility": float(np.median(array)) if array.size else 0.0,
            "sum_true_utility": float(array.sum()) if array.size else 0.0,
        }
    return result


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _load_metric(result_path: Path, variant: str) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for row in payload.get("metrics_table", []):
        if row.get("variant") == variant:
            return dict(row)
    raise ValueError(f"Frozen {variant} metric missing from {result_path}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    metrics = {str(row["variant"]): row for row in result["metrics_table"]}
    regression = result["val_regression_metrics"]
    gate = result["val_sign_gate_metrics_against_y_exec"]
    recovery = result["headroom_recovery"]
    lines = [
        "# EXP024 analysis",
        "",
        "EXP024 uses only already visited Stage-D s0/s1 RGB observations. DINOv2",
        "ViT-B/14 is frozen; only the shared RGB projector and 513-D utility",
        "regression head are trained on Train. Val is evaluated once and Test is locked.",
        "",
        "## RGB audit",
        "",
        f"- Unique Train RGB observations: {result['rgb_audit']['unique_train_rgb_observations']}",
        f"- Unique Val RGB observations: {result['rgb_audit']['unique_val_rgb_observations']}",
        f"- Cache hits / misses: {result['rgb_audit']['cache_hit_count']} / {result['rgb_audit']['cache_miss_count']}",
        f"- Cache extraction time (s): {result['rgb_audit']['extraction_time_sec']:.3f}",
        f"- Cache disk bytes: {result['rgb_audit']['cache_disk_bytes']}",
        "- Future-candidate RGB used: false",
        "",
        "## Train",
        "",
        f"- Episodes: {result['train']['episode_count']}",
        f"- Final SmoothL1 loss: {result['train']['final_loss']:.6f}",
        "",
        "## Val regression/sign diagnostics",
        "",
        f"MAE / RMSE: {regression['mae']:.6f} / {regression['rmse']:.6f}",
        f"Pearson / Spearman: {regression['pearson']} / {regression['spearman']}",
        f"Sign accuracy / balanced accuracy: {gate['accuracy']:.6f} / {gate['balanced_accuracy']:.6f}",
        f"ROC-AUC / PR-AUC: {gate['roc_auc']} / {gate['pr_auc']}",
        "",
        "## Val trajectory metrics",
        "",
        "| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("EXP014", "EXP022", "EXP024", "ExecutedCandidateOracle"):
        row = metrics[name]
        lines.append(f"| {name} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['mean_regret']:.6f} | {row['median_regret']:.6f} | {row['p90_regret']:.6f} | {row['headroom_capture']:.6f} | {row['average_moves']:.6f} | {row['mean_geodesic_cost_m']:.6f} |")
    lines += [
        "",
        "## EXP024 vs EXP022",
        "",
        f"- Accuracy delta: {metrics['EXP024']['accuracy'] - metrics['EXP022']['accuracy']:.6f}",
        f"- Mean-regret delta (EXP024 - EXP022): {metrics['EXP024']['mean_regret'] - metrics['EXP022']['mean_regret']:.6f}",
        f"- Accuracy recovery versus ExecutedCandidateOracle: {recovery['accuracy_recovery']}",
        f"- Mean-regret recovery versus ExecutedCandidateOracle: {recovery['regret_recovery']}",
        "",
        "## Scientific interpretation",
        "",
        "This is a diagnostic RGB-global-embedding result, not a deployment",
        "acceptance decision. Any gain or loss is interpreted relative to EXP022",
        "without changing the frozen candidate ranking or first-step protocol.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    exp014_checkpoint: Path,
    train_predictions: Path,
    val_predictions: Path,
    v0_predictions: Path,
    exp022_result: Path,
    label_mapping: Path,
    rgb_root: Path,
    embedding_cache: Path,
    output: Path,
    runtime_dir: Path,
    seed: int = 42,
    device_name: str = "cuda:0",
    embedding_batch_size: int = 64,
) -> dict[str, Any]:
    """Train on Train and evaluate exactly once on Val; no Test path exists."""
    if not exp014_checkpoint.is_file() or not exp022_result.is_file():
        raise FileNotFoundError("Frozen EXP014 checkpoint and EXP022 result are required")
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"RGB root not found: {rgb_root}")
    _seed_everything(seed)
    device = _device(device_name)
    summary_path = cache_root / "stage_d_feature_summary.json"
    stats_path = cache_root / "stage_d_feature_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(stats_path)
    train_feature_rows = load_jsonl(Path(summary["feature_files"]["train"]))
    val_feature_rows = load_jsonl(Path(summary["feature_files"]["val"]))
    train_pred_rows = load_jsonl(train_predictions)
    val_pred_rows = load_jsonl(val_predictions)
    _assert_split(train_feature_rows, "train", "Stage D Train features")
    _assert_split(val_feature_rows, "val", "Stage D Val features")
    _assert_split(train_pred_rows, "train", "EXP014 Train predictions")
    _assert_split(val_pred_rows, "val", "EXP014 Val predictions")
    train_examples = build_utility_gate_rows(feature_rows=train_feature_rows, prediction_rows=train_pred_rows, current_mean=stats["current_mean"], current_std=stats["current_std"], delta_mean=stats["delta_mean"], delta_std=stats["delta_std"], geometry_mean=stats["geometry_mean"], geometry_std=stats["geometry_std"], split="train")
    val_examples = build_utility_gate_rows(feature_rows=val_feature_rows, prediction_rows=val_pred_rows, current_mean=stats["current_mean"], current_std=stats["current_std"], delta_mean=stats["delta_mean"], delta_std=stats["delta_std"], geometry_mean=stats["geometry_mean"], geometry_std=stats["geometry_std"], split="val")
    train_keys, train_key_set = _attach_rgb_keys(train_examples, train_feature_rows)
    val_keys, val_key_set = _attach_rgb_keys(val_examples, val_feature_rows)
    if train_key_set & val_key_set:
        raise ValueError("Train and Val RGB observation keys overlap")
    all_keys = sorted(set(train_keys + val_keys), key=lambda key: key.tuple)
    cache_started = time.monotonic()
    embeddings, manifest_rows, cache_info = build_or_load_rgb_cache(rgb_root=rgb_root, cache_dir=embedding_cache, keys=all_keys, model_loader=load_dinov2, device=device, batch_size=embedding_batch_size)
    cache_info["cache_call_time_sec"] = time.monotonic() - cache_started
    cache_manifest_path = embedding_cache / "manifest.jsonl"
    cache_summary_path = embedding_cache / "summary.json"
    cache_info["cache_disk_bytes"] = sum(path.stat().st_size for path in embedding_cache.glob("*") if path.is_file())
    cache_info["cache_summary_sha256"] = file_sha256(cache_summary_path)
    cache_map = embedding_index(manifest_rows)

    ranker = _load_ranker(exp014_checkpoint, device)
    train_context, train_predicted = _contextual_features(ranker, train_examples, device)
    val_context, val_predicted = _contextual_features(ranker, val_examples, device)
    train_s0 = _rgb_matrix(train_examples, "s0", embeddings, cache_map)
    train_s1 = _rgb_matrix(train_examples, "s1", embeddings, cache_map)
    val_s0 = _rgb_matrix(val_examples, "s0", embeddings, cache_map)
    val_s1 = _rgb_matrix(val_examples, "s1", embeddings, cache_map)
    train_target = np.asarray([float(row["target_regression"]) for row in train_examples], dtype=np.float32)
    model = RGBContextUtilityRegressor().to(device)
    history, final_loss = _train_regressor(model, train_context, train_predicted, train_s0, train_s1, train_target, seed, device)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "rgb_context_utility_gate_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": "EXP024", "model_state_dict": model.state_dict(), "model_config": {"input_dim": EXP024_INPUT_DIM, "dino_embed_dim": DINO_EMBED_DIM, "rgb_projector_dim": 128, "loss": "SmoothL1Loss"}, "training": {"seed": seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE}, "exp014_frozen": True}, checkpoint)
    training_summary = {"experiment_id": "EXP024", "split": "train", "episode_count": len(train_examples), "target_mean": float(train_target.mean()), "final_loss": final_loss, "loss_history": history, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "loss": "SmoothL1Loss", "selection": "final_epoch_fixed_30", "val_used_for_selection": False, "test_used": False}
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    val_target = np.asarray([float(row["target_regression"]) for row in val_examples], dtype=np.float64)
    val_predicted_utility = _predict(model, val_context, val_predicted, val_s0, val_s1, device)
    corr = _correlations(val_predicted_utility, val_target)
    errors = val_predicted_utility - val_target
    sign_labels = val_target > 0.0
    sign_metrics = gate_metrics(val_predicted_utility, sign_labels.tolist(), 0.0)
    sign_metrics.update({"roc_auc": binary_roc_auc(val_predicted_utility, sign_labels.tolist()), "pr_auc": binary_average_precision(val_predicted_utility, sign_labels.tolist()), "positive_prevalence": float(sign_labels.mean())})
    val_originals = {str(row["episode_id"]): row for row in val_pred_rows}
    exp024_prediction_rows = _prediction_rows(val_examples, val_predicted_utility, val_originals)
    _write_jsonl(runtime_dir / "val_predictions.jsonl", exp024_prediction_rows)
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    alignment = validate_exp016_episode_alignment(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_feature_rows, exp014_prediction_rows=val_pred_rows)
    exp014_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, val_pred_rows)
    exp024_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, exp024_prediction_rows)
    oracle_trajectories, _, _ = build_executed_candidate_oracle_gate_trajectories(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_feature_rows, exp014_prediction_rows=val_pred_rows)
    categories = _categories(label_mapping)
    exp014_metric = _metric_row("EXP014", summarize_trajectory_rows(exp014_trajectories, categories), "predicted utility > 0", "frozen learned c_hat")
    exp022_metric = _load_metric(exp022_result, "EXP022")
    exp024_metric = _metric_row("EXP024", summarize_trajectory_rows(exp024_trajectories, categories), "predicted U_exec > 0", "frozen learned c_hat")
    oracle_metric = _metric_row("ExecutedCandidateOracle", summarize_trajectory_rows(oracle_trajectories, categories), "true U2(c_hat) > 0", "frozen learned c_hat")
    metrics_table = [exp014_metric, exp022_metric, exp024_metric, oracle_metric]
    accuracy_oracle_gap = oracle_metric["accuracy"] - exp014_metric["accuracy"]
    regret_oracle_gap = exp014_metric["mean_regret"] - oracle_metric["mean_regret"]
    result: dict[str, Any] = {
        "experiment_id": "EXP024", "experiment_name": "dinov2_rgb_context_utility_regression_pilot", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "exp014_frozen": True,
        "val_episode_count": len(stage_b_val), "v0_move_eligible_episode_count": len(val_examples), "train": training_summary,
        "model": {"dino_model": DINO_MODEL_NAME, "dino_embedding_dim": DINO_EMBED_DIM, "dino_frozen": True, "input_dim": EXP024_INPUT_DIM, "rgb_projector": "Linear(768,128) -> GELU (shared)", "regression_head": "Linear(513,128) -> GELU -> Linear(128,64) -> GELU -> Linear(64,1)", "loss": "SmoothL1Loss", "target": "raw true_U2(c_hat)"},
        "rgb_audit": {"unique_train_rgb_observations": len(train_key_set), "unique_val_rgb_observations": len(val_key_set), "unique_union_rgb_observations": len(all_keys), **cache_info, "cache_manifest": str(cache_manifest_path.resolve()), "future_candidate_rgb_used": False, "requested_view_roles": ["s0", "s1"]},
        "val_regression_metrics": {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))), **corr},
        "val_sign_gate_metrics_against_y_exec": sign_metrics,
        "metrics_table": metrics_table,
        "headroom_recovery": {"accuracy_gain": exp024_metric["accuracy"] - exp014_metric["accuracy"], "accuracy_oracle_gap": accuracy_oracle_gap, "accuracy_recovery": (exp024_metric["accuracy"] - exp014_metric["accuracy"]) / accuracy_oracle_gap if abs(accuracy_oracle_gap) > 1e-12 else None, "regret_reduction": exp014_metric["mean_regret"] - exp024_metric["mean_regret"], "regret_oracle_gap": regret_oracle_gap, "regret_recovery": (exp014_metric["mean_regret"] - exp024_metric["mean_regret"]) / regret_oracle_gap if abs(regret_oracle_gap) > 1e-12 else None},
        "utility_aware_error_analysis": {"false_move": {"count": int(np.sum((val_predicted_utility > 0.0) & (val_target <= 0.0))), "mean_true_utility": float(val_target[(val_predicted_utility > 0.0) & (val_target <= 0.0)].mean()) if np.any((val_predicted_utility > 0.0) & (val_target <= 0.0)) else 0.0, "total_negative_utility_magnitude": float(-val_target[(val_predicted_utility > 0.0) & (val_target <= 0.0)].sum())}, "false_stay": {"count": int(np.sum((val_predicted_utility <= 0.0) & (val_target > 0.0))), "mean_true_utility": float(val_target[(val_predicted_utility <= 0.0) & (val_target > 0.0)].mean()) if np.any((val_predicted_utility <= 0.0) & (val_target > 0.0)) else 0.0, "total_missed_positive_utility": float(val_target[(val_predicted_utility <= 0.0) & (val_target > 0.0)].sum())}},
        "action_change": _action_change({str(row["episode_id"]): row for row in val_pred_rows}, {str(row["episode_id"]): row for row in exp024_prediction_rows}, val_examples),
        "episode_alignment": alignment,
        "provenance": {"source_commit": _git_commit(), "exp014_checkpoint": str(exp014_checkpoint.resolve()), "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint), "stage_d_feature_summary": str(summary_path.resolve()), "stage_d_feature_summary_sha256": file_sha256(summary_path), "stage_d_feature_stats_sha256": file_sha256(stats_path), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "train_predictions_sha256": file_sha256(train_predictions), "val_predictions_sha256": file_sha256(val_predictions), "v0_val_predictions_sha256": file_sha256(v0_predictions), "stage_b_val_utility_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"), "exp022_result_sha256": file_sha256(exp022_result), "rgb_dataset_summary_sha256": file_sha256(rgb_root / "dataset_summary.json") if (rgb_root / "dataset_summary.json").is_file() else None, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint)},
        "validity": {"candidate_ranking_frozen": True, "first_step_protocol_frozen": True, "exp014_parameters_frozen": True, "dino_parameters_frozen": True, "true_u2_used_only_as_train_target": True, "true_u2_used_as_model_input": False, "future_candidate_rgb_used": False, "val_used_for_selection": False, "test_split_accepted": False, "candidate_identity_mismatch_count": int(result_action_mismatch := _action_change({str(row["episode_id"]): row for row in val_pred_rows}, {str(row["episode_id"]): row for row in exp024_prediction_rows}, val_examples)["candidate_identity_mismatch_count"])},
    }
    if result["validity"]["candidate_identity_mismatch_count"] != 0:
        raise ValueError("EXP024 changed frozen c_hat candidate identity")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    (runtime_dir / "result.json").write_text(payload, encoding="utf-8")
    _write_analysis(output.with_name("analysis.md"), result)
    return result


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = get_data_root()
    rgb_root = Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train")
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--exp014-checkpoint", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/checkpoints/sequential_observation_ranker_best.pth")
    parser.add_argument("--train-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/train_second_step_predictions.jsonl")
    parser.add_argument("--val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/val_second_step_predictions.jsonl")
    parser.add_argument("--v0-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    parser.add_argument("--exp022-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP022_executed_utility_gate/result.json")
    parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--rgb-root", type=Path, default=rgb_root)
    parser.add_argument("--embedding-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_global"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP024_dinov2_rgb_context/result.json")
    parser.add_argument("--runtime-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP024_dinov2_rgb_context"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", dest="device_name", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze(**vars(args))
    print(json.dumps({"experiment_id": "EXP024", "status": "COMPLETED", "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
