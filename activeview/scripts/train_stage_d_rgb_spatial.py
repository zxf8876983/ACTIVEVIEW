#!/usr/bin/env python3
"""Train/evaluate EXP025 with frozen DINOv2 spatial RGB context."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import validate_exp016_episode_alignment
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_executed_gate import build_executed_candidate_oracle_gate_trajectories
from activeview.active_view.stage_d_gate_calibration import binary_average_precision, binary_roc_auc, gate_metrics
from activeview.active_view.stage_d_rgb_context import DINO_EMBED_DIM, DINO_MODEL_NAME, RGBObservationKey, load_dinov2, observation_keys_from_feature_rows
from activeview.active_view.stage_d_rgb_spatial import SPATIAL_TOKEN_COUNT, SpatialRGBUtilityRegressor, build_or_load_spatial_cache, spatial_embedding_index
from activeview.active_view.stage_d_utility_gate import build_utility_gate_rows
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root
from activeview.scripts.train_stage_d_contextual_gate import _latent_features, _load_ranker
from activeview.scripts.train_stage_d_rgb_context import _action_change, _categories, _load_metric, _metric_row


EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def _seed(seed: int) -> None:
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


def _attach_keys(examples: list[dict[str, Any]], feature_rows: Sequence[Mapping[str, Any]]) -> tuple[set[RGBObservationKey], dict[str, tuple[RGBObservationKey, RGBObservationKey]]]:
    _, episode_keys = observation_keys_from_feature_rows(feature_rows)
    requested: set[RGBObservationKey] = set()
    for example in examples:
        pair = episode_keys.get(str(example["episode_id"]))
        if pair is None:
            raise ValueError(f"Missing RGB key for {example['episode_id']}")
        example["rgb_s0_key"], example["rgb_s1_key"] = pair
        requested.update(pair)
    return requested, episode_keys


def _spatial_matrix(examples: Sequence[Mapping[str, Any]], role: str, embeddings: np.ndarray, index: Mapping[tuple[str, str, str, int], int]) -> np.ndarray:
    values: list[np.ndarray] = []
    for example in examples:
        key = example[f"rgb_{role}_key"]
        row_index = index.get(key.tuple)
        if row_index is None:
            raise ValueError(f"Missing spatial RGB embedding for {key.tuple}")
        values.append(np.asarray(embeddings[row_index], dtype=np.float32))
    result = np.stack(values).astype(np.float32)
    expected = (len(examples), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"Invalid spatial RGB matrix: {result.shape}")
    return result


def _train(model: SpatialRGBUtilityRegressor, context: np.ndarray, predicted: np.ndarray, s0: np.ndarray, s1: np.ndarray, targets: np.ndarray, seed: int, device: torch.device) -> tuple[list[float], float]:
    dataset = TensorDataset(torch.from_numpy(context), torch.from_numpy(predicted), torch.from_numpy(s0), torch.from_numpy(s1), torch.from_numpy(targets))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()
    history: list[float] = []
    model.train()
    for _epoch in range(EPOCHS):
        total, count = 0.0, 0
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


def _predict(model: SpatialRGBUtilityRegressor, context: np.ndarray, predicted: np.ndarray, s0: np.ndarray, s1: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(context), BATCH_SIZE):
            end = start + BATCH_SIZE
            values.append(model(torch.from_numpy(context[start:end]).to(device), torch.from_numpy(predicted[start:end]).to(device), torch.from_numpy(s0[start:end]).to(device), torch.from_numpy(s1[start:end]).to(device)).cpu().numpy())
    result = np.concatenate(values).astype(np.float64)
    if result.shape != (len(context),) or not np.isfinite(result).all():
        raise ValueError("EXP025 predictions must be finite")
    return result


def _prediction_rows(examples: Sequence[Mapping[str, Any]], values: np.ndarray, originals: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, value in zip(examples, values):
        move = float(value) > 0.0
        row = dict(originals[str(example["episode_id"])])
        row.update({"predicted_stays": not move, "predicted_candidate_viewpoint_id": int(example["candidate_id"]) if move else None, "predicted_exec_utility": float(value), "utility_gate_target": float(example["true_utility"])})
        rows.append(row)
    return rows


def _correlation(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    def corr(left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return None
        return float(np.corrcoef(left, right)[0, 1])
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end + 1) / 2.0
            start = end
        return result
    return {"pearson": corr(predicted, target), "spearman": corr(ranks(predicted), ranks(target))}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    metrics = {str(row["variant"]): row for row in result["metrics_table"]}
    regression = result["val_regression_metrics"]
    gate = result["val_sign_gate_metrics_against_y_exec"]
    lines = [
        "# EXP025 analysis", "",
        "EXP025 uses frozen DINOv2 ViT-B/14 patch tokens from already visited s0/s1 RGB only. Patch tokens are pooled from 16x16 to 4x4, encoded by a shared 1-layer spatial Transformer, and combined with the frozen EXP014 contextual token and predicted utility. Test remains locked.", "",
        "## Cache and Train", "",
        f"- Unique Train / Val observations: {result['rgb_audit']['unique_train_rgb_observations']} / {result['rgb_audit']['unique_val_rgb_observations']}",
        f"- Cache bytes / extraction seconds: {result['rgb_audit']['cache_disk_bytes']} / {result['rgb_audit']['extraction_time_sec']:.3f}",
        f"- Final Train SmoothL1 loss: {result['train']['final_loss']:.6f}",
        "- Future-candidate RGB used: false", "",
        "## Val regression/sign diagnostics", "",
        f"- MAE / RMSE: {regression['mae']:.6f} / {regression['rmse']:.6f}",
        f"- Pearson / Spearman: {regression['pearson']} / {regression['spearman']}",
        f"- Sign accuracy / balanced accuracy: {gate['accuracy']:.6f} / {gate['balanced_accuracy']:.6f}",
        f"- ROC-AUC / PR-AUC: {gate['roc_auc']} / {gate['pr_auc']}", "",
        "## Val trajectory metrics", "",
        "| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("EXP014", "EXP022", "EXP024", "EXP025", "ExecutedCandidateOracle"):
        row = metrics[name]
        lines.append(f"| {name} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['mean_regret']:.6f} | {row['median_regret']:.6f} | {row['p90_regret']:.6f} | {row['headroom_capture']:.6f} |")
    row025, row024 = metrics["EXP025"], metrics["EXP024"]
    lines += ["", "## EXP025 vs EXP024", "", f"- Accuracy delta: {row025['accuracy'] - row024['accuracy']:.6f}", f"- Macro-F1 delta: {row025['macro_f1'] - row024['macro_f1']:.6f}", f"- Mean-regret delta (EXP025 - EXP024): {row025['mean_regret'] - row024['mean_regret']:.6f}", f"- P90-regret delta: {row025['p90_regret'] - row024['p90_regret']:.6f}", "", "## Interpretation", "", "This is an analysis-only spatial RGB representation pilot; no deployment acceptance is implied."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(*, cache_root: Path, stage_b_root: Path, exp014_checkpoint: Path, train_predictions: Path, val_predictions: Path, v0_predictions: Path, exp022_result: Path, exp024_result: Path, label_mapping: Path, rgb_root: Path, embedding_cache: Path, output: Path, runtime_dir: Path, seed: int = 42, device_name: str = "cuda:0", embedding_batch_size: int = 32) -> dict[str, Any]:
    if not exp014_checkpoint.is_file() or not exp022_result.is_file() or not exp024_result.is_file():
        raise FileNotFoundError("Frozen EXP014 checkpoint and EXP022/EXP024 results are required")
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"RGB root not found: {rgb_root}")
    _seed(seed)
    device = _device(device_name)
    summary_path = cache_root / "stage_d_feature_summary.json"
    stats_path = cache_root / "stage_d_feature_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(stats_path)
    train_feature_rows = load_jsonl(Path(summary["feature_files"]["train"]))
    val_feature_rows = load_jsonl(Path(summary["feature_files"]["val"]))
    train_pred_rows = load_jsonl(train_predictions)
    val_pred_rows = load_jsonl(val_predictions)
    _assert_split(train_feature_rows, "train", "Stage-D Train features")
    _assert_split(val_feature_rows, "val", "Stage-D Val features")
    _assert_split(train_pred_rows, "train", "EXP014 Train predictions")
    _assert_split(val_pred_rows, "val", "EXP014 Val predictions")
    kwargs = dict(current_mean=stats["current_mean"], current_std=stats["current_std"], delta_mean=stats["delta_mean"], delta_std=stats["delta_std"], geometry_mean=stats["geometry_mean"], geometry_std=stats["geometry_std"])
    train_examples = build_utility_gate_rows(feature_rows=train_feature_rows, prediction_rows=train_pred_rows, split="train", **kwargs)
    val_examples = build_utility_gate_rows(feature_rows=val_feature_rows, prediction_rows=val_pred_rows, split="val", **kwargs)
    train_keys, _ = _attach_keys(train_examples, train_feature_rows)
    val_keys, _ = _attach_keys(val_examples, val_feature_rows)
    if train_keys & val_keys:
        raise ValueError("Train and Val RGB observation keys overlap")
    all_keys = sorted(train_keys | val_keys, key=lambda key: key.tuple)
    embeddings, manifest_rows, cache_info = build_or_load_spatial_cache(rgb_root=rgb_root, cache_dir=embedding_cache, keys=all_keys, model_loader=load_dinov2, device=device, batch_size=embedding_batch_size)
    cache_info["cache_disk_bytes"] = sum(path.stat().st_size for path in embedding_cache.glob("*") if path.is_file())
    cache_info["cache_summary_sha256"] = file_sha256(embedding_cache / "summary.json")
    cache_map = spatial_embedding_index(manifest_rows)
    ranker = _load_ranker(exp014_checkpoint, device)
    train_latent = _latent_features(ranker, train_examples, device, BATCH_SIZE)
    val_latent = _latent_features(ranker, val_examples, device, BATCH_SIZE)
    train_context, train_predicted = train_latent[:, :128], train_latent[:, 128:129]
    val_context, val_predicted = val_latent[:, :128], val_latent[:, 128:129]
    train_s0 = _spatial_matrix(train_examples, "s0", embeddings, cache_map)
    train_s1 = _spatial_matrix(train_examples, "s1", embeddings, cache_map)
    val_s0 = _spatial_matrix(val_examples, "s0", embeddings, cache_map)
    val_s1 = _spatial_matrix(val_examples, "s1", embeddings, cache_map)
    targets = np.asarray([float(row["target_regression"]) for row in train_examples], dtype=np.float32)
    model = SpatialRGBUtilityRegressor().to(device)
    history, final_loss = _train(model, train_context, train_predicted, train_s0, train_s1, targets, seed, device)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "spatial_rgb_utility_gate_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": "EXP025", "model_state_dict": model.state_dict(), "model_config": {"input_dim": 513, "spatial_tokens": [16, 768], "spatial_grid": [4, 4], "loss": "SmoothL1Loss"}, "training": {"seed": seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE}, "exp014_frozen": True, "dino_frozen": True}, checkpoint)
    training = {"experiment_id": "EXP025", "split": "train", "episode_count": len(train_examples), "target_mean": float(targets.mean()), "final_loss": final_loss, "loss_history": history, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "loss": "SmoothL1Loss", "selection": "final_epoch_fixed_30", "val_used_for_selection": False, "test_used": False}
    (runtime_dir / "training_summary.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    val_target = np.asarray([float(row["target_regression"]) for row in val_examples], dtype=np.float64)
    predictions = _predict(model, val_context, val_predicted, val_s0, val_s1, device)
    errors = predictions - val_target
    sign_labels = val_target > 0.0
    sign_metrics = gate_metrics(predictions, sign_labels.tolist(), 0.0)
    sign_metrics.update({"roc_auc": binary_roc_auc(predictions, sign_labels.tolist()), "pr_auc": binary_average_precision(predictions, sign_labels.tolist()), "positive_prevalence": float(sign_labels.mean())})
    originals = {str(row["episode_id"]): row for row in val_pred_rows}
    exp025_pred_rows = _prediction_rows(val_examples, predictions, originals)
    _write_jsonl(runtime_dir / "val_predictions.jsonl", exp025_pred_rows)
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    alignment = validate_exp016_episode_alignment(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_feature_rows, exp014_prediction_rows=val_pred_rows)
    exp014_rows = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, val_pred_rows)
    exp025_rows = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, exp025_pred_rows)
    oracle_rows, _, _ = build_executed_candidate_oracle_gate_trajectories(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_feature_rows, exp014_prediction_rows=val_pred_rows)
    categories = _categories(label_mapping)
    metrics = [_metric_row("EXP014", summarize_trajectory_rows(exp014_rows, categories), "predicted utility > 0", "frozen learned c_hat"), _load_metric(exp022_result, "EXP022"), _load_metric(exp024_result, "EXP024"), _metric_row("EXP025", summarize_trajectory_rows(exp025_rows, categories), "predicted U_exec > 0", "frozen learned c_hat"), _metric_row("ExecutedCandidateOracle", summarize_trajectory_rows(oracle_rows, categories), "true U2(c_hat) > 0", "frozen learned c_hat")]
    by_name = {row["variant"]: row for row in metrics}
    oracle_gap_acc = by_name["ExecutedCandidateOracle"]["accuracy"] - by_name["EXP014"]["accuracy"]
    oracle_gap_regret = by_name["EXP014"]["mean_regret"] - by_name["ExecutedCandidateOracle"]["mean_regret"]
    action_change = _action_change({str(row["episode_id"]): row for row in val_pred_rows}, {str(row["episode_id"]): row for row in exp025_pred_rows}, val_examples)
    result: dict[str, Any] = {"experiment_id": "EXP025", "experiment_name": "dinov2_spatial_rgb_utility_regression", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "exp014_frozen": True, "val_episode_count": len(stage_b_val), "v0_move_eligible_episode_count": len(val_examples), "train": training, "model": {"dino_model": DINO_MODEL_NAME, "dino_frozen": True, "patch_grid": [16, 16], "pooled_grid": [4, 4], "spatial_tokens": [SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM], "input_dim": 513, "spatial_projector": "Linear(768,128) -> GELU (shared)", "spatial_encoder": "TransformerEncoder(d_model=128,nhead=4,num_layers=1,dim_feedforward=256,dropout=0.1,shared)", "regression_head": "Linear(513,128) -> GELU -> Linear(128,64) -> GELU -> Linear(64,1)", "loss": "SmoothL1Loss", "target": "raw true_U2(c_hat)"}, "rgb_audit": {"unique_train_rgb_observations": len(train_keys), "unique_val_rgb_observations": len(val_keys), "unique_union_rgb_observations": len(all_keys), **cache_info, "cache_manifest": str((embedding_cache / "manifest.jsonl").resolve()), "future_candidate_rgb_used": False, "requested_view_roles": ["s0", "s1"]}, "val_regression_metrics": {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))), **_correlation(predictions, val_target)}, "val_sign_gate_metrics_against_y_exec": sign_metrics, "metrics_table": metrics, "exp025_vs_exp024_delta": {"accuracy": by_name["EXP025"]["accuracy"] - by_name["EXP024"]["accuracy"], "macro_f1": by_name["EXP025"]["macro_f1"] - by_name["EXP024"]["macro_f1"], "mean_regret": by_name["EXP025"]["mean_regret"] - by_name["EXP024"]["mean_regret"], "p90_regret": by_name["EXP025"]["p90_regret"] - by_name["EXP024"]["p90_regret"]}, "headroom_recovery": {"accuracy_gain": by_name["EXP025"]["accuracy"] - by_name["EXP014"]["accuracy"], "accuracy_oracle_gap": oracle_gap_acc, "accuracy_recovery": (by_name["EXP025"]["accuracy"] - by_name["EXP014"]["accuracy"]) / oracle_gap_acc if abs(oracle_gap_acc) > 1e-12 else None, "regret_reduction": by_name["EXP014"]["mean_regret"] - by_name["EXP025"]["mean_regret"], "regret_oracle_gap": oracle_gap_regret, "regret_recovery": (by_name["EXP014"]["mean_regret"] - by_name["EXP025"]["mean_regret"]) / oracle_gap_regret if abs(oracle_gap_regret) > 1e-12 else None}, "action_change": action_change, "episode_alignment": alignment, "provenance": {"source_commit": _git_commit(), "exp014_checkpoint": str(exp014_checkpoint.resolve()), "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint), "train_predictions_sha256": file_sha256(train_predictions), "val_predictions_sha256": file_sha256(val_predictions), "v0_val_predictions_sha256": file_sha256(v0_predictions), "exp022_result_sha256": file_sha256(exp022_result), "exp024_result_sha256": file_sha256(exp024_result), "rgb_dataset_summary_sha256": file_sha256(rgb_root / "dataset_summary.json") if (rgb_root / "dataset_summary.json").is_file() else None, "spatial_cache_summary_sha256": file_sha256(embedding_cache / "summary.json"), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint)}, "validity": {"candidate_ranking_frozen": True, "first_step_protocol_frozen": True, "exp014_parameters_frozen": True, "dino_parameters_frozen": True, "true_u2_used_only_as_train_target": True, "true_u2_used_as_model_input": False, "future_candidate_rgb_used": False, "val_used_for_selection": False, "test_split_accepted": False, "candidate_identity_mismatch_count": int(action_change["candidate_identity_mismatch_count"])}}
    if result["validity"]["candidate_identity_mismatch_count"] != 0:
        raise ValueError("EXP025 changed frozen c_hat candidate identity")
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
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--exp014-checkpoint", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/checkpoints/sequential_observation_ranker_best.pth")
    parser.add_argument("--train-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/train_second_step_predictions.jsonl")
    parser.add_argument("--val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/val_second_step_predictions.jsonl")
    parser.add_argument("--v0-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    parser.add_argument("--exp022-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP022_executed_utility_gate/result.json")
    parser.add_argument("--exp024-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP024_dinov2_rgb_context/result.json")
    parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--rgb-root", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP025_dinov2_spatial_rgb/result.json")
    parser.add_argument("--runtime-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP025_dinov2_spatial_rgb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", dest="device_name", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser


def main() -> None:
    result = analyze(**vars(build_parser().parse_args()))
    print(json.dumps({"experiment_id": "EXP025", "status": result["status"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
