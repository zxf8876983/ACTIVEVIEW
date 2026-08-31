#!/usr/bin/env python3
"""Train/evaluate EXP026 spatial RGB-D executed-utility regression."""

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
from activeview.active_view.stage_d_rgb_context import DINO_EMBED_DIM, RGBObservationKey, load_dinov2, observation_keys_from_feature_rows
from activeview.active_view.stage_d_rgb_spatial import SPATIAL_TOKEN_COUNT, spatial_embedding_index
from activeview.active_view.stage_d_depth_spatial import SpatialRGBDUtilityRegressor, build_or_load_depth_cache
from activeview.active_view.stage_d_utility_gate import build_utility_gate_rows
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.scripts.train_stage_d_contextual_gate import _latent_features, _load_ranker
from activeview.scripts.train_stage_d_rgb_context import _action_change, _categories, _load_metric, _metric_row


EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    invalid = {"<missing>" if row.get("policy_split") is None else str(row["policy_split"]).lower() for row in rows if str(row.get("policy_split", "")).lower() != split}
    if invalid: raise ValueError(f"{name} must explicitly contain only {split} rows: {sorted(invalid)}")


def _attach_keys(examples: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]]) -> set[RGBObservationKey]:
    _, mapping = observation_keys_from_feature_rows(rows); requested: set[RGBObservationKey] = set()
    for example in examples:
        pair = mapping.get(str(example["episode_id"]))
        if pair is None: raise ValueError(f"Missing observation keys for {example['episode_id']}")
        example["rgb_s0_key"], example["rgb_s1_key"] = pair; requested.update(pair)
    return requested


def _load_spatial_cache(path: Path, keys: Sequence[RGBObservationKey]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.load(path / "embeddings.npy", mmap_mode="r")
    rows = [json.loads(line) for line in (path / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if values.shape != (len(keys), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM) or values.dtype != np.float16:
        raise ValueError("EXP025 spatial cache schema mismatch")
    index = spatial_embedding_index(rows); expected = {key.tuple for key in keys}
    if set(index) != expected: raise ValueError("EXP025 spatial cache keys do not match visited observations")
    return values, rows


def _matrix(examples: Sequence[Mapping[str, Any]], role: str, values: np.ndarray, index: Mapping[tuple[str, str, str, int], int], shape: tuple[int, ...]) -> np.ndarray:
    result = np.stack([np.asarray(values[index[example[f"rgb_{role}_key"].tuple]], dtype=np.float32) for example in examples]).astype(np.float32)
    if result.shape != (len(examples),) + shape or not np.isfinite(result).all(): raise ValueError(f"Invalid {role} cache matrix")
    return result


def _train(model: SpatialRGBDUtilityRegressor, context: np.ndarray, predicted: np.ndarray, rgb0: np.ndarray, rgb1: np.ndarray, depth0: np.ndarray, depth1: np.ndarray, target: np.ndarray, seed: int, device: torch.device) -> tuple[list[float], float]:
    dataset = TensorDataset(*(torch.from_numpy(x) for x in (context, predicted, rgb0, rgb1, depth0, depth1, target)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE); criterion = nn.SmoothL1Loss(); history: list[float] = []; model.train()
    for _ in range(EPOCHS):
        total = count = 0
        for batch in loader:
            output = model(*(item.to(device) for item in batch[:-1])); loss = criterion(output, batch[-1].to(device)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); n = int(batch[-1].shape[0]); total += float(loss.detach().cpu()) * n; count += n
        history.append(total / max(count, 1))
    return history, float(history[-1])


def _predict(model: SpatialRGBDUtilityRegressor, arrays: Sequence[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval(); out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), BATCH_SIZE):
            end = start + BATCH_SIZE; tensors = [torch.from_numpy(x[start:end]).to(device) for x in arrays]; out.append(model(*tensors).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _correlations(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        return None if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12 else float(np.corrcoef(a, b)[0, 1])
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort"); out = np.empty(a.size); i = 0
        while i < a.size:
            j = i + 1
            while j < a.size and a[order[j]] == a[order[i]]: j += 1
            out[order[i:j]] = (i + j + 1) / 2.0; i = j
        return out
    return {"pearson": corr(predicted, target), "spearman": corr(rank(predicted), rank(target))}


def _prediction_rows(examples: Sequence[Mapping[str, Any]], values: np.ndarray, originals: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, value in zip(examples, values):
        move = float(value) > 0.0; row = dict(originals[str(example["episode_id"])]); row.update({"predicted_stays": not move, "predicted_candidate_viewpoint_id": int(example["candidate_id"]) if move else None, "predicted_exec_utility": float(value), "utility_gate_target": float(example["true_utility"]) }); rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def analyze(*, cache_root: Path, stage_b_root: Path, source_root: Path, motion_manifest: Path, scene_root: Path, exp014_checkpoint: Path, train_predictions: Path, val_predictions: Path, v0_predictions: Path, exp022_result: Path, exp024_result: Path, exp025_result: Path, exp025_cache: Path, label_mapping: Path, rgb_root: Path, depth_cache: Path, output: Path, runtime_dir: Path, workers: int = 16, seed: int = 42, device_name: str = "cuda:0") -> dict[str, Any]:
    _seed(seed); device = _device(device_name); summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text()); stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    train_f, val_f = load_jsonl(Path(summary["feature_files"]["train"])), load_jsonl(Path(summary["feature_files"]["val"])); train_p, val_p = load_jsonl(train_predictions), load_jsonl(val_predictions)
    for rows, split, name in ((train_f, "train", "features"), (val_f, "val", "features"), (train_p, "train", "predictions"), (val_p, "val", "predictions")): _assert_split(rows, split, name)
    common = dict(current_mean=stats["current_mean"], current_std=stats["current_std"], delta_mean=stats["delta_mean"], delta_std=stats["delta_std"], geometry_mean=stats["geometry_mean"], geometry_std=stats["geometry_std"])
    train_ex = build_utility_gate_rows(feature_rows=train_f, prediction_rows=train_p, split="train", **common); val_ex = build_utility_gate_rows(feature_rows=val_f, prediction_rows=val_p, split="val", **common)
    train_keys, val_keys = _attach_keys(train_ex, train_f), _attach_keys(val_ex, val_f)
    if train_keys & val_keys: raise ValueError("Train/Val RGB key overlap")
    spatial_values, spatial_rows = _load_spatial_cache(exp025_cache, sorted(train_keys | val_keys, key=lambda key: key.tuple)); spatial_index = spatial_embedding_index(spatial_rows)
    depth_values, depth_rows, depth_info = build_or_load_depth_cache(source_root=source_root, motion_manifest=motion_manifest, scene_root=scene_root, cache_dir=depth_cache, keys=sorted(train_keys | val_keys, key=lambda key: key.tuple), workers=workers); depth_index = { (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"])): i for i, row in enumerate(depth_rows) }
    ranker = _load_ranker(exp014_checkpoint, device); train_latent, val_latent = _latent_features(ranker, train_ex, device, BATCH_SIZE), _latent_features(ranker, val_ex, device, BATCH_SIZE); tc, tp, vc, vp = train_latent[:, :128], train_latent[:, 128:129], val_latent[:, :128], val_latent[:, 128:129]
    tr_rgb0 = _matrix(train_ex, "s0", spatial_values, spatial_index, (16, 768))
    tr_rgb1 = _matrix(train_ex, "s1", spatial_values, spatial_index, (16, 768))
    va_rgb0 = _matrix(val_ex, "s0", spatial_values, spatial_index, (16, 768))
    va_rgb1 = _matrix(val_ex, "s1", spatial_values, spatial_index, (16, 768))
    tr_d0 = _matrix(train_ex, "s0", depth_values, depth_index, (16, 4))
    tr_d1 = _matrix(train_ex, "s1", depth_values, depth_index, (16, 4))
    va_d0 = _matrix(val_ex, "s0", depth_values, depth_index, (16, 4))
    va_d1 = _matrix(val_ex, "s1", depth_values, depth_index, (16, 4))
    target = np.asarray([float(row["target_regression"]) for row in train_ex], dtype=np.float32); model = SpatialRGBDUtilityRegressor().to(device); history, final_loss = _train(model, tc, tp, tr_rgb0, tr_rgb1, tr_d0, tr_d1, target, seed, device)
    runtime_dir.mkdir(parents=True, exist_ok=True); checkpoint = runtime_dir / "checkpoints" / "spatial_rgbd_utility_gate_final.pth"; checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"experiment_id": "EXP026", "model_state_dict": model.state_dict(), "model_config": {"input_dim": 609, "loss": "SmoothL1Loss"}, "exp014_frozen": True, "dino_frozen": True}, checkpoint)
    training = {"episode_count": len(train_ex), "final_loss": final_loss, "loss_history": history, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "test_used": False}; val_target = np.asarray([float(row["target_regression"]) for row in val_ex], dtype=np.float64); predictions = _predict(model, (vc, vp, va_rgb0, va_rgb1, va_d0, va_d1), device); errors = predictions - val_target; sign = gate_metrics(predictions, (val_target > 0).tolist(), 0.0); sign.update({"roc_auc": binary_roc_auc(predictions, (val_target > 0).tolist()), "pr_auc": binary_average_precision(predictions, (val_target > 0).tolist()), "positive_prevalence": float(np.mean(val_target > 0))})
    originals = {str(row["episode_id"]): row for row in val_p}; pred_rows = _prediction_rows(val_ex, predictions, originals); _write_jsonl(runtime_dir / "val_predictions.jsonl", pred_rows)
    stage_b_val, v0_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl"), load_jsonl(v0_predictions); _assert_split(stage_b_val, "val", "Stage B"); _assert_split(v0_val, "val", "v0"); alignment = validate_exp016_episode_alignment(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_f, exp014_prediction_rows=val_p); exp014_rows, exp026_rows = build_stage_d_trajectories(stage_b_val, v0_val, val_f, val_p), build_stage_d_trajectories(stage_b_val, v0_val, val_f, pred_rows); oracle_rows, _, _ = build_executed_candidate_oracle_gate_trajectories(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_f, exp014_prediction_rows=val_p); categories = _categories(label_mapping)
    metrics = [_metric_row("EXP014", summarize_trajectory_rows(exp014_rows, categories), "predicted utility > 0", "frozen learned c_hat"), _load_metric(exp022_result, "EXP022"), _load_metric(exp024_result, "EXP024"), _load_metric(exp025_result, "EXP025"), _metric_row("EXP026", summarize_trajectory_rows(exp026_rows, categories), "predicted U_exec > 0", "frozen learned c_hat"), _metric_row("ExecutedCandidateOracle", summarize_trajectory_rows(oracle_rows, categories), "true U2(c_hat) > 0", "frozen learned c_hat")]; by_name = {row["variant"]: row for row in metrics}; action = _action_change({str(row["episode_id"]): row for row in val_p}, {str(row["episode_id"]): row for row in pred_rows}, val_ex)
    result: dict[str, Any] = {"experiment_id": "EXP026", "experiment_name": "spatial_rgbd_utility_regression", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": True, "stgcn_retrained": False, "exp014_frozen": True, "val_episode_count": len(stage_b_val), "v0_move_eligible_episode_count": len(val_ex), "train": training, "model": {"input_dim": 609, "rgb_spatial_tokens": [16, 768], "depth_spatial": [16, 4], "dino_frozen": True, "depth_sensor": "Habitat metric depth", "loss": "SmoothL1Loss", "target": "raw true_U2(c_hat)"}, "depth_audit": {"unique_train_depth_observations": len(train_keys), "unique_val_depth_observations": len(val_keys), "worker_count": workers, "cache_disk_bytes": int(sum(path.stat().st_size for path in depth_cache.iterdir() if path.is_file())), **depth_info, "invalid_depth_pixel_ratio": float(1.0 - np.mean(np.asarray(depth_values[:, :, 3], dtype=np.float32))), "mean_depth_m": float(np.mean(depth_values[:, :, 0])), "median_depth_m": float(np.median(depth_values[:, :, 0])), "max_clipping_rate_gt_10m": None, "max_clipping_rate_note": "Unavailable after compact cache; raw depth is not retained.", "future_candidate_depth_used": False, "frame_index": 15}, "val_regression_metrics": {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))), **_correlations(predictions, val_target)}, "val_sign_gate_metrics_against_y_exec": sign, "metrics_table": metrics, "exp026_vs_exp025_delta": {"accuracy": by_name["EXP026"]["accuracy"] - by_name["EXP025"]["accuracy"], "macro_f1": by_name["EXP026"]["macro_f1"] - by_name["EXP025"]["macro_f1"], "mean_regret": by_name["EXP026"]["mean_regret"] - by_name["EXP025"]["mean_regret"], "p90_regret": by_name["EXP026"]["p90_regret"] - by_name["EXP025"]["p90_regret"]}, "action_change": action, "episode_alignment": alignment, "provenance": {"source_commit": _git_commit(), "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint), "exp022_result_sha256": file_sha256(exp022_result), "exp024_result_sha256": file_sha256(exp024_result), "exp025_result_sha256": file_sha256(exp025_result), "exp025_cache_summary_sha256": file_sha256(exp025_cache / "summary.json"), "depth_cache_summary_sha256": file_sha256(depth_cache / "summary.json"), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint)}, "validity": {"candidate_ranking_frozen": True, "first_step_protocol_frozen": True, "true_u2_used_only_as_train_target": True, "true_u2_used_as_model_input": False, "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "candidate_identity_mismatch_count": int(action["candidate_identity_mismatch_count"])}}
    if result["validity"]["candidate_identity_mismatch_count"] != 0: raise ValueError("EXP026 candidate identity changed")
    payload = json.dumps(result, indent=2, ensure_ascii=False); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(payload, encoding="utf-8"); (runtime_dir / "result.json").write_text(payload, encoding="utf-8"); output.with_name("analysis.md").write_text("# EXP026 analysis\n\nSee result.json for the complete Val-only diagnostics.\n", encoding="utf-8"); return result


def _git_commit() -> str | None:
    import subprocess
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root(); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"); parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b"); parser.add_argument("--source-root", type=Path, default=data_root / "datasets/offline/hm3d-train"); parser.add_argument("--motion-manifest", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json"); parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train"); parser.add_argument("--exp014-checkpoint", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/checkpoints/sequential_observation_ranker_best.pth"); parser.add_argument("--train-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/train_second_step_predictions.jsonl"); parser.add_argument("--val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/val_second_step_predictions.jsonl"); parser.add_argument("--v0-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); parser.add_argument("--exp022-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP022_executed_utility_gate/result.json"); parser.add_argument("--exp024-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP024_dinov2_rgb_context/result.json"); parser.add_argument("--exp025-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP025_dinov2_spatial_rgb/result.json"); parser.add_argument("--exp025-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4")); parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"); parser.add_argument("--rgb-root", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train")); parser.add_argument("--depth-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/habitat_depth_spatial4x4")); parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP026_spatial_rgbd_utility_regression/result.json"); parser.add_argument("--runtime-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP026_spatial_rgbd_utility_regression")); parser.add_argument("--workers", type=int, default=16); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", dest="device_name", default="cuda:0"); return parser


def main() -> None:
    result = analyze(**vars(build_parser().parse_args())); print(json.dumps({"experiment_id": "EXP026", "status": result["status"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__": main()
