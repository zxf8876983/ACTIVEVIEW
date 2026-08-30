#!/usr/bin/env python3
"""Calibrate EXP014's second-step Move/Stay threshold on Train, then evaluate Val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import StageDDataset, collate_stage_d, load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import (
    build_exp016_variant_trajectories,
    validate_exp016_episode_alignment,
)
from activeview.active_view.stage_d_evaluation import (
    build_stage_d_trajectories,
    predict_second_step_dataset,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_gate_calibration import (
    _aligned_gate_examples,
    _assert_rows_for_split,
    binary_average_precision,
    binary_roc_auc,
    build_thresholded_prediction_rows,
    calibrate_train_threshold,
    candidate_identity_audit,
    gate_metrics,
    load_calibration_artifact,
)
from activeview.active_view.stage_d_policy import SequentialObservationRanker
from activeview.active_view.utility_label_builder import file_sha256


REFERENCE_METRICS = {
    "EXP014": {"accuracy": 0.6582540930864375, "macro_f1": 0.6101526052247462, "mean_regret": 1.4224626188609946},
    "OracleGate_LearnedCandidate": {"accuracy": 0.7200257381854579, "macro_f1": 0.6701903958101036, "mean_regret": 0.9691383557931994},
}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _load_frozen_predictions(
    *,
    cache_root: Path,
    checkpoint: Path,
    split: str,
    output_path: Path,
    device_name: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], Path]:
    """Run inference only with the frozen EXP014 checkpoint when needed."""
    if split not in {"train", "val"}:
        raise ValueError("EXP017 prediction generation accepts Train or Val only")
    summary_path = cache_root / "stage_d_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    dataset = StageDDataset(Path(summary["feature_files"][split]), stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stage_d, num_workers=0)
    device = _device(device_name)
    model = SequentialObservationRanker().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    rows = predict_second_step_dataset(model, loader, device)
    _write_jsonl(output_path, rows)
    return rows, output_path


def _load_or_generate_predictions(
    *,
    path: Path | None,
    cache_root: Path,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], Path]:
    output_path = output_dir / "runtime" / f"{split}_second_step_predictions.jsonl"
    if path is not None and path.exists():
        return load_jsonl(path), path
    if path is None and output_path.exists():
        return load_jsonl(output_path), output_path
    return _load_frozen_predictions(
        cache_root=cache_root,
        checkpoint=checkpoint,
        split=split,
        output_path=output_path,
        device_name=device_name,
        batch_size=batch_size,
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metric_row(name: str, summary: Mapping[str, Any], threshold: float | str | None) -> dict[str, Any]:
    regret = summary["decision_regret"]
    movement = summary["movement"]
    return {
        "variant": name,
        "threshold": (None if threshold is None else (threshold if isinstance(threshold, str) else float(threshold))),
        "accuracy": float(summary["recognition"]["accuracy"]),
        "macro_f1": float(summary["recognition"]["macro_f1"]),
        "mean_regret": float(regret["mean"]),
        "median_regret": float(regret["median"]),
        "p90_regret": float(regret["p90"]),
        "headroom_capture": float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]),
        "average_moves": float(movement["average_moves"]),
        "mean_geodesic_cost_m": float(movement["trajectory_geodesic_cost_m"]["mean"]),
    }


def _reference_checks(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, expected in REFERENCE_METRICS.items():
        actual = metrics[name]
        checks[name] = {
            key: {
                "actual": float(actual[key]),
                "reference": float(value),
                "within_abs_tolerance_1e-5": abs(float(actual[key]) - float(value)) <= 1e-5,
            }
            for key, value in expected.items()
        }
    return checks


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    checkpoint: Path,
    train_predictions: Path | None,
    val_predictions: Path | None,
    v0_predictions: Path,
    label_mapping: Path,
    output_dir: Path,
    calibration_artifact: Path | None = None,
    result_output: Path | None = None,
    device_name: str = "cuda:0",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Calibrate on Train, freeze the artifact, then evaluate the Val policies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cache_root / "stage_d_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_cache = load_jsonl(Path(summary["feature_files"]["train"]))
    train_pred_rows, train_pred_path = _load_or_generate_predictions(
        path=train_predictions, cache_root=cache_root, checkpoint=checkpoint, split="train",
        output_dir=output_dir, device_name=device_name, batch_size=batch_size,
    )
    calibration = calibrate_train_threshold(train_pred_rows, train_cache)
    calibration.update({
        "status": "FROZEN",
        "test_used": False,
        "frozen_exp014_checkpoint": str(checkpoint.resolve()),
        "frozen_exp014_checkpoint_sha256": file_sha256(checkpoint),
        "frozen_train_predictions": str(train_pred_path.resolve()),
        "frozen_train_predictions_sha256": file_sha256(train_pred_path),
        "stage_d_train_cache": str(Path(summary["feature_files"]["train"]).resolve()),
        "stage_d_train_cache_sha256": file_sha256(Path(summary["feature_files"]["train"])),
        "train_episode_count": len(train_cache),
        "train_prediction_episode_count": len(train_pred_rows),
        "source_commit": _git_commit(),
    })
    calibration_path = calibration_artifact or (output_dir / "calibration.json")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    frozen_calibration = load_calibration_artifact(calibration_path)

    # No Val artifact is loaded before the Train calibration has been written
    # and reloaded.  This makes the Train->freeze->Val ordering explicit.
    val_cache = load_jsonl(Path(summary["feature_files"]["val"]))
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_rows_for_split(stage_b_val, "val", "Stage B Val utility")
    _assert_rows_for_split(v0_val, "val", "Stage C-v0 Val prediction")
    val_pred_rows, val_pred_path = _load_or_generate_predictions(
        path=val_predictions, cache_root=cache_root, checkpoint=checkpoint, split="val",
        output_dir=output_dir, device_name=device_name, batch_size=batch_size,
    )
    validate_exp016_episode_alignment(
        stage_b_rows=stage_b_val,
        v0_prediction_rows=v0_val,
        cache_rows=val_cache,
        exp014_prediction_rows=val_pred_rows,
    )
    categories = _categories(label_mapping)
    zero_rows = build_thresholded_prediction_rows(val_pred_rows, val_cache, 0.0)
    calibrated_rows = build_thresholded_prediction_rows(
        val_pred_rows, val_cache, float(frozen_calibration["selected_tau"])
    )
    identity_audit = candidate_identity_audit(zero_rows, calibrated_rows)
    if not identity_audit["candidate_identity_unchanged"]:
        raise ValueError("EXP017 gate-only calibration changed learned candidate identity")
    exp014_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_cache, val_pred_rows)
    zero_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_cache, zero_rows)
    for expected, actual in zip(exp014_trajectories, zero_trajectories):
        if (expected["selected_viewpoint_id"], expected["moves"]) != (actual["selected_viewpoint_id"], actual["moves"]):
            raise ValueError(f"EXP014 tau=0 action mismatch: {expected['episode_id']}")
    calibrated_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_cache, calibrated_rows)
    oracle_gate_trajectories, _ = build_exp016_variant_trajectories(
        stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_cache,
        exp014_prediction_rows=val_pred_rows, gate="oracle", candidate="learned",
    )
    summaries = {
        "EXP014": summarize_trajectory_rows(zero_trajectories, categories),
        "EXP017": summarize_trajectory_rows(calibrated_trajectories, categories),
        "OracleGate_LearnedCandidate": summarize_trajectory_rows(oracle_gate_trajectories, categories),
    }
    metrics_table = [
        _metric_row("EXP014 tau=0", summaries["EXP014"], 0.0),
        _metric_row("EXP017 Train-calibrated", summaries["EXP017"], float(frozen_calibration["selected_tau"])),
        _metric_row("OracleGate + LearnedCandidate", summaries["OracleGate_LearnedCandidate"], "oracle"),
    ]
    exp_metric = metrics_table[0]
    calibrated_metric = metrics_table[1]
    oracle_metric = metrics_table[2]
    gate_accuracy_gap = oracle_metric["accuracy"] - exp_metric["accuracy"]
    gate_regret_gap = exp_metric["mean_regret"] - oracle_metric["mean_regret"]
    val_scores, val_labels = _aligned_gate_examples(val_pred_rows, val_cache, split="val")
    gate_diagnostics = {
        "tau_0": gate_metrics(val_scores, val_labels, 0.0),
        "tau_calibrated": gate_metrics(val_scores, val_labels, float(frozen_calibration["selected_tau"])),
        "roc_auc": binary_roc_auc(val_scores, val_labels),
        "pr_auc": binary_average_precision(val_scores, val_labels),
    }
    metrics_by_name = {"EXP014": exp_metric, "OracleGate_LearnedCandidate": oracle_metric}
    references = _reference_checks(metrics_by_name)
    if not all(item["within_abs_tolerance_1e-5"] for check in references.values() for item in check.values()):
        raise ValueError("Frozen EXP014 or OracleGate reference mismatch")
    result = {
        "experiment_id": "EXP017",
        "experiment_name": "second_step_gate_calibration",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": False,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "episode_count": len(stage_b_val),
        "v0_move_eligible_episode_count": sum(not bool(row["predicted_stays"]) for row in v0_val),
        "calibration": {
            "artifact": str(calibration_path.resolve()),
            "artifact_sha256": file_sha256(calibration_path),
            "selected_tau": float(frozen_calibration["selected_tau"]),
            "train_gate_metrics": frozen_calibration["train_gate_metrics"],
            "train_episode_count": len(train_cache),
            "train_prediction_episode_count": len(train_pred_rows),
        },
        "metrics_table": metrics_table,
        "headroom_recovery": {
            "gate_oracle_accuracy_gap": gate_accuracy_gap,
            "calibration_accuracy_gain": calibrated_metric["accuracy"] - exp_metric["accuracy"],
            "calibration_gate_gap_recovery": _safe_fraction(calibrated_metric["accuracy"] - exp_metric["accuracy"], gate_accuracy_gap),
            "gate_oracle_mean_regret_gap": gate_regret_gap,
            "calibration_mean_regret_reduction": exp_metric["mean_regret"] - calibrated_metric["mean_regret"],
            "calibration_regret_gap_recovery": _safe_fraction(exp_metric["mean_regret"] - calibrated_metric["mean_regret"], gate_regret_gap),
        },
        "val_gate_diagnostics": gate_diagnostics,
        "candidate_identity_audit": identity_audit,
        "reference_checks": references,
        "source_episode_counts": {
            "stage_b_val": len(stage_b_val),
            "stage_c_v0_val_predictions": len(v0_val),
            "stage_d_cache_val": len(val_cache),
            "exp014_val_predictions": len(val_pred_rows),
        },
        "provenance": {
            "frozen_exp014_checkpoint": str(checkpoint.resolve()),
            "frozen_exp014_checkpoint_sha256": file_sha256(checkpoint),
            "stage_d_train_cache": str(Path(summary["feature_files"]["train"]).resolve()),
            "stage_d_train_cache_sha256": file_sha256(Path(summary["feature_files"]["train"])),
            "stage_d_val_cache": str(Path(summary["feature_files"]["val"]).resolve()),
            "stage_d_val_cache_sha256": file_sha256(Path(summary["feature_files"]["val"])),
            "train_predictions": str(train_pred_path.resolve()),
            "train_predictions_sha256": file_sha256(train_pred_path),
            "val_predictions": str(val_pred_path.resolve()),
            "val_predictions_sha256": file_sha256(val_pred_path),
            "source_commit": _git_commit(),
        },
        "validity": {
            "threshold_fit_split": "train",
            "threshold_applied_split": "val",
            "candidate_ranking_frozen": True,
            "first_step_protocol_frozen": True,
            "test_split_accepted": False,
        },
    }
    destination = result_output or (output_dir / "result.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path)
    parser.add_argument("--val-predictions", type=Path)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        cache_root=args.cache_root, stage_b_root=args.stage_b_root, checkpoint=args.checkpoint,
        train_predictions=args.train_predictions, val_predictions=args.val_predictions,
        v0_predictions=args.v0_predictions, label_mapping=args.label_mapping,
        output_dir=args.output_dir, calibration_artifact=args.calibration_artifact,
        result_output=args.result_output, device_name=args.device, batch_size=args.batch_size,
    )
    print(json.dumps({"experiment_id": "EXP017", "split": "val", "test_used": False, "selected_tau": result["calibration"]["selected_tau"], "metrics_table": result["metrics_table"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
