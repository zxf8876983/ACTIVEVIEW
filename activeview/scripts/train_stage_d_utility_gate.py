#!/usr/bin/env python3
"""Train and evaluate EXP022's raw executed-utility regression gate."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
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

from activeview.active_view.stage_c_metrics import regression_metrics
from activeview.active_view.stage_d_contextual_gate import CONTEXTUAL_GATE_INPUT_DIM
from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import validate_exp016_episode_alignment
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_executed_gate import build_executed_candidate_oracle_gate_trajectories
from activeview.active_view.stage_d_gate_calibration import binary_average_precision, binary_roc_auc, gate_metrics
from activeview.active_view.stage_d_utility_gate import (
    UtilityExecutedGateMLP,
    apply_utility_gate_decision,
    build_utility_gate_rows,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.scripts.train_stage_d_contextual_gate import _latent_features, _load_ranker


EXP022_EPOCHS = 30
EXP022_BATCH_SIZE = 256
EXP022_LEARNING_RATE = 1e-3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    invalid = {
        "<missing>" if row.get("policy_split") is None else str(row["policy_split"]).lower()
        for row in rows
        if str(row.get("policy_split", "")).lower() != split
    }
    if invalid:
        raise ValueError(f"{name} must explicitly contain only {split} rows: {sorted(invalid)}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _train_regressor(
    model: UtilityExecutedGateMLP,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[float], float]:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    loader = DataLoader(
        dataset,
        batch_size=EXP022_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=EXP022_LEARNING_RATE)
    criterion = nn.SmoothL1Loss()
    history: list[float] = []
    model.train()
    for _epoch in range(EXP022_EPOCHS):
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_targets in loader:
            predictions = model(batch_features.to(device))
            loss = criterion(predictions, batch_targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch_features.size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        history.append(total_loss / max(total_count, 1))
    return history, float(history[-1])


def _predict_regression(
    model: UtilityExecutedGateMLP,
    features: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        values = model(torch.from_numpy(features).to(device)).cpu().numpy().astype(np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("EXP022 predicted executed utilities must be finite")
    return values


def _prediction_rows(
    examples: Sequence[Mapping[str, Any]],
    predicted: Sequence[float],
    originals: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, value in zip(examples, predicted):
        decision = apply_utility_gate_decision(int(example["candidate_id"]), float(value))
        row = dict(originals[str(example["episode_id"])])
        row.update(
            {
                "predicted_stays": decision["predicted_stays"],
                "predicted_candidate_viewpoint_id": decision["predicted_candidate_viewpoint_id"],
                "predicted_exec_utility": decision["predicted_utility"],
                "utility_gate_target": float(example["target_regression"]),
            }
        )
        rows.append(row)
    return rows


def _utility_error_analysis(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted_move = predicted > 0.0
    target_move = target > 0.0

    def _summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(values.size),
            "mean_true_utility": float(values.mean()) if values.size else 0.0,
            "median_true_utility": float(np.median(values)) if values.size else 0.0,
        }

    false_move = target[predicted_move & ~target_move]
    false_stay = target[~predicted_move & target_move]
    false_move_summary = _summary(false_move)
    false_move_summary["total_negative_utility_magnitude"] = float(-false_move.sum())
    false_stay_summary = _summary(false_stay)
    false_stay_summary["total_missed_positive_utility"] = float(false_stay.sum())
    return {
        "false_move": false_move_summary,
        "false_stay": false_stay_summary,
    }


def _action_change(
    exp014: Mapping[str, Mapping[str, Any]],
    exp022: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {
        "stay_to_move": [],
        "move_to_stay": [],
        "both_stay": [],
        "both_move": [],
    }
    mismatches = 0
    for example in examples:
        episode_id = str(example["episode_id"])
        old = exp014[episode_id]
        new = exp022[episode_id]
        old_move = not bool(old["predicted_stays"])
        new_move = not bool(new["predicted_stays"])
        if old_move and new_move:
            key = "both_move"
            mismatches += int(
                int(old["predicted_candidate_viewpoint_id"])
                != int(new["predicted_candidate_viewpoint_id"])
            )
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
        positive = int(np.sum(array > 0.0))
        result[key] = {
            "count": int(array.size),
            "positive_count": positive,
            "nonpositive_count": int(array.size - positive),
            "mean_true_utility": float(array.mean()) if array.size else 0.0,
            "median_true_utility": float(np.median(array)) if array.size else 0.0,
            "sum_true_utility": float(array.sum()) if array.size else 0.0,
        }
    return result


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


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _load_result_metric(path: Path, variant: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("metrics_table", []):
        if row.get("variant") == variant:
            return dict(row)
    return None


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    exp014_checkpoint: Path,
    train_predictions: Path,
    val_predictions: Path,
    v0_predictions: Path,
    exp019_result: Path,
    exp020_result: Path,
    label_mapping: Path,
    output: Path,
    runtime_dir: Path,
    seed: int = 42,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Train on Train raw U2 and evaluate the fixed zero-sign gate once on Val."""
    _seed_everything(seed)
    device = _device(device_name)
    if not exp014_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen EXP014 checkpoint not found: {exp014_checkpoint}")
    for path, name in ((exp019_result, "EXP019 result"), (exp020_result, "EXP020 result")):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

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

    train_examples = build_utility_gate_rows(
        feature_rows=train_feature_rows,
        prediction_rows=train_pred_rows,
        current_mean=stats["current_mean"],
        current_std=stats["current_std"],
        delta_mean=stats["delta_mean"],
        delta_std=stats["delta_std"],
        geometry_mean=stats["geometry_mean"],
        geometry_std=stats["geometry_std"],
        split="train",
    )
    frozen_ranker = _load_ranker(exp014_checkpoint, device)
    train_x = _latent_features(frozen_ranker, train_examples, device, EXP022_BATCH_SIZE)
    train_y = np.asarray([float(row["target_regression"]) for row in train_examples], dtype=np.float32)
    model = UtilityExecutedGateMLP().to(device)
    history, final_loss = _train_regressor(model, train_x, train_y, seed=seed, device=device)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "executed_utility_gate_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": "EXP022",
            "model_state_dict": model.state_dict(),
            "model_config": {"input_dim": CONTEXTUAL_GATE_INPUT_DIM, "hidden_dim": 64, "activation": "GELU", "loss": "SmoothL1Loss"},
            "training": {"seed": seed, "epochs": EXP022_EPOCHS, "batch_size": EXP022_BATCH_SIZE, "learning_rate": EXP022_LEARNING_RATE},
            "exp014_frozen": True,
            "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint),
        },
        checkpoint,
    )
    training_summary = {
        "experiment_id": "EXP022",
        "split": "train",
        "episode_count": len(train_examples),
        "positive_target_count": int(np.sum(train_y > 0.0)),
        "nonpositive_target_count": int(np.sum(train_y <= 0.0)),
        "target_mean": float(train_y.mean()),
        "final_loss": final_loss,
        "loss_history": history,
        "epochs": EXP022_EPOCHS,
        "batch_size": EXP022_BATCH_SIZE,
        "learning_rate": EXP022_LEARNING_RATE,
        "loss": "SmoothL1Loss",
        "selection": "final_epoch_fixed_30",
        "val_used_for_selection": False,
        "test_used": False,
        "exp014_checkpoint": str(exp014_checkpoint.resolve()),
        "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint),
    }
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    val_examples = build_utility_gate_rows(
        feature_rows=val_feature_rows,
        prediction_rows=val_pred_rows,
        current_mean=stats["current_mean"],
        current_std=stats["current_std"],
        delta_mean=stats["delta_mean"],
        delta_std=stats["delta_std"],
        geometry_mean=stats["geometry_mean"],
        geometry_std=stats["geometry_std"],
        split="val",
    )
    val_x = _latent_features(frozen_ranker, val_examples, device, EXP022_BATCH_SIZE)
    val_predicted = _predict_regression(model, val_x, device=device)
    val_target = np.asarray([float(row["target_regression"]) for row in val_examples], dtype=np.float64)
    regression = regression_metrics(val_predicted, val_target)
    sign = gate_metrics(val_predicted, (val_target > 0.0).tolist(), 0.0)
    sign.update(
        {
            "positive_prevalence": float(np.mean(val_target > 0.0)),
            "roc_auc": binary_roc_auc(val_predicted, (val_target > 0.0).tolist()),
            "pr_auc": binary_average_precision(val_predicted, (val_target > 0.0).tolist()),
        }
    )
    val_originals = {str(row["episode_id"]): row for row in val_pred_rows}
    exp022_prediction_rows = _prediction_rows(val_examples, val_predicted, val_originals)
    _write_jsonl(runtime_dir / "val_utility_gate_predictions.jsonl", exp022_prediction_rows)

    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    alignment = validate_exp016_episode_alignment(
        stage_b_rows=stage_b_val,
        v0_prediction_rows=v0_val,
        cache_rows=val_feature_rows,
        exp014_prediction_rows=val_pred_rows,
    )
    exp014_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, val_pred_rows)
    exp022_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, exp022_prediction_rows)
    oracle_trajectories, _, _ = build_executed_candidate_oracle_gate_trajectories(
        stage_b_rows=stage_b_val,
        v0_prediction_rows=v0_val,
        cache_rows=val_feature_rows,
        exp014_prediction_rows=val_pred_rows,
    )
    categories = _categories(label_mapping)
    exp014_summary = summarize_trajectory_rows(exp014_trajectories, categories)
    exp022_summary = summarize_trajectory_rows(exp022_trajectories, categories)
    oracle_summary = summarize_trajectory_rows(oracle_trajectories, categories)
    exp020_metric = _load_result_metric(exp020_result, "EXP020")
    if exp020_metric is None:
        raise ValueError("Frozen EXP020 metric is required")
    exp014_metric = _load_result_metric(exp019_result, "EXP014")
    if exp014_metric is None:
        raise ValueError("Frozen EXP014 metric is required")
    table = [
        exp014_metric,
        exp020_metric,
        _metric_row("EXP022", exp022_summary, "predicted executed utility > 0", "frozen learned c_hat"),
        _metric_row("ExecutedCandidateOracle", oracle_summary, "true U2(c_hat) > 0", "frozen learned c_hat"),
    ]
    metrics = {str(row["variant"]): row for row in table}
    accuracy_oracle_gap = metrics["ExecutedCandidateOracle"]["accuracy"] - metrics["EXP014"]["accuracy"]
    regret_oracle_gap = metrics["EXP014"]["mean_regret"] - metrics["ExecutedCandidateOracle"]["mean_regret"]
    result = {
        "experiment_id": "EXP022",
        "experiment_name": "executed_candidate_utility_regression_gate",
        "status": "COMPLETED",
        "decision": "ACCEPT",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "exp014_frozen": True,
        "val_episode_count": len(stage_b_val),
        "v0_move_episode_count": len(val_examples),
        "train": training_summary,
        "model": {"input_dim": CONTEXTUAL_GATE_INPUT_DIM, "hidden_dim": 64, "activation": "GELU", "loss": "SmoothL1Loss", "token_extracted_before": "EXP014.utility_head"},
        "val_regression_metrics": regression,
        "val_sign_gate_metrics_against_y_exec": sign,
        "utility_aware_error_analysis": _utility_error_analysis(val_predicted, val_target),
        "metrics_table": table,
        "headroom_recovery": {
            "oracle_accuracy_gap": accuracy_oracle_gap,
            "accuracy_gain": metrics["EXP022"]["accuracy"] - metrics["EXP014"]["accuracy"],
            "accuracy_recovery": (metrics["EXP022"]["accuracy"] - metrics["EXP014"]["accuracy"]) / accuracy_oracle_gap if abs(accuracy_oracle_gap) > 1e-12 else None,
            "oracle_regret_gap": regret_oracle_gap,
            "regret_reduction": metrics["EXP014"]["mean_regret"] - metrics["EXP022"]["mean_regret"],
            "regret_recovery": (metrics["EXP014"]["mean_regret"] - metrics["EXP022"]["mean_regret"]) / regret_oracle_gap if abs(regret_oracle_gap) > 1e-12 else None,
        },
        "action_change": _action_change(
            {str(row["episode_id"]): row for row in val_pred_rows},
            {str(row["episode_id"]): row for row in exp022_prediction_rows},
            val_examples,
        ),
        "episode_alignment": alignment,
        "provenance": {
            "source_commit": _git_commit(),
            "exp014_checkpoint": str(exp014_checkpoint.resolve()),
            "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint),
            "stage_d_feature_summary": str(summary_path.resolve()),
            "stage_d_feature_summary_sha256": file_sha256(summary_path),
            "stage_d_feature_stats": str(stats_path.resolve()),
            "stage_d_feature_stats_sha256": file_sha256(stats_path),
            "stage_d_train_features": str(Path(summary["feature_files"]["train"]).resolve()),
            "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])),
            "stage_d_val_features": str(Path(summary["feature_files"]["val"]).resolve()),
            "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])),
            "train_predictions": str(train_predictions.resolve()),
            "train_predictions_sha256": file_sha256(train_predictions),
            "val_predictions": str(val_predictions.resolve()),
            "val_predictions_sha256": file_sha256(val_predictions),
            "v0_val_predictions": str(v0_predictions.resolve()),
            "stage_b_val_utility": str((stage_b_root / "utility_labels" / "val.jsonl").resolve()),
            "exp019_result": str(exp019_result.resolve()),
            "exp019_result_sha256": file_sha256(exp019_result),
            "exp020_result": str(exp020_result.resolve()),
            "exp020_result_sha256": file_sha256(exp020_result),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "validity": {
            "candidate_ranking_frozen": True,
            "first_step_protocol_frozen": True,
            "contextual_token_before_utility_head": True,
            "exp014_parameters_frozen": True,
            "true_u2_used_only_as_train_target": True,
            "true_u2_used_as_model_input": False,
            "val_used_for_selection": False,
            "strict_decision": "predicted_U_exec > 0",
            "final_epoch_used": True,
            "test_split_accepted": False,
        },
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    (runtime_dir / "result.json").write_text(payload, encoding="utf-8")
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--exp014-checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--exp019-result", type=Path, required=True)
    parser.add_argument("--exp020-result", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        cache_root=args.cache_root,
        stage_b_root=args.stage_b_root,
        exp014_checkpoint=args.exp014_checkpoint,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        v0_predictions=args.v0_predictions,
        exp019_result=args.exp019_result,
        exp020_result=args.exp020_result,
        label_mapping=args.label_mapping,
        output=args.output,
        runtime_dir=args.runtime_dir,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps({"experiment_id": "EXP022", "split": "val", "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"], "val_regression_metrics": result["val_regression_metrics"], "val_sign_gate_metrics_against_y_exec": result["val_sign_gate_metrics_against_y_exec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
