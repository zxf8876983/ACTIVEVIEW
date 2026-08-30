#!/usr/bin/env python3
"""Train EXP019's executed-candidate binary gate on Train and evaluate Val."""

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

from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import validate_exp016_episode_alignment
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_executed_gate import (
    build_executed_candidate_oracle_gate_trajectories,
)
from activeview.active_view.stage_d_executed_gate_training import (
    ExecutedCandidateGateMLP,
    build_executed_gate_examples,
)
from activeview.active_view.stage_d_gate_calibration import binary_average_precision, binary_roc_auc, gate_metrics
from activeview.active_view.utility_label_builder import file_sha256


REFERENCE = {
    "EXP014": {"accuracy": 0.6582540930864375, "macro_f1": 0.6101526052247462, "mean_regret": 1.4224626188609946},
    "ExecutedCandidateOracleGate": {"accuracy": 0.7431186101379853, "macro_f1": 0.6932308383305563, "mean_regret": 0.7613394852938011},
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    invalid = {"<missing>" if row.get("policy_split") is None else str(row["policy_split"]).lower() for row in rows if str(row.get("policy_split", "")).lower() != split}
    if invalid:
        raise ValueError(f"{name} must explicitly contain only {split} rows: {sorted(invalid)}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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


def _reference_checks(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected in REFERENCE.items():
        result[name] = {key: {"actual": float(metrics[name][key]), "reference": float(value), "within_abs_tolerance_1e-5": abs(float(metrics[name][key]) - float(value)) <= 1e-5} for key, value in expected.items()}
    return result


def _prediction_rows(examples: Sequence[Mapping[str, Any]], logits: Sequence[float], originals: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, logit in zip(examples, logits):
        original = originals[str(example["episode_id"])]
        move = float(logit) > 0.0
        row = dict(original)
        row["predicted_stays"] = not move
        row["predicted_candidate_viewpoint_id"] = int(example["candidate_id"]) if move else None
        row["gate_logit"] = float(logit)
        row["gate_probability"] = float(1.0 / (1.0 + np.exp(-float(logit))))
        row["gate_target"] = int(example["target"])
        rows.append(row)
    return rows


def _action_change(exp014: Mapping[str, Mapping[str, Any]], exp019: Mapping[str, Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {"stay_to_move": [], "move_to_stay": [], "both_stay": [], "both_move": []}
    for example in examples:
        episode_id = str(example["episode_id"])
        old_move = not bool(exp014[episode_id]["predicted_stays"])
        new_move = not bool(exp019[episode_id]["predicted_stays"])
        if old_move and new_move:
            key = "both_move"
        elif not old_move and not new_move:
            key = "both_stay"
        elif not old_move:
            key = "stay_to_move"
        else:
            key = "move_to_stay"
        groups[key].append(float(example["true_utility"]))
    result: dict[str, Any] = {}
    for key, values in groups.items():
        positive = sum(value > 0.0 for value in values)
        result[key] = {
            "count": len(values),
            "positive_count": positive,
            "nonpositive_count": len(values) - positive,
            "mean_true_utility": float(np.mean(values)) if values else 0.0,
            "median_true_utility": float(np.median(values)) if values else 0.0,
        }
    return result


def _train_gate(model: nn.Module, features: np.ndarray, targets: np.ndarray, *, seed: int, epochs: int, batch_size: int, learning_rate: float, device: torch.device) -> tuple[list[float], float]:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history: list[float] = []
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_targets in loader:
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = int(batch_features.size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        history.append(total_loss / total_count)
    return history, float(history[-1])


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    train_predictions: Path,
    val_predictions: Path,
    v0_predictions: Path,
    label_mapping: Path,
    output: Path,
    runtime_dir: Path,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Train on Train examples exactly 30 epochs, then evaluate Val once."""
    _seed_everything(seed)
    device = _device(device_name)
    summary_path = cache_root / "stage_d_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    statistics = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    train_features_rows = load_jsonl(Path(summary["feature_files"]["train"]))
    val_features_rows = load_jsonl(Path(summary["feature_files"]["val"]))
    train_pred_rows = load_jsonl(train_predictions)
    val_pred_rows = load_jsonl(val_predictions)
    _assert_split(train_features_rows, "train", "Stage D Train features")
    _assert_split(val_features_rows, "val", "Stage D Val features")
    _assert_split(train_pred_rows, "train", "EXP014 Train predictions")
    _assert_split(val_pred_rows, "val", "EXP014 Val predictions")
    train_examples = build_executed_gate_examples(feature_rows=train_features_rows, prediction_rows=train_pred_rows, geometry_mean=statistics["geometry_mean"], geometry_std=statistics["geometry_std"], split="train")
    val_examples = build_executed_gate_examples(feature_rows=val_features_rows, prediction_rows=val_pred_rows, geometry_mean=statistics["geometry_mean"], geometry_std=statistics["geometry_std"], split="val")
    train_x = np.stack([example["features"] for example in train_examples]).astype(np.float32)
    train_y = np.asarray([example["target"] for example in train_examples], dtype=np.float32)
    val_x = np.stack([example["features"] for example in val_examples]).astype(np.float32)
    val_y = np.asarray([example["target"] for example in val_examples], dtype=bool)
    model = ExecutedCandidateGateMLP().to(device)
    history, final_loss = _train_gate(model, train_x, train_y, seed=seed, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "executed_candidate_gate_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": "EXP019", "model_state_dict": model.state_dict(), "model_config": {"input_dim": 12, "hidden_dim": 64}, "training": {"seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate}}, checkpoint)
    training_summary = {"experiment_id": "EXP019", "split": "train", "episode_count": len(train_examples), "positive_count": int(train_y.sum()), "negative_count": int(train_y.size - train_y.sum()), "final_loss": final_loss, "loss_history": history, "selection": "final_epoch_fixed_30", "val_used_for_selection": False, "test_used": False}
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    model.eval()
    with torch.inference_mode():
        val_logits = model(torch.from_numpy(val_x).to(device)).cpu().numpy().astype(np.float64)
    val_gate_metrics = gate_metrics(val_logits, val_y.tolist(), 0.0)
    val_gate_metrics["positive_prevalence"] = float(val_y.mean())
    val_gate_metrics["roc_auc"] = binary_roc_auc(val_logits, val_y.tolist())
    val_gate_metrics["pr_auc"] = binary_average_precision(val_logits, val_y.tolist())
    val_originals = {str(row["episode_id"]): row for row in val_pred_rows}
    exp019_rows = _prediction_rows(val_examples, val_logits, val_originals)
    _write_jsonl(runtime_dir / "val_gate_predictions.jsonl", exp019_rows)
    candidate_identity_mismatch = sum(int(row["predicted_candidate_viewpoint_id"] != int(example["candidate_id"])) for row, example in zip(exp019_rows, val_examples) if not bool(row["predicted_stays"]))

    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    alignment = validate_exp016_episode_alignment(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_features_rows, exp014_prediction_rows=val_pred_rows)
    exp014_rows = build_stage_d_trajectories(stage_b_val, v0_val, val_features_rows, val_pred_rows)
    exp019_rows_trajectory = build_stage_d_trajectories(stage_b_val, v0_val, val_features_rows, exp019_rows)
    oracle_rows, _, _ = build_executed_candidate_oracle_gate_trajectories(stage_b_rows=stage_b_val, v0_prediction_rows=v0_val, cache_rows=val_features_rows, exp014_prediction_rows=val_pred_rows)
    categories = [name for name, _ in sorted(json.loads(label_mapping.read_text(encoding="utf-8")).items(), key=lambda item: int(item[1]))]
    summaries = {"EXP014": summarize_trajectory_rows(exp014_rows, categories), "EXP019": summarize_trajectory_rows(exp019_rows_trajectory, categories), "ExecutedCandidateOracleGate": summarize_trajectory_rows(oracle_rows, categories)}
    table = [_metric_row("EXP014", summaries["EXP014"], "predicted utility > 0", "frozen learned c_hat"), _metric_row("EXP019", summaries["EXP019"], "learned gate p > 0.5", "frozen learned c_hat"), _metric_row("ExecutedCandidateOracleGate", summaries["ExecutedCandidateOracleGate"], "true U2(c_hat) > 0", "frozen learned c_hat")]
    metrics = {"EXP014": table[0], "EXP019": table[1], "ExecutedCandidateOracleGate": table[2]}
    references = _reference_checks(metrics)
    if not all(item["within_abs_tolerance_1e-5"] for checks in references.values() for item in checks.values()):
        raise ValueError("Frozen EXP014 or executed-candidate oracle reference mismatch")
    oracle_accuracy_gap = metrics["ExecutedCandidateOracleGate"]["accuracy"] - metrics["EXP014"]["accuracy"]
    oracle_regret_gap = metrics["EXP014"]["mean_regret"] - metrics["ExecutedCandidateOracleGate"]["mean_regret"]
    result = {
        "experiment_id": "EXP019", "experiment_name": "executed_candidate_gate", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False,
        "episode_count": len(stage_b_val), "v0_move_eligible_episode_count": len(val_examples), "train": training_summary, "val_gate_metrics_against_y_exec": val_gate_metrics, "metrics_table": table,
        "headroom_recovery": {"oracle_accuracy_gap": oracle_accuracy_gap, "exp019_accuracy_gain": metrics["EXP019"]["accuracy"] - metrics["EXP014"]["accuracy"], "executed_gate_accuracy_recovery": (metrics["EXP019"]["accuracy"] - metrics["EXP014"]["accuracy"]) / oracle_accuracy_gap if abs(oracle_accuracy_gap) > 1e-12 else None, "oracle_regret_gap": oracle_regret_gap, "exp019_regret_reduction": metrics["EXP014"]["mean_regret"] - metrics["EXP019"]["mean_regret"], "executed_gate_regret_recovery": (metrics["EXP014"]["mean_regret"] - metrics["EXP019"]["mean_regret"]) / oracle_regret_gap if abs(oracle_regret_gap) > 1e-12 else None},
        "action_change": _action_change({str(row["episode_id"]): row for row in val_pred_rows}, {str(row["episode_id"]): row for row in exp019_rows}, val_examples), "candidate_identity_mismatch_count": int(candidate_identity_mismatch), "episode_alignment": alignment,
        "provenance": {"source_commit": _git_commit(), "stage_d_feature_summary": str(summary_path.resolve()), "stage_d_feature_summary_sha256": file_sha256(summary_path), "stage_d_train_features": str(Path(summary["feature_files"]["train"]).resolve()), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features": str(Path(summary["feature_files"]["val"]).resolve()), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "stage_d_feature_stats": str((cache_root / "stage_d_feature_stats.json").resolve()), "stage_d_feature_stats_sha256": file_sha256(cache_root / "stage_d_feature_stats.json"), "train_predictions": str(train_predictions.resolve()), "train_predictions_sha256": file_sha256(train_predictions), "val_predictions": str(val_predictions.resolve()), "val_predictions_sha256": file_sha256(val_predictions), "v0_val_predictions": str(v0_predictions.resolve()), "v0_val_predictions_sha256": file_sha256(v0_predictions), "stage_b_val_utility": str((stage_b_root / "utility_labels" / "val.jsonl").resolve()), "stage_b_val_utility_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint)},
        "validity": {"candidate_ranking_frozen": True, "first_step_protocol_frozen": True, "true_u2_used_only_as_train_target_and_offline_oracle": True, "true_u2_used_as_model_input": False, "val_used_for_selection": False, "fixed_threshold_probability": 0.5, "final_epoch_used": True, "exp014_retrained": False, "test_split_accepted": False},
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
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(cache_root=args.cache_root, stage_b_root=args.stage_b_root, train_predictions=args.train_predictions, val_predictions=args.val_predictions, v0_predictions=args.v0_predictions, label_mapping=args.label_mapping, output=args.output, runtime_dir=args.runtime_dir, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device_name=args.device)
    print(json.dumps({"experiment_id": "EXP019", "split": "val", "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"], "val_gate_metrics_against_y_exec": result["val_gate_metrics_against_y_exec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
