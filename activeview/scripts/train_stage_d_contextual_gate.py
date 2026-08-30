#!/usr/bin/env python3
"""Train and evaluate EXP020's frozen-EXP014 contextual-latent gate."""

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

from activeview.active_view.stage_d_contextual_gate import (
    CONTEXTUAL_GATE_INPUT_DIM,
    ContextualExecutedGateMLP,
    apply_contextual_gate_decision,
    build_contextual_gate_rows,
    contextual_candidate_tokens,
    freeze_exp014_ranker,
)
from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_error_decomposition import validate_exp016_episode_alignment
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_executed_gate import build_executed_candidate_oracle_gate_trajectories
from activeview.active_view.stage_d_gate_calibration import binary_average_precision, binary_roc_auc, gate_metrics
from activeview.active_view.stage_d_policy import SequentialObservationRanker
from activeview.active_view.utility_label_builder import file_sha256


EXP020_EPOCHS = 30
EXP020_BATCH_SIZE = 256
EXP020_LEARNING_RATE = 1e-3


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


def _load_ranker(checkpoint: Path, device: torch.device) -> SequentialObservationRanker:
    model = SequentialObservationRanker().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return freeze_exp014_ranker(model)


def _latent_features(
    model: SequentialObservationRanker,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Extract one c_hat token per row while attending to all p2/p3 tokens."""
    outputs: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        count = len(batch)
        max_candidates = max(len(row["candidate_geometry"]) for row in batch)
        s0 = torch.from_numpy(np.stack([row["s0_feature"] for row in batch])).to(device)
        s1 = torch.from_numpy(np.stack([row["s1_feature"] for row in batch])).to(device)
        delta = torch.from_numpy(np.stack([row["delta_semantic"] for row in batch])).to(device)
        geometry = torch.zeros((count, max_candidates, 11), dtype=torch.float32, device=device)
        mask = torch.zeros((count, max_candidates), dtype=torch.bool, device=device)
        selected = []
        for index, row in enumerate(batch):
            candidate_geometry = torch.from_numpy(row["candidate_geometry"])
            length = candidate_geometry.shape[0]
            geometry[index, :length] = candidate_geometry
            mask[index, :length] = torch.from_numpy(row["candidate_mask"])
            selected.append(int(row["selected_index"]))
        with torch.inference_mode():
            tokens = contextual_candidate_tokens(model, s0, s1, delta, geometry, mask)
            selected_tokens = tokens[torch.arange(count, device=device), torch.tensor(selected, device=device)]
        predicted = torch.tensor(
            [float(row["predicted_utility"]) for row in batch], dtype=torch.float32, device=device
        ).unsqueeze(1)
        outputs.append(torch.cat([selected_tokens, predicted], dim=1).cpu().numpy())
    if not outputs:
        raise ValueError("Cannot extract contextual features from empty rows")
    features = np.concatenate(outputs, axis=0).astype(np.float32)
    if features.shape != (len(rows), CONTEXTUAL_GATE_INPUT_DIM) or not np.isfinite(features).all():
        raise ValueError("Invalid EXP020 contextual feature matrix")
    return features


def _train_gate(
    model: ContextualExecutedGateMLP,
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
        batch_size=EXP020_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=EXP020_LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    history: list[float] = []
    model.train()
    for _epoch in range(EXP020_EPOCHS):
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_targets in loader:
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch_features.size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        history.append(total_loss / max(total_count, 1))
    return history, float(history[-1])


def _prediction_rows(
    examples: Sequence[Mapping[str, Any]],
    logits: Sequence[float],
    originals: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, logit in zip(examples, logits):
        decision = apply_contextual_gate_decision(int(example["candidate_id"]), float(logit))
        row = dict(originals[str(example["episode_id"])])
        row.update(
            {
                "predicted_stays": decision["predicted_stays"],
                "predicted_candidate_viewpoint_id": decision["predicted_candidate_viewpoint_id"],
                "max_predicted_utility": float(example["predicted_utility"]),
                "gate_logit": decision["gate_logit"],
                "gate_probability": decision["gate_probability"],
                "gate_target": int(example["target"]),
            }
        )
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _action_change(
    exp014: Mapping[str, Mapping[str, Any]],
    exp020: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {
        "stay_to_move": [],
        "move_to_stay": [],
        "both_stay": [],
        "both_move": [],
    }
    for example in examples:
        episode_id = str(example["episode_id"])
        old_move = not bool(exp014[episode_id]["predicted_stays"])
        new_move = not bool(exp020[episode_id]["predicted_stays"])
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


def _reference(actual: Mapping[str, Any], expected: Mapping[str, float]) -> dict[str, Any]:
    return {
        key: {
            "actual": float(actual[key]),
            "reference": float(value),
            "within_abs_tolerance_1e-5": abs(float(actual[key]) - float(value)) <= 1e-5,
        }
        for key, value in expected.items()
    }


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    exp014_checkpoint: Path,
    train_predictions: Path,
    val_predictions: Path,
    v0_predictions: Path,
    exp019_result: Path,
    label_mapping: Path,
    output: Path,
    runtime_dir: Path,
    seed: int = 42,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Train EXP020 on Train and evaluate its fixed 0.5 gate once on Val."""
    _seed_everything(seed)
    device = _device(device_name)
    if not exp014_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen EXP014 checkpoint not found: {exp014_checkpoint}")
    if not exp019_result.is_file():
        raise FileNotFoundError(f"Frozen EXP019 result not found: {exp019_result}")

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
    train_examples = build_contextual_gate_rows(
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
    train_x = _latent_features(frozen_ranker, train_examples, device, EXP020_BATCH_SIZE)
    train_y = np.asarray([int(row["target"]) for row in train_examples], dtype=np.float32)
    gate = ContextualExecutedGateMLP().to(device)
    history, final_loss = _train_gate(gate, train_x, train_y, seed=seed, device=device)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "contextual_executed_gate_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": "EXP020",
            "model_state_dict": gate.state_dict(),
            "model_config": {"input_dim": CONTEXTUAL_GATE_INPUT_DIM, "hidden_dim": 64, "activation": "GELU"},
            "training": {"seed": seed, "epochs": EXP020_EPOCHS, "batch_size": EXP020_BATCH_SIZE, "learning_rate": EXP020_LEARNING_RATE},
            "exp014_frozen": True,
            "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint),
        },
        checkpoint,
    )
    training_summary = {
        "experiment_id": "EXP020",
        "split": "train",
        "episode_count": len(train_examples),
        "positive_count": int(train_y.sum()),
        "negative_count": int(train_y.size - train_y.sum()),
        "final_loss": final_loss,
        "loss_history": history,
        "epochs": EXP020_EPOCHS,
        "batch_size": EXP020_BATCH_SIZE,
        "learning_rate": EXP020_LEARNING_RATE,
        "selection": "final_epoch_fixed_30",
        "val_used_for_selection": False,
        "test_used": False,
        "exp014_checkpoint": str(exp014_checkpoint.resolve()),
        "exp014_checkpoint_sha256": file_sha256(exp014_checkpoint),
    }
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    # Val is materialized only after the Train-only gate has been fully fixed.
    val_examples = build_contextual_gate_rows(
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
    val_x = _latent_features(frozen_ranker, val_examples, device, EXP020_BATCH_SIZE)
    gate.eval()
    with torch.inference_mode():
        val_logits = gate(torch.from_numpy(val_x).to(device)).cpu().numpy().astype(np.float64)
    val_y = [int(row["target"]) for row in val_examples]
    val_gate_metrics = gate_metrics(val_logits, val_y, 0.0)
    val_gate_metrics.update(
        {
            "positive_prevalence": float(np.mean(val_y)),
            "roc_auc": binary_roc_auc(val_logits, val_y),
            "pr_auc": binary_average_precision(val_logits, val_y),
            "move_rate": float(np.mean(val_logits > 0.0)),
        }
    )
    val_originals = {str(row["episode_id"]): row for row in val_pred_rows}
    frozen_candidate_mismatch = sum(
        not bool(val_originals[str(example["episode_id"])]
                 ["predicted_stays"])
        and int(val_originals[str(example["episode_id"])]
                ["predicted_candidate_viewpoint_id"])
        != int(example["candidate_id"])
        for example in val_examples
    )
    if frozen_candidate_mismatch:
        raise ValueError(
            "Frozen EXP014 candidate identity mismatch: "
            f"{frozen_candidate_mismatch}"
        )
    exp020_prediction_rows = _prediction_rows(val_examples, val_logits, val_originals)
    _write_jsonl(runtime_dir / "val_gate_predictions.jsonl", exp020_prediction_rows)

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
    exp020_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_feature_rows, exp020_prediction_rows)
    oracle_trajectories, _, _ = build_executed_candidate_oracle_gate_trajectories(
        stage_b_rows=stage_b_val,
        v0_prediction_rows=v0_val,
        cache_rows=val_feature_rows,
        exp014_prediction_rows=val_pred_rows,
    )
    categories = _categories(label_mapping)
    exp014_summary = summarize_trajectory_rows(exp014_trajectories, categories)
    exp020_summary = summarize_trajectory_rows(exp020_trajectories, categories)
    oracle_summary = summarize_trajectory_rows(oracle_trajectories, categories)
    frozen_exp019 = json.loads(exp019_result.read_text(encoding="utf-8"))
    frozen_table = {str(row["variant"]): row for row in frozen_exp019["metrics_table"]}
    table = [
        _metric_row("EXP014", exp014_summary, "predicted utility > 0", "frozen learned c_hat"),
        dict(frozen_table["EXP019"]),
        _metric_row("EXP020", exp020_summary, "contextual gate p > 0.5", "frozen learned c_hat"),
        _metric_row("ExecutedCandidateOracle", oracle_summary, "true U2(c_hat) > 0", "frozen learned c_hat"),
    ]
    metrics = {str(row["variant"]): row for row in table}
    if abs(metrics["EXP014"]["accuracy"] - 0.6582540930864375) > 1e-5:
        raise ValueError("Frozen EXP014 trajectory reference mismatch")
    if abs(metrics["ExecutedCandidateOracle"]["accuracy"] - 0.7431186101379853) > 1e-5:
        raise ValueError("Frozen executed-candidate oracle reference mismatch")
    candidate_identity_mismatch = frozen_candidate_mismatch
    if candidate_identity_mismatch:
        raise ValueError(f"EXP020 candidate identity mismatch: {candidate_identity_mismatch}")

    accuracy_oracle_gap = metrics["ExecutedCandidateOracle"]["accuracy"] - metrics["EXP014"]["accuracy"]
    regret_oracle_gap = metrics["EXP014"]["mean_regret"] - metrics["ExecutedCandidateOracle"]["mean_regret"]
    result = {
        "experiment_id": "EXP020",
        "experiment_name": "contextual_latent_executed_gate",
        "status": "COMPLETED",
        "decision": "INCONCLUSIVE",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "exp014_frozen": True,
        "episode_count": len(stage_b_val),
        "v0_move_eligible_episode_count": len(val_examples),
        "train": training_summary,
        "model": {
            "input_dim": CONTEXTUAL_GATE_INPUT_DIM,
            "hidden_dim": 64,
            "activation": "GELU",
            "frozen_contextual_token_dim": 128,
            "token_extracted_before": "EXP014.utility_head",
            "parameters_trained": "EXP020 gate only",
        },
        "val_gate_metrics_against_y_exec": val_gate_metrics,
        "metrics_table": table,
        "headroom_recovery": {
            "accuracy_gain": metrics["EXP020"]["accuracy"] - metrics["EXP014"]["accuracy"],
            "accuracy_oracle_gap": accuracy_oracle_gap,
            "accuracy_recovery": (metrics["EXP020"]["accuracy"] - metrics["EXP014"]["accuracy"]) / accuracy_oracle_gap if abs(accuracy_oracle_gap) > 1e-12 else None,
            "regret_reduction": metrics["EXP014"]["mean_regret"] - metrics["EXP020"]["mean_regret"],
            "regret_oracle_gap": regret_oracle_gap,
            "regret_recovery": (metrics["EXP014"]["mean_regret"] - metrics["EXP020"]["mean_regret"]) / regret_oracle_gap if abs(regret_oracle_gap) > 1e-12 else None,
        },
        "action_change": _action_change(
            {str(row["episode_id"]): row for row in val_pred_rows},
            {str(row["episode_id"]): row for row in exp020_prediction_rows},
            val_examples,
        ),
        "candidate_identity_mismatch_count": int(candidate_identity_mismatch),
        "frozen_exp014_candidate_identity_mismatch_count": int(frozen_candidate_mismatch),
        "episode_alignment": alignment,
        "representation_comparison": {
            "EXP019": {"representation": "normalized 11-D candidate geometry + predicted utility", "roc_auc": float(frozen_exp019["val_gate_metrics_against_y_exec"]["roc_auc"]), "pr_auc": float(frozen_exp019["val_gate_metrics_against_y_exec"]["pr_auc"]), "trajectory_accuracy": float(metrics["EXP019"]["accuracy"])},
            "EXP020": {"representation": "frozen EXP014 contextual token + predicted utility", "roc_auc": float(val_gate_metrics["roc_auc"]), "pr_auc": float(val_gate_metrics["pr_auc"]), "trajectory_accuracy": float(metrics["EXP020"]["accuracy"])},
        },
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
            "fixed_threshold_probability": 0.5,
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
        label_mapping=args.label_mapping,
        output=args.output,
        runtime_dir=args.runtime_dir,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps({"experiment_id": "EXP020", "split": "val", "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"], "val_gate_metrics_against_y_exec": result["val_gate_metrics_against_y_exec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
