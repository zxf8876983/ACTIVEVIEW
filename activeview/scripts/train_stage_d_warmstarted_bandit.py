#!/usr/bin/env python3
"""Train and evaluate EXP023's supervised-warm-start contextual bandit."""

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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_metrics import regression_metrics
from activeview.active_view.stage_d_contextual_bandit import (
    ContextualBanditRanker,
    action_name,
    action_probabilities,
    expected_reward_loss_with_entropy,
    select_bandit_actions,
    supervised_candidate_utility_loss,
)
from activeview.active_view.stage_d_dataset import StageDDataset, collate_stage_d, load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.utility_label_builder import file_sha256
from activeview.scripts.train_stage_d_contextual_bandit import (
    _action_quality,
    _assert_split,
    _categories,
    _metric_row,
    _offline_oracle_actions,
)


PHASE_A_EPOCHS = 20
PHASE_B_EPOCHS = 10
BATCH_SIZE = 256
PHASE_A_LR = 1e-3
PHASE_B_LR = 1e-4
ENTROPY_BETA = 0.001


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _loader(dataset: StageDDataset, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_stage_d,
        num_workers=0,
    )


def _phase_a(
    model: ContextualBanditRanker,
    dataset: StageDDataset,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[float], float, float]:
    """Warm-start q2/q3 with masked SmoothL1 utility regression."""
    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE_A_LR)
    history: list[float] = []
    model.train()
    for _epoch in range(PHASE_A_EPOCHS):
        total_loss = 0.0
        total_count = 0
        for batch in _loader(dataset, seed + _epoch):
            scores = model(
                batch["s0_feature"].to(device),
                batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device),
                batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            )
            loss = supervised_candidate_utility_loss(
                scores,
                batch["utility_targets"].to(device),
                batch["candidate_mask"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["s0_feature"].size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        history.append(total_loss / max(total_count, 1))
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_stage_d, num_workers=0):
            scores = model(
                batch["s0_feature"].to(device),
                batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device),
                batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            ).cpu().numpy()
            valid = batch["candidate_mask"].numpy()
            predictions.extend(float(value) for value in scores[valid])
            targets.extend(float(value) for value in batch["utility_targets"].numpy()[valid])
    phase_a_metrics = regression_metrics(predictions, targets)
    return history, float(history[-1]), float(phase_a_metrics["mae"])


def _phase_b(
    model: ContextualBanditRanker,
    dataset: StageDDataset,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    """Fine-tune expected Train utility with fixed entropy bonus."""
    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE_B_LR)
    history: list[dict[str, Any]] = []
    for epoch in range(PHASE_B_EPOCHS):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        total_count = 0
        stay_probability = 0.0
        p2_probability = 0.0
        p3_probability = 0.0
        for batch in _loader(dataset, seed + 1000 + epoch):
            s0 = batch["s0_feature"].to(device)
            s1 = batch["s1_feature"].to(device)
            delta = batch["delta_semantic"].to(device)
            geometry = batch["candidate_geometry"].to(device)
            mask = batch["candidate_mask"].to(device)
            scores = model(s0, s1, delta, geometry, mask)
            loss, expected, entropy = expected_reward_loss_with_entropy(
                scores,
                batch["utility_targets"].to(device),
                mask,
                beta=ENTROPY_BETA,
            )
            probabilities = action_probabilities(scores, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["s0_feature"].size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_reward += float(expected.detach().mean().cpu()) * count
            total_entropy += float(entropy.detach().mean().cpu()) * count
            stay_probability += float(probabilities[:, 0].detach().mean().cpu()) * count
            p2_probability += float(probabilities[:, 1].detach().mean().cpu()) * count
            p3_probability += float(probabilities[:, 2].detach().mean().cpu()) * count
            total_count += count
        means = {
            "mean_loss": total_loss / max(total_count, 1),
            "mean_expected_reward": total_reward / max(total_count, 1),
            "mean_entropy": total_entropy / max(total_count, 1),
            "mean_stay_probability": stay_probability / max(total_count, 1),
            "mean_p2_probability": p2_probability / max(total_count, 1),
            "mean_p3_probability": p3_probability / max(total_count, 1),
        }
        means["collapse_indicator"] = bool(means["mean_stay_probability"] > 0.99)
        means["epoch"] = epoch + 1
        history.append(means)
    return history, float(history[-1]["mean_loss"])


def _prediction_rows(dataset: StageDDataset, model: ContextualBanditRanker, *, device: torch.device) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_stage_d, num_workers=0)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            scores = model(
                batch["s0_feature"].to(device),
                batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device),
                batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            ).cpu().numpy()
            valid = batch["candidate_mask"].numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                count = int(valid[index].sum())
                ids = [int(value) for value in batch["candidate_ids"][index][:count]]
                values = [float(value) for value in scores[index, :count]]
                stays, selected_id, _ = select_bandit_actions(values, ids)
                rows.append(
                    {
                        "episode_id": str(episode_id),
                        "policy_split": str(batch["policy_split"][index]).lower(),
                        "remaining_candidate_ids": ids,
                        "predicted_utilities": values,
                        "predicted_stays": bool(stays),
                        "predicted_candidate_viewpoint_id": None if stays else int(selected_id),
                        "bandit_action": action_name(stays, selected_id, ids),
                    }
                )
    return rows


def _load_result_metric(path: Path, variant: str) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next((dict(row) for row in payload.get("metrics_table", []) if row.get("variant") == variant), None)


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    v0_predictions: Path,
    exp021_result: Path,
    label_mapping: Path,
    output: Path,
    runtime_dir: Path,
    seed: int = 42,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Run fixed Phase-A/Phase-B Train and exactly one Val evaluation."""
    _seed_everything(seed)
    device = _device(device_name)
    if not exp021_result.is_file():
        raise FileNotFoundError(f"Frozen EXP021 result not found: {exp021_result}")
    summary_path = cache_root / "stage_d_feature_summary.json"
    stats_path = cache_root / "stage_d_feature_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(stats_path)
    train_set = StageDDataset(Path(summary["feature_files"]["train"]), stats)
    val_set = StageDDataset(Path(summary["feature_files"]["val"]), stats)
    _assert_split(train_set.rows, "train", "Stage D Train features")
    _assert_split(val_set.rows, "val", "Stage D Val features")
    model = ContextualBanditRanker().to(device)
    phase_a_history, phase_a_loss, phase_a_mae = _phase_a(model, train_set, seed=seed, device=device)
    phase_b_history, phase_b_loss = _phase_b(model, train_set, seed=seed, device=device)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "warmstarted_contextual_bandit_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": "EXP023",
            "model_state_dict": model.state_dict(),
            "model_config": {"stay_score": 0.0, "action_set": ["Stay", "p2", "p3"]},
            "training": {"seed": seed, "phase_a_epochs": PHASE_A_EPOCHS, "phase_b_epochs": PHASE_B_EPOCHS, "batch_size": BATCH_SIZE, "phase_a_learning_rate": PHASE_A_LR, "phase_b_learning_rate": PHASE_B_LR, "entropy_beta": ENTROPY_BETA},
        },
        checkpoint,
    )
    training_summary = {
        "experiment_id": "EXP023",
        "split": "train",
        "episode_count": len(train_set),
        "phase_a": {"epochs": PHASE_A_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": PHASE_A_LR, "loss": "SmoothL1Loss", "loss_history": phase_a_history, "final_loss": phase_a_loss, "utility_mae": phase_a_mae},
        "phase_b": {"epochs": PHASE_B_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": PHASE_B_LR, "entropy_beta": ENTROPY_BETA, "entropy_used": True, "loss": "negative_expected_reward_minus_entropy_bonus", "history": phase_b_history, "final_loss": phase_b_loss, "final_mean_expected_reward": phase_b_history[-1]["mean_expected_reward"], "final_mean_entropy": phase_b_history[-1]["mean_entropy"], "final_action_probabilities": {"Stay": phase_b_history[-1]["mean_stay_probability"], "p2": phase_b_history[-1]["mean_p2_probability"], "p3": phase_b_history[-1]["mean_p3_probability"]}, "final_collapse_indicator": phase_b_history[-1]["collapse_indicator"]},
        "val_used_for_selection": False,
        "test_used": False,
    }
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    val_policy_rows = _prediction_rows(val_set, model, device=device)
    (runtime_dir / "val_predictions.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in val_policy_rows), encoding="utf-8")
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    if len(stage_b_val) != len(v0_val) or len(val_policy_rows) != len(val_set):
        raise ValueError("EXP023 canonical Val episode counts are not aligned")
    exp023_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_set.rows, val_policy_rows)
    oracle_trajectories = build_fixed_first_oracle(stage_b_val, v0_val, val_set.rows)
    categories = _categories(label_mapping)
    exp023_summary = summarize_trajectory_rows(exp023_trajectories, categories)
    oracle_summary = summarize_trajectory_rows(oracle_trajectories, categories)
    exp021_metric = _load_result_metric(exp021_result, "EXP021")
    exp014_metric = _load_result_metric(exp021_result, "EXP014")
    if exp021_metric is None or exp014_metric is None:
        raise ValueError("Frozen EXP021 result must contain EXP014 and EXP021 metrics")
    exp023_metric = _metric_row("EXP023", exp023_summary)
    exp023_metric.update({"gate": "argmax([0,q2,q3])", "candidate": "model-selected p2/p3"})
    oracle_metric = _metric_row("Fixed-first Second-Step Oracle", oracle_summary)
    oracle_metric.update({"gate": "argmax([0,true_U2(p2),true_U2(p3)])", "candidate": "oracle"})
    table = [exp014_metric, exp021_metric, exp023_metric, oracle_metric]
    metrics = {str(row["variant"]): row for row in table}
    accuracy_oracle_gap = metrics["Fixed-first Second-Step Oracle"]["accuracy"] - metrics["EXP014"]["accuracy"]
    regret_oracle_gap = metrics["EXP014"]["mean_regret"] - metrics["Fixed-first Second-Step Oracle"]["mean_regret"]
    action_quality = _action_quality(stage_b_rows=stage_b_val, v0_rows=v0_val, cache_rows=val_set.rows, policy_rows=val_policy_rows)
    result = {
        "experiment_id": "EXP023",
        "experiment_name": "supervised_warmstarted_contextual_bandit",
        "status": "COMPLETED",
        "decision": "ACCEPT",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "first_step_protocol_frozen": True,
        "train": training_summary,
        "val_episode_count": len(stage_b_val),
        "v0_move_episode_count": len(val_set),
        "metrics_table": table,
        "action_quality": action_quality,
        "offline_oracle_action_diagnostics": {"train": _offline_oracle_actions(train_set.rows), "val": _offline_oracle_actions(val_set.rows)},
        "headroom_recovery": {
            "oracle_accuracy_gap": accuracy_oracle_gap,
            "accuracy_gain": metrics["EXP023"]["accuracy"] - metrics["EXP014"]["accuracy"],
            "accuracy_recovery": (metrics["EXP023"]["accuracy"] - metrics["EXP014"]["accuracy"]) / accuracy_oracle_gap if abs(accuracy_oracle_gap) > 1e-12 else None,
            "oracle_regret_gap": regret_oracle_gap,
            "regret_reduction": metrics["EXP014"]["mean_regret"] - metrics["EXP023"]["mean_regret"],
            "regret_recovery": (metrics["EXP014"]["mean_regret"] - metrics["EXP023"]["mean_regret"]) / regret_oracle_gap if abs(regret_oracle_gap) > 1e-12 else None,
        },
        "provenance": {
            "source_commit": _git_commit(),
            "stage_d_feature_summary": str(summary_path.resolve()),
            "stage_d_feature_summary_sha256": file_sha256(summary_path),
            "stage_d_train_features": str(Path(summary["feature_files"]["train"]).resolve()),
            "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])),
            "stage_d_val_features": str(Path(summary["feature_files"]["val"]).resolve()),
            "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])),
            "stage_d_feature_stats": str((cache_root / "stage_d_feature_stats.json").resolve()),
            "stage_d_feature_stats_sha256": file_sha256(cache_root / "stage_d_feature_stats.json"),
            "stage_b_val_utility": str((stage_b_root / "utility_labels" / "val.jsonl").resolve()),
            "stage_b_val_utility_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"),
            "v0_val_predictions": str(v0_predictions.resolve()),
            "v0_val_predictions_sha256": file_sha256(v0_predictions),
            "exp021_result": str(exp021_result.resolve()),
            "exp021_result_sha256": file_sha256(exp021_result),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "validity": {
            "phase_a_train_only": True,
            "phase_b_train_only": True,
            "true_u2_used_only_as_train_reward_or_target": True,
            "true_u2_used_as_model_input": False,
            "val_used_for_selection": False,
            "stay_score_fixed_zero": True,
            "final_phase_b_epoch_used": True,
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
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--exp021-result", type=Path, required=True)
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
        v0_predictions=args.v0_predictions,
        exp021_result=args.exp021_result,
        label_mapping=args.label_mapping,
        output=args.output,
        runtime_dir=args.runtime_dir,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps({"experiment_id": "EXP023", "split": "val", "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"], "action_quality": result["action_quality"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
