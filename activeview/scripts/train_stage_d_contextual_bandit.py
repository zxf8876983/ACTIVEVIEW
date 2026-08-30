#!/usr/bin/env python3
"""Train and evaluate the EXP021 offline full-information contextual bandit."""

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

from activeview.active_view.stage_d_contextual_bandit import (
    ContextualBanditRanker,
    action_name,
    expected_reward_loss,
    select_bandit_actions,
)
from activeview.active_view.stage_d_dataset import (
    StageDDataset,
    collate_stage_d,
    load_jsonl,
    load_stage_d_statistics,
)
from activeview.active_view.stage_d_evaluation import (
    build_fixed_first_oracle,
    build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.active_view.utility_label_builder import file_sha256


EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


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


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _train(
    model: ContextualBanditRanker,
    dataset: StageDDataset,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[float], list[float], float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate_stage_d,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[float] = []
    reward_history: list[float] = []
    model.train()
    for _epoch in range(EPOCHS):
        total_loss = 0.0
        total_reward = 0.0
        total_count = 0
        for batch in loader:
            scores = model(
                batch["s0_feature"].to(device),
                batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device),
                batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            )
            loss, expected = expected_reward_loss(
                scores,
                batch["utility_targets"].to(device),
                batch["candidate_mask"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["s0_feature"].size(0))
            total_loss += float(loss.detach().cpu()) * count
            total_reward += float(expected.detach().mean().cpu()) * count
            total_count += count
        history.append(total_loss / max(total_count, 1))
        reward_history.append(total_reward / max(total_count, 1))
    return history, reward_history, float(history[-1])


def _prediction_rows(
    dataset: StageDDataset,
    model: ContextualBanditRanker,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
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
            mask = batch["candidate_mask"].numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                count = int(mask[index].sum())
                ids = [int(value) for value in batch["candidate_ids"][index][:count]]
                values = [float(value) for value in scores[index, :count]]
                stays, selected_id, _score = select_bandit_actions(values, ids)
                rows.append(
                    {
                        "episode_id": str(episode_id),
                        "policy_split": "val" if str(batch["policy_split"][index]).lower() == "val" else "train",
                        "remaining_candidate_ids": ids,
                        "predicted_utilities": values,
                        "predicted_stays": bool(stays),
                        "predicted_candidate_viewpoint_id": None if stays else int(selected_id),
                        "bandit_action": action_name(stays, selected_id, ids),
                    }
                )
    return rows


def _action_quality(
    *,
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    v0 = {str(row["episode_id"]): row for row in v0_rows}
    cache = {str(row["episode_id"]): row for row in cache_rows}
    policy = {str(row["episode_id"]): row for row in policy_rows}
    oracle_rows = build_fixed_first_oracle(stage_b_rows, v0_rows, cache_rows)
    oracle = {str(row["episode_id"]): row for row in oracle_rows}
    action_counts = {"Stay": 0, "p2": 0, "p3": 0}
    exact = 0
    move_match = 0
    both_move = 0
    candidate_hit = 0
    selected_utilities: list[float] = []
    for episode_id, prediction in policy.items():
        if bool(v0[episode_id]["predicted_stays"]):
            continue
        cached = cache[episode_id]
        ids = [int(value) for value in cached["remaining_candidate_ids"]]
        stays = bool(prediction["predicted_stays"])
        selected_id = None if stays else int(prediction["predicted_candidate_viewpoint_id"])
        model_action = action_name(stays, selected_id, ids)
        action_counts[model_action] += 1
        oracle_row = oracle[episode_id]
        if int(oracle_row["moves"]) == 1:
            oracle_action = "Stay"
        else:
            oracle_id = int(oracle_row["selected_viewpoint_id"])
            oracle_action = "p2" if oracle_id == ids[0] else "p3"
        exact += int(model_action == oracle_action)
        model_move = model_action != "Stay"
        oracle_move = oracle_action != "Stay"
        move_match += int(model_move == oracle_move)
        if model_move and oracle_move:
            both_move += 1
            candidate_hit += int(selected_id == int(oracle_row["selected_viewpoint_id"]))
        if model_move:
            selected_utilities.append(float(cached["second_step_utility_targets"][ids.index(selected_id)]))
        else:
            selected_utilities.append(0.0)
    denominator = len(policy)
    return {
        "episode_count": denominator,
        "action_counts": action_counts,
        "stay_rate": action_counts["Stay"] / denominator if denominator else 0.0,
        "p2_rate": action_counts["p2"] / denominator if denominator else 0.0,
        "p3_rate": action_counts["p3"] / denominator if denominator else 0.0,
        "exact_action_match_count": exact,
        "exact_action_match_rate": exact / denominator if denominator else 0.0,
        "binary_move_stay_match_count": move_match,
        "binary_move_stay_match_rate": move_match / denominator if denominator else 0.0,
        "candidate_exact_hit_both_move_count": candidate_hit,
        "candidate_exact_hit_both_move_rate": candidate_hit / both_move if both_move else 0.0,
        "candidate_exact_hit_both_move_denominator": both_move,
        "selected_action_mean_true_utility": float(np.mean(selected_utilities)) if selected_utilities else 0.0,
    }


def _metric_row(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": name,
        "accuracy": float(summary["recognition"]["accuracy"]),
        "macro_f1": float(summary["recognition"]["macro_f1"]),
        "mean_regret": float(summary["decision_regret"]["mean"]),
        "median_regret": float(summary["decision_regret"]["median"]),
        "p90_regret": float(summary["decision_regret"]["p90"]),
        "headroom_capture": float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]),
        "average_moves": float(summary["movement"]["average_moves"]),
        "mean_geodesic_cost_m": float(summary["movement"]["trajectory_geodesic_cost_m"]["mean"]),
    }


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
    v0_predictions: Path,
    exp019_result: Path,
    label_mapping: Path,
    output: Path,
    runtime_dir: Path,
    device_name: str = "cuda:0",
    seed: int = 42,
    exp020_result: Path | None = None,
) -> dict[str, Any]:
    """Run fixed Train contextual-bandit optimization and one Val evaluation."""
    _seed_everything(seed)
    device = _device(device_name)
    summary_path = cache_root / "stage_d_feature_summary.json"
    stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_set = StageDDataset(Path(summary["feature_files"]["train"]), stats)
    val_set = StageDDataset(Path(summary["feature_files"]["val"]), stats)
    _assert_split(train_set.rows, "train", "Stage D Train features")
    _assert_split(val_set.rows, "val", "Stage D Val features")
    model = ContextualBanditRanker().to(device)
    train_loss_history, train_reward_history, final_loss = _train(model, train_set, seed=seed, device=device)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime_dir / "checkpoints" / "contextual_bandit_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": "EXP021",
            "model_state_dict": model.state_dict(),
            "model_config": {"stay_score": 0.0, "s0_dim": 275, "s1_dim": 275, "delta_dim": 19, "geometry_dim": 11},
            "training": {"seed": seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "entropy_used": False},
        },
        checkpoint,
    )
    training_summary = {
        "experiment_id": "EXP021",
        "split": "train",
        "episode_count": len(train_set),
        "final_loss": final_loss,
        "final_mean_expected_reward": train_reward_history[-1],
        "loss_history": train_loss_history,
        "expected_reward_history": train_reward_history,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "entropy_used": False,
        "val_used_for_selection": False,
        "test_used": False,
    }
    (runtime_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    val_policy_rows = _prediction_rows(val_set, model, device=device)
    _write_jsonl(runtime_dir / "val_predictions.jsonl", val_policy_rows)
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_val = load_jsonl(v0_predictions)
    _assert_split(stage_b_val, "val", "Stage B Val utility")
    _assert_split(v0_val, "val", "Stage C-v0 Val predictions")
    if len(stage_b_val) != len(v0_val) or len(val_policy_rows) != len(val_set):
        raise ValueError("EXP021 canonical Val episode counts are not aligned")
    exp021_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_set.rows, val_policy_rows)
    oracle_trajectories = build_fixed_first_oracle(stage_b_val, v0_val, val_set.rows)
    categories = _categories(label_mapping)
    exp021_summary = summarize_trajectory_rows(exp021_trajectories, categories)
    oracle_summary = summarize_trajectory_rows(oracle_trajectories, categories)
    exp014_metric = _load_result_metric(REPO_ROOT / "experiments/stage_d/EXP019_executed_candidate_gate/result.json", "EXP014")
    exp019_metric = _load_result_metric(exp019_result, "EXP019")
    if exp014_metric is None or exp019_metric is None:
        raise ValueError("Frozen EXP014/EXP019 result metrics are required")
    table = [exp014_metric, exp019_metric, _metric_row("EXP021", exp021_summary), _metric_row("Fixed-first Second-Step Oracle", oracle_summary)]
    if exp020_result is not None:
        exp020_metric = _load_result_metric(exp020_result, "EXP020")
        if exp020_metric is not None:
            table.insert(2, exp020_metric)
    by_variant = {str(row["variant"]): row for row in table}
    exp014 = by_variant["EXP014"]
    oracle = by_variant["Fixed-first Second-Step Oracle"]
    oracle_accuracy_gap = float(oracle["accuracy"]) - float(exp014["accuracy"])
    oracle_regret_gap = float(exp014["mean_regret"]) - float(oracle["mean_regret"])
    action_quality = _action_quality(stage_b_rows=stage_b_val, v0_rows=v0_val, cache_rows=val_set.rows, policy_rows=val_policy_rows)
    result = {
        "experiment_id": "EXP021",
        "experiment_name": "contextual_bandit_joint_second_step_policy",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "model": {"architecture": "Stage-D contextual encoders + 2-layer Transformer + action head", "stay_score": 0.0, "entropy_used": False},
        "train": training_summary,
        "val_episode_count": len(stage_b_val),
        "v0_move_episode_count": len(val_set),
        "metrics_table": table,
        "action_quality": action_quality,
        "headroom_recovery": {
            "oracle_accuracy_gap": oracle_accuracy_gap,
            "accuracy_gain": float(by_variant["EXP021"]["accuracy"]) - float(exp014["accuracy"]),
            "accuracy_recovery": (float(by_variant["EXP021"]["accuracy"]) - float(exp014["accuracy"])) / oracle_accuracy_gap if abs(oracle_accuracy_gap) > 1e-12 else None,
            "oracle_regret_gap": oracle_regret_gap,
            "regret_reduction": float(exp014["mean_regret"]) - float(by_variant["EXP021"]["mean_regret"]),
            "regret_recovery": (float(exp014["mean_regret"]) - float(by_variant["EXP021"]["mean_regret"])) / oracle_regret_gap if abs(oracle_regret_gap) > 1e-12 else None,
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
            "exp019_result": str(exp019_result.resolve()),
            "exp019_result_sha256": file_sha256(exp019_result),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "validity": {
            "first_step_protocol_frozen": True,
            "candidate_identity_is_model_selected": True,
            "true_u2_used_only_as_train_rewards": True,
            "true_u2_used_as_model_input": False,
            "policy_objective": "maximize mean Train expected reward under softmax([0,q2,q3])",
            "val_used_for_selection": False,
            "test_split_accepted": False,
        },
    }
    if exp020_result is not None and exp020_result.is_file():
        result["provenance"]["exp020_result"] = str(exp020_result.resolve())
        result["provenance"]["exp020_result_sha256"] = file_sha256(exp020_result)
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
    parser.add_argument("--exp019-result", type=Path, required=True)
    parser.add_argument("--exp020-result", type=Path)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        cache_root=args.cache_root,
        stage_b_root=args.stage_b_root,
        v0_predictions=args.v0_predictions,
        exp019_result=args.exp019_result,
        exp020_result=args.exp020_result,
        label_mapping=args.label_mapping,
        output=args.output,
        runtime_dir=args.runtime_dir,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps({"experiment_id": "EXP021", "split": "val", "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"], "action_quality": result["action_quality"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
