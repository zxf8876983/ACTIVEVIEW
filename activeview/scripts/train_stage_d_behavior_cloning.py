#!/usr/bin/env python3
"""Train and evaluate EXP027 Spatial-RGB Fixed-first Oracle behavior cloning."""

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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_behavior_cloning import SpatialRGBBehaviorCloner, oracle_action_index, select_behavior_action
from activeview.active_view.stage_d_dataset import StageDDataset, load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_error_decomposition import _index
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, observation_keys_from_feature_rows
from activeview.active_view.stage_d_rgb_spatial import SPATIAL_TOKEN_COUNT, spatial_embedding_index
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


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
        raise ValueError(f"{name} must explicitly contain only {split}: {sorted(invalid)}")


def _load_spatial_cache(path: Path, keys: Sequence[RGBObservationKey]) -> tuple[np.ndarray, dict[tuple[str, str, str, int], int]]:
    embeddings = np.load(path / "embeddings.npy", mmap_mode="r")
    rows = [json.loads(line) for line in (path / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if embeddings.shape != (len(keys), SPATIAL_TOKEN_COUNT, 768) or embeddings.dtype != np.float16:
        raise ValueError("EXP025 spatial cache schema mismatch")
    index = spatial_embedding_index(rows)
    expected = {key.tuple for key in keys}
    if set(index) != expected:
        raise ValueError("EXP025 cache keys do not exactly match visited s0/s1 observations")
    return embeddings, index


def _oracle_labels(
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Derive labels from and cross-check against the frozen trajectory oracle."""
    trajectories = build_fixed_first_oracle(stage_b_rows, v0_rows, feature_rows)
    features = _index(feature_rows, "Stage D features")
    labels: dict[str, int] = {}
    for trajectory in trajectories:
        episode_id = str(trajectory["episode_id"])
        if episode_id not in features:
            continue
        row = features[episode_id]
        ids = [int(value) for value in row["remaining_candidate_ids"]]
        if int(trajectory["moves"]) == 1:
            label = 0
        elif int(trajectory["moves"]) == 2:
            selected = int(trajectory["selected_viewpoint_id"])
            if selected not in ids:
                raise ValueError(f"Frozen oracle selected unknown candidate {selected}: {episode_id}")
            label = ids.index(selected) + 1
        else:
            raise ValueError(f"Unexpected frozen oracle move count {trajectory['moves']}: {episode_id}")
        direct = oracle_action_index(row["second_step_utility_targets"])
        if label != direct:
            raise ValueError(f"Oracle label mismatch for {episode_id}: trajectory={label}, direct={direct}")
        labels[episode_id] = label
    expected = set(features)
    if set(labels) != expected:
        raise ValueError("Frozen oracle labels do not cover the feature episode universe")
    return labels


class _BehaviorDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path, stats: Mapping[str, np.ndarray], spatial: np.ndarray, spatial_index: Mapping[tuple[str, str, str, int], int], labels: Mapping[str, int]) -> None:
        base = StageDDataset(path, stats)
        self.rows = base.rows
        self.items: list[dict[str, Any]] = []
        for index, row in enumerate(self.rows):
            item = base[index]
            common = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
            key0 = RGBObservationKey(*common, int(row["s0_viewpoint_id"]))
            key1 = RGBObservationKey(*common, int(row["s1_viewpoint_id"]))
            if key0.tuple not in spatial_index or key1.tuple not in spatial_index:
                raise ValueError(f"Missing visited RGB cache key for {row['episode_id']}")
            episode_id = str(row["episode_id"])
            item.update({"rgb_s0": spatial_index[key0.tuple], "rgb_s1": spatial_index[key1.tuple], "oracle_action": int(labels[episode_id])})
            self.items.append(item)
        self.spatial = spatial

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "s0_feature": item["s0_feature"], "s1_feature": item["s1_feature"], "delta_semantic": item["delta_semantic"],
            "candidate_geometry": item["candidate_geometry"], "candidate_ids": item["candidate_ids"],
            "rgb_s0": torch.from_numpy(np.asarray(self.spatial[item["rgb_s0"]], dtype=np.float32)),
            "rgb_s1": torch.from_numpy(np.asarray(self.spatial[item["rgb_s1"]], dtype=np.float32)),
            "oracle_action": int(item["oracle_action"]), "episode_id": item["episode_id"],
        }


def _collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    # Keep a fixed p2/p3 output contract even when a batch contains only one candidate.
    max_candidates = max(2, max(len(item["candidate_ids"]) for item in batch))
    geometry = torch.zeros((size, max_candidates, 11), dtype=torch.float32)
    mask = torch.zeros((size, max_candidates), dtype=torch.bool)
    ids: list[list[int]] = []
    for index, item in enumerate(batch):
        count = len(item["candidate_ids"])
        geometry[index, :count] = item["candidate_geometry"]
        mask[index, :count] = True
        ids.append(list(item["candidate_ids"]))
    return {
        "s0_feature": torch.stack([item["s0_feature"] for item in batch]), "s1_feature": torch.stack([item["s1_feature"] for item in batch]),
        "delta_semantic": torch.stack([item["delta_semantic"] for item in batch]), "candidate_geometry": geometry, "candidate_mask": mask,
        "candidate_ids": ids, "rgb_s0": torch.stack([item["rgb_s0"] for item in batch]), "rgb_s1": torch.stack([item["rgb_s1"] for item in batch]),
        "oracle_action": torch.tensor([int(item["oracle_action"]) for item in batch], dtype=torch.long), "episode_id": [str(item["episode_id"]) for item in batch],
    }


def _train(model: SpatialRGBBehaviorCloner, dataset: _BehaviorDataset, seed: int, device: torch.device) -> tuple[list[float], float]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=_collate, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        count = 0
        for batch in loader:
            logits = model(batch["s0_feature"].to(device), batch["s1_feature"].to(device), batch["delta_semantic"].to(device), batch["candidate_geometry"].to(device), batch["candidate_mask"].to(device), batch["rgb_s0"].to(device), batch["rgb_s1"].to(device))
            loss = criterion(logits, batch["oracle_action"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(batch["oracle_action"].shape[0]); total += float(loss.detach().cpu()) * n; count += n
        history.append(total / max(count, 1))
    return history, float(history[-1])


def _predict(model: SpatialRGBBehaviorCloner, dataset: _BehaviorDataset, device: torch.device) -> tuple[list[dict[str, Any]], np.ndarray]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, num_workers=0)
    rows: list[dict[str, Any]] = []; predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["s0_feature"].to(device), batch["s1_feature"].to(device), batch["delta_semantic"].to(device), batch["candidate_geometry"].to(device), batch["candidate_mask"].to(device), batch["rgb_s0"].to(device), batch["rgb_s1"].to(device)).cpu().numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                count = len(batch["candidate_ids"][index]); action, selected_id = select_behavior_action(logits[index], batch["candidate_ids"][index])
                predictions.append(action)
                rows.append({"episode_id": str(episode_id), "policy_split": str(batch["candidate_ids"][index] and dataset.items[len(rows)]["policy_split"]), "remaining_candidate_ids": batch["candidate_ids"][index], "predicted_stays": action == 0, "predicted_candidate_viewpoint_id": selected_id, "behavior_action": ["Stay", "p2", "p3"][action], "logits": [float(value) for value in logits[index]], "candidate_count": count})
    return rows, np.asarray(predictions, dtype=np.int64)


def _metric_row(name: str, summary: Mapping[str, Any], gate: str, candidate: str) -> dict[str, Any]:
    return {"variant": name, "gate": gate, "candidate": candidate, "accuracy": float(summary["recognition"]["accuracy"]), "macro_f1": float(summary["recognition"]["macro_f1"]), "mean_regret": float(summary["decision_regret"]["mean"]), "median_regret": float(summary["decision_regret"]["median"]), "p90_regret": float(summary["decision_regret"]["p90"]), "headroom_capture": float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]), "average_moves": float(summary["movement"]["average_moves"]), "mean_geodesic_cost_m": float(summary["movement"]["trajectory_geodesic_cost_m"]["mean"])}


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _load_metric(path: Path, variant: str) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("metrics_table", [])
    value = next((dict(row) for row in rows if row.get("variant") == variant), None)
    if value is None:
        raise ValueError(f"Missing {variant} metric in {path}")
    return value


def _action_stats(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for true, pred in zip(labels, predictions):
        confusion[int(true), int(pred)] += 1
    true_counts = np.bincount(labels, minlength=3); pred_counts = np.bincount(predictions, minlength=3)
    move_true = labels > 0; move_pred = predictions > 0
    tp = int(np.sum(move_true & move_pred)); fp = int(np.sum(~move_true & move_pred)); fn = int(np.sum(move_true & ~move_pred)); tn = int(np.sum(~move_true & ~move_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0; stay_precision = tn / (tn + fn) if tn + fn else 0.0; stay_recall = tn / (tn + fp) if tn + fp else 0.0
    return {"episode_count": int(labels.size), "oracle_distribution": {name: {"count": int(count), "rate": float(count / labels.size)} for name, count in zip(("Stay", "p2", "p3"), true_counts)}, "learned_distribution": {name: {"count": int(count), "rate": float(count / labels.size)} for name, count in zip(("Stay", "p2", "p3"), pred_counts)}, "exact_imitation_accuracy": float(np.mean(labels == predictions)), "confusion_matrix_oracle_rows_pred_columns": confusion.tolist(), "binary_move_stay": {"accuracy": float(np.mean(move_true == move_pred)), "balanced_accuracy": 0.5 * (recall + stay_recall), "move_precision": precision, "move_recall": recall, "move_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "stay_precision": stay_precision, "stay_recall": stay_recall, "confusion": {"stay_stay": tn, "stay_move": fn, "move_stay": fp, "move_move": tp}}}


def _utility_diagnostics(rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    selected: list[float] = []; harmful: list[float] = []; missed: list[float] = []; both_move = candidate_hits = 0
    for row, pred, oracle in zip(rows, predictions, labels):
        utilities = [float(value) for value in row["second_step_utility_targets"]]
        selected_utility = 0.0 if int(pred) == 0 else utilities[int(pred) - 1]
        selected.append(selected_utility)
        if int(pred) > 0 and selected_utility <= 0.0: harmful.append(selected_utility)
        if int(pred) == 0 and int(oracle) > 0: missed.append(max(utilities))
        if int(pred) > 0 and int(oracle) > 0:
            both_move += 1; candidate_hits += int(int(pred) == int(oracle))
    return {"selected_action_mean_true_utility": float(np.mean(selected)), "harmful_move": {"count": len(harmful), "rate": float(len(harmful) / len(rows)), "mean_true_utility": float(np.mean(harmful)) if harmful else 0.0, "median_true_utility": float(np.median(harmful)) if harmful else 0.0, "summed_negative_utility": float(-sum(harmful))}, "missed_beneficial_move": {"count": len(missed), "rate": float(len(missed) / len(rows)), "mean_oracle_utility": float(np.mean(missed)) if missed else 0.0, "median_oracle_utility": float(np.median(missed)) if missed else 0.0, "summed_missed_positive_utility": float(sum(missed))}, "both_move_count": both_move, "candidate_exact_hit_count": candidate_hits, "candidate_exact_hit_rate": float(candidate_hits / both_move) if both_move else 0.0}


def analyze(*, cache_root: Path, stage_b_root: Path, v0_predictions: Path, v0_train_predictions: Path, exp014_predictions: Path, exp023_result: Path, exp025_result: Path, spatial_cache: Path, label_mapping: Path, output: Path, runtime_dir: Path, seed: int = 42, device_name: str = "cuda:0") -> dict[str, Any]:
    _seed(seed); device = _device(device_name)
    summary_path = cache_root / "stage_d_feature_summary.json"; stats_path = cache_root / "stage_d_feature_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")); stats = load_stage_d_statistics(stats_path)
    train_features = load_jsonl(Path(summary["feature_files"]["train"])); val_features = load_jsonl(Path(summary["feature_files"]["val"]))
    stage_b_train = load_jsonl(stage_b_root / "utility_labels" / "train.jsonl"); stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_train = load_jsonl(v0_train_predictions); v0_val = load_jsonl(v0_predictions)
    for rows, split, name in ((train_features, "train", "Train features"), (val_features, "val", "Val features"), (stage_b_train, "train", "Stage B Train"), (stage_b_val, "val", "Stage B Val"), (v0_train, "train", "v0 Train"), (v0_val, "val", "v0 Val")): _assert_split(rows, split, name)
    train_labels = _oracle_labels(stage_b_train, v0_train, train_features); val_labels = _oracle_labels(stage_b_val, v0_val, val_features)
    train_keys, train_episode_keys = observation_keys_from_feature_rows(train_features); val_keys, val_episode_keys = observation_keys_from_feature_rows(val_features)
    if set(train_keys) & set(val_keys): raise ValueError("Train/Val RGB observation keys overlap")
    spatial, spatial_idx = _load_spatial_cache(spatial_cache, sorted(set(train_keys) | set(val_keys), key=lambda key: key.tuple))
    train_set = _BehaviorDataset(Path(summary["feature_files"]["train"]), stats, spatial, spatial_idx, train_labels); val_set = _BehaviorDataset(Path(summary["feature_files"]["val"]), stats, spatial, spatial_idx, val_labels)
    model = SpatialRGBBehaviorCloner().to(device); history, final_loss = _train(model, train_set, seed, device)
    runtime_dir.mkdir(parents=True, exist_ok=True); checkpoint = runtime_dir / "checkpoints" / "spatial_rgb_behavior_cloner_final.pth"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": "EXP027", "model_state_dict": model.state_dict(), "model_config": {"action_set": ["Stay", "p2", "p3"], "loss": "CrossEntropyLoss", "rgb_spatial_tokens": [16, 768], "dino_frozen": True}}, checkpoint)
    val_prediction_rows, val_predictions = _predict(model, val_set, device); train_prediction_rows, train_predictions = _predict(model, train_set, device)
    for rows, split in ((train_prediction_rows, "train"), (val_prediction_rows, "val")):
        for row in rows: row["policy_split"] = split
    (runtime_dir / "train_predictions.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in train_prediction_rows), encoding="utf-8")
    (runtime_dir / "val_predictions.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in val_prediction_rows), encoding="utf-8")
    exp027_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_features, val_prediction_rows)
    exp014_rows = load_jsonl(exp014_predictions); _assert_split(exp014_rows, "val", "EXP014 second-step predictions")
    exp014_trajectories = build_stage_d_trajectories(stage_b_val, v0_val, val_features, exp014_rows)
    oracle_trajectories = build_fixed_first_oracle(stage_b_val, v0_val, val_features)
    categories = _categories(label_mapping); exp027_summary = summarize_trajectory_rows(exp027_trajectories, categories); exp014_summary = summarize_trajectory_rows(exp014_trajectories, categories); oracle_summary = summarize_trajectory_rows(oracle_trajectories, categories)
    table = [_metric_row("EXP014", exp014_summary, "frozen Stage-D learned action", "frozen learned candidate"), _load_metric(exp023_result, "EXP023"), _load_metric(exp025_result, "EXP025"), _metric_row("EXP027 BC", exp027_summary, "CrossEntropy oracle action", "model-selected p2/p3"), _metric_row("Fixed-first Second-Step Oracle", oracle_summary, "argmax([0,true_U2(p2),true_U2(p3)])", "oracle")]
    metrics = {str(row["variant"]): row for row in table}; train_action_stats = _action_stats(np.asarray([train_labels[str(row["episode_id"])] for row in train_features], dtype=np.int64), train_predictions); action_stats = _action_stats(np.asarray([val_labels[str(row["episode_id"])] for row in val_features], dtype=np.int64), val_predictions); utility = _utility_diagnostics(val_features, val_predictions, np.asarray([val_labels[str(row["episode_id"])] for row in val_features], dtype=np.int64))
    oracle_gap = metrics["Fixed-first Second-Step Oracle"]["accuracy"] - metrics["EXP014"]["accuracy"]; regret_gap = metrics["EXP014"]["mean_regret"] - metrics["Fixed-first Second-Step Oracle"]["mean_regret"]
    result: dict[str, Any] = {"experiment_id": "EXP027", "experiment_name": "spatial_rgb_oracle_behavior_cloning", "status": "COMPLETED", "decision": "INCONCLUSIVE", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "first_step_protocol_frozen": True, "train_episode_count": len(train_set), "val_episode_count": len(stage_b_val), "v0_move_episode_count": len(val_set), "train": {"episode_count": len(train_set), "oracle_distribution": train_action_stats["oracle_distribution"], "final_ce_loss": final_loss, "loss_history": history, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "class_weights": None, "val_used_for_selection": False}, "model": {"architecture": "state encoders + shared spatial RGB projector/Transformer + learnable Stay head + shared candidate head", "action_set": ["Stay", "p2", "p3"], "loss": "CrossEntropyLoss", "dino_frozen": True, "spatial_tokens": [16, 768], "true_utility_used_as_model_input": False}, "val_action_diagnostics": action_stats | utility, "metrics_table": table, "headroom_recovery": {"accuracy_oracle_gap": oracle_gap, "accuracy_gain": metrics["EXP027 BC"]["accuracy"] - metrics["EXP014"]["accuracy"], "accuracy_recovery": (metrics["EXP027 BC"]["accuracy"] - metrics["EXP014"]["accuracy"]) / oracle_gap if abs(oracle_gap) > 1e-12 else None, "regret_oracle_gap": regret_gap, "regret_reduction": metrics["EXP014"]["mean_regret"] - metrics["EXP027 BC"]["mean_regret"], "regret_recovery": (metrics["EXP014"]["mean_regret"] - metrics["EXP027 BC"]["mean_regret"]) / regret_gap if abs(regret_gap) > 1e-12 else None}, "rgb_audit": {"unique_train_rgb_observations": len(train_keys), "unique_val_rgb_observations": len(val_keys), "spatial_cache_path": str(spatial_cache.resolve()), "spatial_cache_sha256": file_sha256(spatial_cache / "summary.json"), "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_skeleton_used": False}, "provenance": {"source_commit": _git_commit(), "stage_d_feature_summary_sha256": file_sha256(summary_path), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "stage_b_train_sha256": file_sha256(stage_b_root / "utility_labels" / "train.jsonl"), "stage_b_val_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"), "v0_train_predictions_sha256": file_sha256(v0_train_predictions), "v0_val_predictions_sha256": file_sha256(v0_predictions), "exp014_val_predictions_sha256": file_sha256(exp014_predictions), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint)}, "validity": {"oracle_labels_match_frozen_fixed_first": True, "candidate_ids_remain_independent": True, "true_utility_used_as_input": False, "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_skeleton_used": False, "candidate_identity_mismatch_count": 0, "test_split_accepted": False}}
    payload = json.dumps(result, indent=2, ensure_ascii=False); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(payload, encoding="utf-8"); (runtime_dir / "result.json").write_text(payload, encoding="utf-8"); return result


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root(); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--v0-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    parser.add_argument("--v0-train-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl")
    parser.add_argument("--exp014-predictions", type=Path, default=data_root / "experiments/stage_d/EXP017_second_step_gate_calibration/runtime/val_second_step_predictions.jsonl")
    parser.add_argument("--exp023-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP023_warmstarted_contextual_bandit/result.json")
    parser.add_argument("--exp025-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP025_dinov2_spatial_rgb/result.json")
    parser.add_argument("--spatial-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4"))
    parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP027_spatial_rgb_oracle_behavior_cloning/result.json")
    parser.add_argument("--runtime-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP027_spatial_rgb_oracle_behavior_cloning"))
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", dest="device_name", default="cuda:0"); return parser


def main() -> None:
    args = build_parser().parse_args(); result = analyze(**vars(args)); print(json.dumps({"experiment_id": "EXP027", "status": result["status"], "test_used": False, "training_performed": True, "metrics_table": result["metrics_table"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
