"""文件用途：
    提供 ActiveView 统一评估能力。

主要输入：
    - 预测、标签与评估协议。
主要输出：
    - Accuracy、Macro-F1、regret 或摘要。
项目角色：
    - 属于 evaluation 评估模块。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from activeview.methods.active_view.geometry import order_candidates, second_step_decision, trajectory_cost
from activeview.methods.baselines.policies import _by_id, _selected_row, build_baseline_trajectories


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _classification(rows: Sequence[Mapping[str, Any]], classes: int) -> dict[str, Any]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for row in rows:
        confusion[int(row["label_id"]), int(row["predicted_label_id"])] += 1
    f1: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in range(classes):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        value = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(value)
        per_class[str(class_id)] = {"support": int(support), "accuracy": recall, "f1": value}
    total = int(confusion.sum())
    return {"n": total, "accuracy": float(np.trace(confusion) / total) if total else 0.0, "macro_f1": float(np.mean(f1)) if f1 else 0.0, "per_class": per_class}


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90))}


def summarize_trajectory_rows(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> dict[str, Any]:
    regrets = np.maximum(0.0, np.asarray([float(row["regret"]) for row in rows], dtype=np.float64))
    safe_positive = [row for row in rows if float(row["safe_oracle_utility"]) > 1e-6]
    captures = [float(row["selected_true_utility"]) / float(row["safe_oracle_utility"]) for row in safe_positive]
    clipped = np.clip(np.asarray(captures, dtype=np.float64), 0.0, 1.0) if captures else np.asarray([], dtype=np.float64)
    safe_sum = sum(float(row["safe_oracle_utility"]) for row in safe_positive)
    selected_sum = sum(max(0.0, float(row["selected_true_utility"])) for row in safe_positive)
    move_counts = {"move_0": 0, "move_1": 0, "move_2": 0}
    costs: list[float] = []
    for row in rows:
        move_counts[f"move_{int(row['moves'])}"] += 1
        costs.append(float(row["trajectory_geodesic_cost_m"]))
    return {
        "episode_count": len(rows),
        "recognition": _classification(rows, len(categories)),
        "decision_regret": _summary(regrets.tolist()),
        "positive_headroom_capture": {
            "episode_count": len(safe_positive),
            "raw_mean": float(np.mean(captures)) if captures else 0.0,
            "clipped_mean": float(np.mean(clipped)) if captures else 0.0,
            "aggregate_positive_clipped_ratio": float(selected_sum / safe_sum) if safe_sum else 0.0,
        },
        "movement": {
            **{f"{name}_rate": count / len(rows) if rows else 0.0 for name, count in move_counts.items()},
            "average_moves": float(np.mean([int(row["moves"]) for row in rows])) if rows else 0.0,
            "trajectory_geodesic_cost_m": _summary(costs),
        },
    }


def build_fixed_first_oracle(
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose the true best of Stay/p2/p3 after the frozen first action."""
    base = build_baseline_trajectories(stage_b_rows, v0_prediction_rows)["FrozenStageCv0"]
    cache = {str(row["episode_id"]): row for row in cache_rows}
    utilities = {str(row["episode_id"]): row for row in stage_b_rows}
    output: list[dict[str, Any]] = []
    for row in base:
        episode_id = str(row["episode_id"])
        cached = cache.get(episode_id)
        if cached is None or row["moves"] == 0:
            output.append(row)
            continue
        record = utilities[episode_id]
        by_id = _by_id(record)
        candidates = list(cached["remaining_candidate_ids"])
        targets = [float(value) for value in cached["second_step_utility_targets"]]
        best_index = int(np.argmax(np.asarray([0.0] + targets, dtype=np.float64)))
        if best_index == 0:
            output.append(row)
            continue
        selected_id = int(candidates[best_index - 1])
        cost = trajectory_cost(float(row["trajectory_geodesic_cost_m"]), float(cached["second_step_candidate_geodesic"][best_index - 1]))
        output.append(_selected_row(record, selected_id, moves=2, cost=cost))
    return output


def predict_second_step_dataset(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Predict U2 and apply the Stay-inclusive decision rule."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            predicted = model(
                batch["s0_feature"].to(device), batch["s1_feature"].to(device),
                batch["delta_semantic"].to(device), batch["candidate_geometry"].to(device),
                batch["candidate_mask"].to(device),
            ).cpu().numpy()
            valid = batch["candidate_mask"].numpy()
            geodesic = batch["candidate_geodesic"].numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                ids = list(batch["candidate_ids"][index])
                values = [float(value) for value in predicted[index][valid[index]]]
                distances = [float(value) for value in geodesic[index][valid[index]]]
                stays, selected_id, max_value = second_step_decision(values, ids, distances)
                rows.append({
                    "episode_id": str(episode_id), "remaining_candidate_ids": ids,
                    "predicted_utilities": values, "predicted_stays": bool(stays),
                    "predicted_candidate_viewpoint_id": None if stays else int(selected_id),
                    "max_predicted_utility": float(max_value),
                })
    return rows


def build_trajectories(
    stage_b_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    second_predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize final Stage D trajectories without exposing future outcomes."""
    predictions = {str(row["episode_id"]): row for row in second_predictions}
    if len(predictions) != len(second_predictions):
        raise ValueError("Duplicate Stage D prediction episode_id")
    v0 = {str(row["episode_id"]): row for row in v0_prediction_rows}
    cache = {str(row["episode_id"]): row for row in cache_rows}
    output: list[dict[str, Any]] = []
    for record in stage_b_rows:
        episode_id = str(record["episode_id"])
        first = v0.get(episode_id)
        if first is None:
            raise ValueError(f"Missing first-step prediction for {episode_id}")
        if bool(first["predicted_stays"]):
            output.append(_selected_row(record, None, moves=0, cost=0.0))
            continue
        p1_id = int(first["predicted_candidate_viewpoint_id"])
        by_id = _by_id(record)
        p1_geo = float(by_id[p1_id]["geodesic_distance_m"])
        second = predictions.get(episode_id)
        cached = cache.get(episode_id)
        if second is None or cached is None:
            # No finite s1→candidate path was available. The real protocol
            # terminates at the visited p1 rather than fabricating a move.
            output.append(_selected_row(record, p1_id, moves=1, cost=p1_geo))
            continue
        if bool(second["predicted_stays"]):
            output.append(_selected_row(record, p1_id, moves=1, cost=p1_geo))
            continue
        selected_id = int(second["predicted_candidate_viewpoint_id"])
        if selected_id not in set(int(value) for value in cached["remaining_candidate_ids"]):
            raise ValueError(f"Stage D selected unknown remaining candidate for {episode_id}")
        index = [int(value) for value in cached["remaining_candidate_ids"]].index(selected_id)
        second_geo = float(cached["second_step_candidate_geodesic"][index])
        output.append(_selected_row(record, selected_id, moves=2, cost=trajectory_cost(p1_geo, second_geo)))
    return output


def summarize_methods(method_rows: Mapping[str, Sequence[Mapping[str, Any]]], categories: Sequence[str]) -> dict[str, Any]:
    summaries = {name: summarize_trajectory_rows(rows, categories) for name, rows in method_rows.items()}
    baseline = summaries.get("NoMove", {}).get("recognition", {}).get("accuracy")
    if baseline is not None:
        for summary in summaries.values():
            mean_cost = float(summary["movement"]["trajectory_geodesic_cost_m"]["mean"])
            accuracy_delta = float(summary["recognition"]["accuracy"]) - float(baseline)
            summary["movement"]["accuracy_gain_vs_NoMove_per_average_meter"] = (
                accuracy_delta / mean_cost if mean_cost > 1e-12 else 0.0
            )
    return summaries
