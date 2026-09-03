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

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


NEAR_ZERO_TOLERANCE = 1e-6


def _classification(rows: Sequence[tuple[int, int, float]], num_classes: int) -> Dict[str, Any]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    entropies: List[float] = []
    for target, prediction, entropy in rows:
        confusion[int(target), int(prediction)] += 1
        entropies.append(float(entropy))
    f1_values: List[float] = []
    per_class: Dict[str, Dict[str, float | int]] = {}
    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(class_id)] = {"support": int(support), "accuracy": recall, "f1": f1}
    count = int(confusion.sum())
    return {
        "n": count,
        "accuracy": float(np.trace(confusion) / count) if count else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "per_class": per_class,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            ranks[inverse == index] = ranks[inverse == index].mean()
    return ranks


def regression_metrics(predicted: Sequence[float], target: Sequence[float]) -> Dict[str, float | int]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.size == 0 or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Regression arrays must be finite, non-empty and shape-aligned")
    error = pred - truth
    huber = np.where(np.abs(error) <= 1.0, 0.5 * error**2, np.abs(error) - 0.5)
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    pearson_denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(truth_centered**2))
    pred_rank = _rankdata(pred)
    truth_rank = _rankdata(truth)
    rank_denominator = np.sqrt(np.sum((pred_rank - pred_rank.mean())**2) * np.sum((truth_rank - truth_rank.mean())**2))
    return {
        "n": int(pred.size), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
        "huber": float(np.mean(huber)),
        "pearson": float(np.sum(pred_centered * truth_centered) / pearson_denominator) if pearson_denominator else 0.0,
        "spearman": float(np.sum((pred_rank - pred_rank.mean()) * (truth_rank - truth_rank.mean())) / rank_denominator) if rank_denominator else 0.0,
    }


def move_stay_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float | int]:
    truth = np.asarray([not bool(row["safe_oracle_stays"]) for row in rows], dtype=bool)
    pred = np.asarray([not bool(row["predicted_stays"]) for row in rows], dtype=bool)
    tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred)); fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred)); total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": total, "accuracy": float((tp + tn) / total) if total else 0.0, "precision": precision, "recall": recall, "f1": f1, "move_support": int(truth.sum()), "stay_support": int((~truth).sum())}


def _recognition_summary(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> Dict[str, Any]:
    recognition_rows: Dict[str, List[tuple[int, int, float]]] = defaultdict(list)
    for row in rows:
        target = int(row["label_id"])
        recognition_rows["NoMove"].append((target, int(row["current_predicted_label_id"]), float(row["current_entropy"])))
        recognition_rows["StageC"].append((target, int(row["selected_predicted_label_id"]), float(row["selected_entropy"])))
        recognition_rows["CandidateOracle"].append((target, int(row["candidate_oracle_predicted_label_id"]), float(row["candidate_oracle_entropy"])))
        recognition_rows["SafeOracle"].append((target, int(row["safe_oracle_predicted_label_id"]), float(row["safe_oracle_entropy"])))
    recognition = {name: _classification(values, len(categories)) for name, values in recognition_rows.items()}
    no_move = recognition["NoMove"]
    for name in ("StageC", "CandidateOracle", "SafeOracle"):
        recognition[name]["accuracy_gain_vs_NoMove"] = recognition[name]["accuracy"] - no_move["accuracy"]
        recognition[name]["macro_f1_gain_vs_NoMove"] = recognition[name]["macro_f1"] - no_move["macro_f1"]
    return recognition


def summarize_policy_predictions(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> Dict[str, Any]:
    if not rows:
        return {"episode_count": 0, "regression": {}, "candidate_oracle_hit_rate": 0.0, "safe_action_match_rate": 0.0, "move_stay": move_stay_metrics([]), "recognition": {}, "decision_regret": {}, "positive_headroom_capture": {}}
    pred_utilities = [value for row in rows for value in row["predicted_utilities"]]
    true_utilities = [value for row in rows for value in row["utility_targets"]]
    regrets = np.asarray([float(row["regret"]) for row in rows], dtype=np.float64)
    if np.any(regrets < -1e-5):
        raise ValueError("Decision regret is materially negative")
    regrets = np.maximum(regrets, 0.0)
    positive_rows = [row for row in rows if float(row["safe_oracle_utility"]) > NEAR_ZERO_TOLERANCE]
    captures = [float(row["selected_true_utility"]) / float(row["safe_oracle_utility"]) for row in positive_rows]
    clipped = [float(np.clip(value, 0.0, 1.0)) for value in captures]
    safe_sum = sum(float(row["safe_oracle_utility"]) for row in positive_rows)
    selected_positive_sum = sum(max(0.0, float(row["selected_true_utility"])) for row in positive_rows)
    recognition = _recognition_summary(rows, categories)
    per_region: Dict[str, Dict[str, Any]] = {}
    for region in sorted({str(row["region"]) for row in rows}):
        region_rows = [row for row in rows if str(row["region"]) == region]
        per_region[region] = _recognition_summary(region_rows, categories)
    return {
        "episode_count": len(rows),
        "regression": regression_metrics(pred_utilities, true_utilities),
        "candidate_oracle_hit_rate": float(np.mean([int(row["predicted_candidate_viewpoint_id"]) == int(row["candidate_oracle_viewpoint_id"]) for row in rows])),
        "safe_action_match_rate": float(np.mean([row["predicted_action"] == row["safe_oracle_action"] for row in rows])),
        "move_stay": move_stay_metrics(rows),
        "recognition": recognition,
        "per_region": per_region,
        "decision_regret": {"mean": float(regrets.mean()), "median": float(np.median(regrets)), "p90": float(np.percentile(regrets, 90)), "zero_or_near_zero_ratio": float(np.mean(regrets <= NEAR_ZERO_TOLERANCE))},
        "positive_headroom_capture": {"episode_count": len(positive_rows), "raw_mean": float(np.mean(captures)) if captures else 0.0, "clipped_mean": float(np.mean(clipped)) if clipped else 0.0, "aggregate_positive_clipped_ratio": float(selected_positive_sum / safe_sum) if safe_sum else 0.0},
    }
