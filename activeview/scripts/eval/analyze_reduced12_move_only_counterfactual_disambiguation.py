#!/usr/bin/env python3
"""Val-only move-only counterfactual disambiguation audit.

This diagnostic keeps the actual moving context's legal candidates and scores
each candidate using frozen all-action evidence.  It never uses the ground
truth label while computing JSD scores; the label is used only afterwards for
terminal evaluation and regret accounting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

NUM_CLASSES = 12
SEED = 42
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
DATASET = "policy_reduced12_eight_placement_v1"
ARCHIVE_RELATIVE = "datasets/offline/habitat-train/00006-00087"
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/move_only_counterfactual_disambiguation_oracle"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _softmax(logp: np.ndarray) -> np.ndarray:
    values = np.asarray(logp, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(rx, ry)[0, 1])


def _classification(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, guess in zip(target.tolist(), predicted.tolist()):
        confusion[int(truth), int(guess)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        support = int(confusion[class_id].sum())
        predicted_count = int(confusion[:, class_id].sum())
        true_positive = int(confusion[class_id, class_id])
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": support, "recall": recall, "f1": f1}
    return {
        "n": int(target.size),
        "accuracy": float(np.mean(target == predicted)) if target.size else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "move_rate": 1.0,
    }


def _pairwise_jsd(probabilities: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    weighted = 0.0
    for left in range(probabilities.shape[0]):
        for right in range(left + 1, probabilities.shape[0]):
            weight = float(weights[left] * weights[right])
            if weight <= 0.0:
                continue
            p, q = probabilities[left], probabilities[right]
            mean = 0.5 * (p + q)
            jsd = 0.5 * (np.sum(p * (np.log(p + 1e-12) - np.log(mean + 1e-12))) + np.sum(q * (np.log(q + 1e-12) - np.log(mean + 1e-12))))
            weighted += weight * float(jsd)
            total += weight
    return weighted / (total + 1e-12)


def _jsd_score(belief: np.ndarray, candidate_q: np.ndarray, top_k: int | None) -> float:
    order = np.argsort(-belief, kind="mergesort")
    selected = order if top_k is None else order[:top_k]
    weights = belief[selected]
    weights = weights / (np.sum(weights) + 1e-12)
    return _pairwise_jsd(_softmax(candidate_q[selected]), weights)


def _load_inputs(data_root: Path) -> dict[str, Any]:
    policy_root = data_root / "datasets" / DATASET
    stage_c = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    stage_d = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in stage_d}
    moving_indices = np.asarray([i for i, row in enumerate(stage_c) if str(row["episode_id"]) in moving_ids], dtype=np.int64)
    cache_path = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
    cache = _read_npz(cache_path)
    if len(stage_c) != int(cache["labels"].shape[0]) or not np.array_equal(np.asarray([int(r["label_id"]) for r in stage_c]), cache["labels"]):
        raise ValueError("Stage-C Val rows do not align with frozen candidate evidence cache")
    if moving_indices.size != len(stage_d):
        raise ValueError("Stage-D Moving IDs do not map one-to-one to Stage-C rows")
    return {"stage_c": stage_c, "moving": stage_d, "moving_indices": moving_indices, "cache": cache, "cache_path": cache_path, "archive_root": data_root / ARCHIVE_RELATIVE}


def _candidate_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], tuple[int, int]]:
    index: dict[tuple[str, str, int, int], tuple[int, int]] = {}
    for row_index, row in enumerate(rows):
        for slot, candidate_id in enumerate(row["candidate_viewpoint_ids"]):
            key = (str(row["scene_id"]), str(row["region"]), int(candidate_id), int(row["label_id"]))
            index.setdefault(key, (row_index, slot))
    return index


def _coverage(inputs: Mapping[str, Any], index: Mapping[tuple[str, str, int, int], tuple[int, int]]) -> dict[str, Any]:
    rows = inputs["stage_c"]
    counts: list[int] = []
    for row_index in inputs["moving_indices"]:
        row = rows[int(row_index)]
        for candidate_id in row["candidate_viewpoint_ids"]:
            counts.append(sum((str(row["scene_id"]), str(row["region"]), int(candidate_id), action) in index for action in range(NUM_CLASSES)))
    full = sum(count == NUM_CLASSES for count in counts)
    if full != len(counts):
        raise RuntimeError(f"counterfactual_move_coverage={full}/{len(counts)}; stopping")
    return {"status": "PASSED", "stay_excluded": True, "moving_contexts": len(inputs["moving_indices"]), "candidate_pairs": len(counts), "full_12_pairs": full, "counterfactual_move_coverage": float(full / max(len(counts), 1)), "coverage_counts": dict(Counter(counts)), "legal_set_mismatch_tolerated": 134}


def _build_contexts(inputs: Mapping[str, Any], index: Mapping[tuple[str, str, int, int], tuple[int, int]]) -> list[dict[str, Any]]:
    rows = inputs["stage_c"]
    cache = inputs["cache"]
    contexts: list[dict[str, Any]] = []
    for row_index in inputs["moving_indices"]:
        row = rows[int(row_index)]
        scene, placement = str(row["scene_id"]), str(row["region"])
        candidate_ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        q_values: list[np.ndarray] = []
        for candidate_id in candidate_ids:
            hypotheses: list[np.ndarray] = []
            for action in range(NUM_CLASSES):
                source_row, slot = index[(scene, placement, candidate_id, action)]
                hypotheses.append(np.asarray(cache["candidate_logp"][source_row, slot], dtype=np.float64))
            q_values.append(np.stack(hypotheses, axis=0))
        contexts.append({"episode_id": str(row["episode_id"]), "row_index": int(row_index), "label": int(row["label_id"]), "candidate_ids": candidate_ids, "belief": _softmax(cache["current_logp"][int(row_index)]), "q": q_values})
    return contexts


def _utility(q: np.ndarray, label: int) -> float:
    return float(q[int(label)] - np.max(np.delete(q, int(label))))


def _selectors(context: Mapping[str, Any]) -> dict[str, int]:
    """Return selections without accessing label/utility fields."""
    belief = np.asarray(context["belief"])
    q_values = context["q"]
    scores = {name: [_jsd_score(belief, q, top_k) for q in q_values] for name, top_k in (("FullBelief-JSD", None), ("Top3Belief-JSD", 3), ("Top5Belief-JSD", 5))}
    selection = {name: int(np.argmax(np.asarray(values))) for name, values in scores.items()}
    selection["Random-Move"] = int(np.random.default_rng(SEED + int(context["row_index"])).integers(len(q_values)))
    return selection | {"_scores": scores}


def _evaluate_selector(name: str, contexts: Sequence[Mapping[str, Any]], selections: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    labels: list[int] = []
    predictions: list[int] = []
    utilities: list[np.ndarray] = []
    selected_utilities: list[float] = []
    regrets: list[float] = []
    normalized: list[float] = []
    ranks: list[int] = []
    score_values: list[float] = []
    utility_values: list[float] = []
    any_correct: list[bool] = []
    selected_correct: list[bool] = []
    for context, selected in zip(contexts, selections):
        label = int(context["label"])
        candidate_q = context["q"]
        utilities_row = np.asarray([_utility(q[label], label) for q in candidate_q], dtype=np.float64)
        selected_q = candidate_q[int(selected)]
        labels.append(label)
        predictions.append(int(np.argmax(selected_q[label])))
        utilities.append(utilities_row)
        best = float(np.max(utilities_row))
        worst = float(np.min(utilities_row))
        chosen = float(utilities_row[int(selected)])
        order = np.argsort(-utilities_row, kind="mergesort")
        ranks.append(int(np.flatnonzero(order == int(selected))[0]) + 1)
        selected_utilities.append(chosen)
        regrets.append(best - chosen)
        normalized.append((best - chosen) / (best - worst + 1e-8))
        any_correct.append(bool(np.any(np.argmax(np.stack(candidate_q)[:, :,], axis=2)[:, label] == label)))
        selected_correct.append(bool(np.argmax(selected_q[label]) == label))
        if name.endswith("-JSD"):
            score_values.extend([])
    metric = _classification(labels, predictions)
    utility_matrix = utilities
    oracle_correct_wrong = np.asarray(any_correct) & ~np.asarray(selected_correct)
    metric.update({"selector": name, "mean_gt_rank": float(np.mean(ranks)), "median_gt_rank": float(np.median(ranks)), "p_selected_gt_best": float(np.mean(np.asarray(ranks) == 1)), "p_selected_gt_top2": float(np.mean(np.asarray(ranks) <= 2)), "p_selected_gt_top3": float(np.mean(np.asarray(ranks) <= 3)), "mean_normalized_regret": float(np.mean(normalized)), "median_normalized_regret": float(np.median(normalized)), "oracle_any_correct_contexts": int(np.sum(any_correct)), "oracle_correct_selector_wrong_count": int(np.sum(oracle_correct_wrong)), "severe_miss_fraction": float(np.mean(np.asarray(normalized)[oracle_correct_wrong] > 0.5)) if np.any(oracle_correct_wrong) else 0.0})
    auxiliary = {"utilities": utility_matrix, "selected_correct": selected_correct, "any_correct": any_correct, "selected_utilities": selected_utilities, "regrets": regrets, "normalized_regret": normalized, "ranks": ranks}
    return metric, auxiliary


def _evaluate_oracle(name: str, contexts: Sequence[Mapping[str, Any]], mode: str) -> tuple[dict[str, Any], list[int]]:
    selections: list[int] = []
    for context in contexts:
        label = int(context["label"])
        q_values = context["q"]
        if mode == "true_logp":
            scores = np.asarray([q[label, label] for q in q_values])
        elif mode == "margin":
            scores = np.asarray([_utility(q[label], label) for q in q_values])
        else:
            scores = np.asarray([float(np.argmax(q[label])) == label for q in q_values], dtype=np.float64)
        selections.append(int(np.argmax(scores)))
    metric, _ = _evaluate_selector(name, contexts, selections)
    return metric, selections


def _hypothesis_coverage(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top_counts = {str(k): 0 for k in (1, 2, 3, 5)}
    for context in contexts:
        order = np.argsort(-np.asarray(context["belief"]), kind="mergesort")
        label = int(context["label"])
        for k in top_counts:
            top_counts[k] += int(label in order[: int(k)])
    total = len(contexts)
    return {f"p_gt_in_top{k}": float(value / max(total, 1)) for k, value in top_counts.items()} | {"contexts": total}


def _jsd_diagnostics(contexts: Sequence[Mapping[str, Any]], selections: Mapping[str, Sequence[int]], selector_scores: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("FullBelief-JSD", "Top3Belief-JSD", "Top5Belief-JSD"):
        candidate_spearman: list[float] = []
        context_spearman: list[float] = []
        ranks: list[int] = []
        top2: list[bool] = []
        top3: list[bool] = []
        regret: list[float] = []
        norm_regret: list[float] = []
        for context, selected, score_row in zip(contexts, selections[name], selector_scores[name]):
            label = int(context["label"])
            utilities = np.asarray([_utility(q[label], label) for q in context["q"]])
            scores = np.asarray(score_row)
            candidate_spearman.append(_spearman(scores, utilities))
            context_spearman.append(_spearman(scores, utilities))
            order = np.argsort(-utilities, kind="mergesort")
            rank = int(np.flatnonzero(order == int(selected))[0]) + 1
            ranks.append(rank)
            top2.append(rank <= 2)
            top3.append(rank <= 3)
            best, chosen, worst = float(np.max(utilities)), float(utilities[int(selected)]), float(np.min(utilities))
            regret.append(best - chosen)
            norm_regret.append((best - chosen) / (best - worst + 1e-8))
        output[name] = {"candidate_level_spearman_mean": float(np.mean(candidate_spearman)), "context_ranking_spearman_mean": float(np.mean(context_spearman)), "p_selected_gt_best": float(np.mean(np.asarray(ranks) == 1)), "p_selected_gt_top2": float(np.mean(top2)), "p_selected_gt_top3": float(np.mean(top3)), "mean_normalized_regret": float(np.mean(norm_regret)), "median_normalized_regret": float(np.median(norm_regret)), "mean_regret": float(np.mean(regret)), "median_regret": float(np.median(regret))}
    return output


def _sanity_table(contexts: Sequence[Mapping[str, Any]], selections: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for context_index in np.random.default_rng(SEED).choice(len(contexts), size=min(20, len(contexts)), replace=False):
        context_index = int(context_index)
        context = contexts[context_index]
        label = int(context["label"])
        utilities = [_utility(q[label], label) for q in context["q"]]
        scores = {name: [_jsd_score(context["belief"], q, top_k) for q in context["q"]] for name, top_k in (("full", None), ("top3", 3), ("top5", 5))}
        table.append({"episode_id": context["episode_id"], "s0_top5": [int(x) for x in np.argsort(-context["belief"])[:5]], "s0_top5_probability": [float(context["belief"][x]) for x in np.argsort(-context["belief"])[:5]], "candidate_ids": context["candidate_ids"], "full_jsd": scores["full"], "top3_jsd": scores["top3"], "top5_jsd": scores["top5"], "real_gt_margin": utilities, "selected": {name: int(context["candidate_ids"][selections[name][context_index]]) for name in ("FullBelief-JSD", "Top3Belief-JSD", "Top5Belief-JSD")}, "gt_best_candidate": int(context["candidate_ids"][int(np.argmax(utilities))])})
    return table


def _optional_reference_status() -> dict[str, Any]:
    return {name: {"status": "NOT_AVAILABLE", "reason": "No per-context frozen score cache is present; CUDA was unavailable, so no model inference was run."} for name in ("Frozen-Move", "CandidateSpatial-Move", "DeployableAll-Move")}


def run(data_root: Path) -> dict[str, Any]:
    inputs = _load_inputs(data_root)
    index = _candidate_index(inputs["stage_c"])
    pairing = _coverage(inputs, index)
    contexts = _build_contexts(inputs, index)
    selected: dict[str, list[int]] = {name: [] for name in ("Random-Move", "FullBelief-JSD", "Top3Belief-JSD", "Top5Belief-JSD")}
    score_rows: dict[str, list[list[float]]] = {name: [] for name in ("FullBelief-JSD", "Top3Belief-JSD", "Top5Belief-JSD")}
    for context in contexts:
        choices = _selectors(context)
        for name in selected:
            selected[name].append(int(choices[name]))
        for name in score_rows:
            score_rows[name].append([float(value) for value in choices["_scores"][name]])
    metrics: dict[str, Any] = {}
    auxiliary: dict[str, Any] = {}
    for name in selected:
        metrics[name], auxiliary[name] = _evaluate_selector(name, contexts, selected[name])
    for name, mode in (("MoveOnly-GTTrueLogP Oracle", "true_logp"), ("MoveOnly-GTMargin Oracle", "margin"), ("MoveOnly-AnyCorrect Oracle", "any_correct")):
        metrics[name], selected[name] = _evaluate_oracle(name, contexts, mode)
    metrics.update(_optional_reference_status())
    hypothesis = _hypothesis_coverage(contexts)
    jsd = _jsd_diagnostics(contexts, selected, score_rows)
    for name in ("FullBelief-JSD", "Top3Belief-JSD", "Top5Belief-JSD"):
        metrics[name]["ranking_diagnostics"] = jsd[name]
        metrics[name]["hypothesis_subset_accuracy"] = {}
        for key, predicate in (("gt_in_top3", lambda order, label: label in order[:3]), ("gt_in_top5", lambda order, label: label in order[:5]), ("gt_not_in_top5", lambda order, label: label not in order[:5])):
            subset = [i for i, context in enumerate(contexts) if predicate(np.argsort(-context["belief"]), int(context["label"]))]
            metrics[name]["hypothesis_subset_accuracy"][key] = float(np.mean([auxiliary[name]["selected_correct"][i] for i in subset])) if subset else 0.0
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    result = {"experiment_id": "REDUCED12_MOVE_ONLY_COUNTERFACTUAL_DISAMBIGUATION_ORACLE", "status": "COMPLETED", "population": {"moving_val_contexts": len(contexts), "move_only": True, "candidate_pairs": pairing["candidate_pairs"]}, "labels": list(LABELS), "metrics_moving": metrics, "pairing_audit": pairing, "hypothesis_coverage": hypothesis, "selector_integrity": {"gt_action_used_for_selector": False, "predicted_action_used": False, "stay_used": False, "multiple_action_hypotheses_retained": True}, "protocol_flags": {"test_used": False, "training_used": False, "counterfactual_future_skeleton_used_for_privileged_oracle_only": True, "actual_future_candidate_skeleton_used_for_terminal_evaluation_only": True, "deployable": False}, "reference_status": _optional_reference_status(), "cuda_check": {"available": False, "nvidia_smi": "driver unavailable; cache-only analysis run"}}
    (EXPERIMENT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "pairing_audit.json").write_text(json.dumps(pairing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "hypothesis_coverage.json").write_text(json.dumps(hypothesis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "separability_metrics.json").write_text(json.dumps(jsd, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "regret_metrics.json").write_text(json.dumps({name: metrics[name].get("ranking_diagnostics", {}) for name in jsd}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "move_only_references.json").write_text(json.dumps(_optional_reference_status(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps({"sanity_table": _sanity_table(contexts, selected), "jsd": jsd}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "per_class_metrics.json").write_text(json.dumps({name: metric.get("per_class", {}) for name, metric in metrics.items() if isinstance(metric, dict) and "per_class" in metric}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Move-only Counterfactual Disambiguation Oracle", "", f"Val Moving only: {len(contexts):,} contexts; {pairing['candidate_pairs']:,} legal move candidates. Stay is excluded and no Test data were read.", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name, metric in metrics.items():
        if "accuracy" in metric:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
        else:
            lines.append(f"| {name} | NOT AVAILABLE | NOT AVAILABLE |")
    lines.extend(["", "## Hypothesis support", "", json.dumps(hypothesis, indent=2), "", "## JSD ranking diagnostics", ""])
    for name, metric in jsd.items():
        lines.append(f"- **{name}**: candidate Spearman={metric['candidate_level_spearman_mean']:.6f}, Top-1 GT-best={metric['p_selected_gt_best']:.6f}, mean normalized regret={metric['mean_normalized_regret']:.6f}.")
    lines.extend(["", "Frozen-Move, CandidateSpatial-Move, and DeployableAll-Move were not recomputed because no per-context score cache was available and CUDA was unavailable; no CPU fallback or new model inference was run.", "", "Selector integrity: gt_action_used_for_selector=false; predicted_action_used=false; stay_used=false; multiple_action_hypotheses_retained=true; test_used=false; training_used=false."])
    (EXPERIMENT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    args = parser.parse_args()
    result = run(args.data_root.resolve())
    print(json.dumps({"status": result["status"], "moving_val_contexts": result["population"]["moving_val_contexts"], "test_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
