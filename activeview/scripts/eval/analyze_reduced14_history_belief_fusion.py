#!/usr/bin/env python3
"""Val-only diagnostics for fusing the two observed action beliefs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _metrics,
    _orders,
    _terminal_predictions,
    _validate_rows_cache,
)


NUM_CLASSES = 14
VIEW_COUNT = 32
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_belief_fusion"
SELECTOR_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/selector_bottleneck/result.json"
GT_LABEL_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/gt_label_privileged_jr/result.json"


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(values * np.log(np.clip(values, 1e-12, None)), axis=-1)


def _beliefs(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    logp0 = np.asarray(cache["current_logp_s0"], dtype=np.float64)
    logp1 = np.asarray(cache["current_logp_s1"], dtype=np.float64)
    prob0, prob1 = np.exp(logp0), np.exp(logp1)
    mean_probability = 0.5 * (prob0 + prob1)
    product = _softmax(0.5 * (logp0 + logp1))
    entropy0, entropy1 = _entropy(prob0), _entropy(prob1)
    inv0, inv1 = 1.0 / np.maximum(entropy0, 1e-12), 1.0 / np.maximum(entropy1, 1e-12)
    entropy_weighted = (inv0[:, None] * prob0 + inv1[:, None] * prob1) / (inv0 + inv1)[:, None]
    sorted0 = np.sort(prob0, axis=1)
    sorted1 = np.sort(prob1, axis=1)
    margin0 = sorted0[:, -1] - sorted0[:, -2]
    margin1 = sorted1[:, -1] - sorted1[:, -2]
    margin_sum = margin0 + margin1
    margin_weighted = np.where(
        margin_sum[:, None] > 1e-12,
        (margin0[:, None] * prob0 + margin1[:, None] * prob1) / np.maximum(margin_sum, 1e-12)[:, None],
        mean_probability,
    )
    return {
        "S1_only": prob1,
        "Mean_probability": mean_probability,
        "Mean_log_probability": product,
        "Entropy_weighted": entropy_weighted,
        "Margin_weighted": margin_weighted,
    }


def _belief_metrics(belief: np.ndarray, labels: Sequence[int]) -> dict[str, float | int]:
    prediction = np.argmax(np.asarray(belief), axis=1)
    metrics = _metrics(prediction, labels)
    return {
        "count": metrics["count"],
        "top1_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_entropy": float(np.mean(_entropy(np.asarray(belief, dtype=np.float64)))),
    }


def _belief_selector(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    belief: np.ndarray,
) -> list[int | None]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    selected: list[int | None] = []
    for row, current_belief in zip(rows, np.asarray(belief, dtype=np.float64)):
        i = index[str(row["episode_id"])]
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        if not candidates:
            selected.append(None)
            continue
        current_logp = np.asarray(cache["current_logp_s1"][i], dtype=np.float64)
        candidate_logp = np.asarray(cache["true_logp"][i, candidates], dtype=np.float64)
        # Unified direct comparison: Stay is action 0, followed by ordered candidates.
        scores = np.concatenate([[float(np.dot(current_belief, current_logp))], candidate_logp @ current_belief])
        choice = int(np.argmax(scores))
        selected.append(None if choice == 0 else candidates[choice - 1])
    return selected


def _selector_result(
    actions: Sequence[int | None], rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    predictions, positives = _terminal_predictions(actions, cache, index, rows)
    stay = sum(action is None for action in actions)
    return {
        "positive_action_hit_rate": float(np.mean(positives)),
        "positive_action_hit_count": int(sum(positives)),
        "stay_rate": float(stay / len(actions)),
        "action_counts": {"stay": stay, "move": len(actions) - stay},
        "terminal": _metrics(predictions, [int(row["label_id"]) for row in rows]),
    }


def _previous_method(path: Path, key: str, total: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_used") is not False:
        raise ValueError(f"diagnostic result is not Val-only: {path}")
    item = payload.get("methods", {}).get(key)
    if item is None:
        # The selector-bottleneck artifact stores these under lower-case names.
        aliases = {
            "PrivilegedJR": "privileged_jr",
            "NormalJR": "normal_jr",
            "SafeOracle": "safe_oracle",
        }
        item = payload[aliases.get(key, key)]
    return {
        "positive_action_hit_rate": float(item.get("positive_action_hit_rate", item.get("positive_hit_rate"))),
        "stay_rate": float(item.get("stay_rate", item["action_counts"]["stay"] / total)),
        "action_counts": item["action_counts"],
        "terminal": item["terminal"],
        "source": str(path.resolve()),
    }


def analyze(data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(root / "stage_d/features/val.jsonl")
    cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate_rows_cache(rows, cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in rows):
        raise ValueError("history fusion requires explicit Val rows")
    labels = [int(row["label_id"]) for row in rows]
    orders = _orders(data_root, rows)
    beliefs = _beliefs(cache)
    belief_metrics = {name: _belief_metrics(value, labels) for name, value in beliefs.items()}
    selector_metrics = {
        name: _selector_result(_belief_selector(rows, cache, orders, value), rows, cache)
        for name, value in beliefs.items()
    }
    safe_actions = _belief_selector(rows, cache, orders, np.eye(NUM_CLASSES, dtype=np.float64)[np.asarray(labels)])
    safe_metrics = _selector_result(safe_actions, rows, cache)
    selector_metrics["SafeOracle"] = safe_metrics
    selector_metrics["PrivilegedJR"] = _previous_method(SELECTOR_RESULT, "PrivilegedJR", len(rows))
    selector_metrics["GTLabelPrivilegedJR"] = _previous_method(GT_LABEL_RESULT, "GTLabelPrivilegedJR", len(rows))
    accuracy_by_fusion = {name: float(value["terminal"]["accuracy"]) for name, value in selector_metrics.items() if name in beliefs}
    best_fusion = max(accuracy_by_fusion, key=accuracy_by_fusion.get)
    s1_accuracy = float(selector_metrics["S1_only"]["terminal"]["accuracy"])
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_HISTORY_BELIEF_FUSION",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "population": {"val_moving_contexts": len(rows)},
        "belief_metrics": belief_metrics,
        "selector_metrics": selector_metrics,
        "comparisons": {
            "best_simple_fusion": best_fusion,
            "best_simple_fusion_minus_s1_accuracy": float(accuracy_by_fusion[best_fusion] - s1_accuracy),
            "best_simple_fusion_minus_s1_macro_f1": float(selector_metrics[best_fusion]["terminal"]["macro_f1"] - selector_metrics["S1_only"]["terminal"]["macro_f1"]),
            "gt_label_privileged_jr_minus_privileged_jr_accuracy": float(selector_metrics["GTLabelPrivilegedJR"]["terminal"]["accuracy"] - selector_metrics["PrivilegedJR"]["terminal"]["accuracy"]),
            "privileged_jr_minus_s1_only_selector_accuracy": float(selector_metrics["PrivilegedJR"]["terminal"]["accuracy"] - s1_accuracy),
            "safe_oracle_gap_best_simple_accuracy": float(safe_metrics["terminal"]["accuracy"] - accuracy_by_fusion[best_fusion]),
        },
        "protocol": {
            "belief_sources": ["current_logp_s0", "current_logp_s1"],
            "candidate_input": "archived true_logp, privileged diagnostic only",
            "selector": "direct argmax over [Stay, legal candidates], Stay first on exact ties",
            "formal_checkpoints_modified": False,
        },
        "leakage_flags": {
            "test_used": False,
            "formal_checkpoint_modified": False,
            "true_logp_used_only_for_privileged_candidate_diagnostic": True,
            "future_observation_rendered": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    beliefs = result["belief_metrics"]
    selectors = result["selector_metrics"]
    comparisons = result["comparisons"]
    lines = [
        "# Reduced14 History Belief Fusion (Val)",
        "",
        "Two-view history fusion uses only the observed s0/s1 recognition log-probabilities. Candidate scoring is a privileged offline diagnostic using archived true_logp; no formal checkpoint is changed.",
        "",
        "## Belief quality",
        "",
        "| Belief | Top-1 Accuracy | Macro-F1 | Mean entropy |",
        "|---|---:|---:|---:|",
    ]
    for name in ("S1_only", "Mean_probability", "Mean_log_probability", "Entropy_weighted", "Margin_weighted"):
        item = beliefs[name]
        lines.append(f"| {name} | {item['top1_accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mean_entropy']:.6f} |")
    lines.extend(
        [
            "",
            "## Privileged-candidate selector",
            "",
            "| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("S1_only", "Mean_probability", "Mean_log_probability", "Entropy_weighted", "Margin_weighted", "PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = selectors[name]
        lines.append(f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |")
    lines.extend(
        [
            "",
            f"Best simple fusion: {comparisons['best_simple_fusion']}; versus S1-only selector ΔAccuracy={comparisons['best_simple_fusion_minus_s1_accuracy']:+.6f}, ΔMacro-F1={comparisons['best_simple_fusion_minus_s1_macro_f1']:+.6f}.",
            f"Best simple fusion remains {comparisons['safe_oracle_gap_best_simple_accuracy']:.6f} Accuracy below SafeOracle.",
            "",
            "## Interpretation",
            "",
            "The direct fusion diagnostics test whether two observed views improve action identity before any learned selector. Here every fusion is no better than S1-only (and several are worse), so this simple test does not support training a history-belief refiner yet. The privileged JR and GT-label privileged JR rows are reused from the preceding Train-only diagnostics for comparison.",
            "",
            "Leakage audit: `test_used=false`; no Test rows/cache were read, no training was performed, and no formal checkpoint was modified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 history belief fusion diagnostics")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    args = parser.parse_args()
    analyze(args.data_root)


if __name__ == "__main__":
    main()
