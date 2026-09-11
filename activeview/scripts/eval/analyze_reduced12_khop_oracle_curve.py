#!/usr/bin/env python3
"""Audit lattice-reachable and greedy sequential oracle ceilings on Val."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    classification,
    gt_margin,
    lattice_distance,
    legal_ids,
    load_train_val,
)


OUT = Path("experiments/reduced12_eight_placement_v1/khop_oracle_curve")
HOPS = (0, 1, 2, 3, 4, 5, 6)


def _contexts(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    cache = data["caches"]["val"]
    contexts: list[dict[str, Any]] = []
    for row in data["stage_d_rows"]["val"]:
        index = int(row["cache_index"])
        candidates = legal_ids(cache, index)
        h1 = int(row["s1_viewpoint_id"])
        if not candidates or h1 not in candidates:
            raise ValueError(f"invalid moving context candidate set: {row['episode_id']}")
        label = int(row["label_id"])
        margins = {}
        predictions = {}
        correct = {}
        for candidate in candidates:
            slot = int(np.flatnonzero(cache["candidate_ids"][index] == candidate)[0])
            logp = np.asarray(cache["candidate_logp"][index, slot], dtype=np.float32)
            margins[candidate] = gt_margin(logp, label)
            predictions[candidate] = int(np.argmax(logp))
            correct[candidate] = bool(predictions[candidate] == label)
        contexts.append({"row": row, "index": index, "label": label, "candidates": candidates, "h1": h1, "margins": margins, "predictions": predictions, "correct": correct})
    return contexts


def _classification_for(contexts: Sequence[Mapping[str, Any]], selected: Sequence[int]) -> dict[str, Any]:
    labels = [int(context["label"]) for context in contexts]
    predictions = [int(context["predictions"][action]) if action != -1 else int(np.argmax(context["row"]["s1_feature"][256:268])) for context, action in zip(contexts, selected)]
    result = classification(labels, predictions)
    result["move_rate"] = float(np.mean(np.asarray(selected) >= 0)) if selected else 0.0
    result["stay_rate"] = 1.0 - result["move_rate"]
    return result


def _best(context: Mapping[str, Any], candidates: Sequence[int]) -> int:
    return max((int(candidate) for candidate in candidates), key=lambda candidate: (float(context["margins"][candidate]), -candidate))


def _khop(contexts: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_by_hop: dict[str, list[int]] = {str(hop): [] for hop in HOPS}
    selected_by_hop["Full"] = []
    coverage: dict[str, int] = {str(hop): 0 for hop in HOPS}
    coverage["Full"] = 0
    min_any: list[int | None] = []
    min_best: list[int | None] = []
    for context in contexts:
        h1 = int(context["h1"])
        candidates = list(context["candidates"])
        correct = context["correct"]
        for hop in HOPS:
            allowed = [candidate for candidate in candidates if lattice_distance(h1, candidate) <= hop]
            selected_by_hop[str(hop)].append(_best(context, allowed))
            coverage[str(hop)] += int(any(correct[candidate] for candidate in allowed))
        selected_by_hop["Full"].append(_best(context, candidates))
        coverage["Full"] += int(any(correct.values()))
        correct_hops = [lattice_distance(h1, candidate) for candidate in candidates if correct[candidate]]
        min_any.append(min(correct_hops) if correct_hops else None)
        best = _best(context, candidates)
        min_best.append(lattice_distance(h1, best))
    metrics: dict[str, Any] = {}
    for name, actions in selected_by_hop.items():
        item = _classification_for(contexts, actions)
        item["any_correct_coverage"] = float(coverage[name] / len(contexts))
        item["any_correct_contexts"] = int(coverage[name])
        item["hop"] = name
        metrics[name] = item
    full_acc = float(metrics["Full"]["accuracy"])
    for hop in HOPS:
        previous = metrics["0"]["accuracy"] if hop == 0 else metrics[str(hop - 1)]["accuracy"]
        metrics[str(hop)]["delta_accuracy_vs_previous"] = float(metrics[str(hop)]["accuracy"] - previous)
        metrics[str(hop)]["full_accuracy_gap"] = float(full_acc - metrics[str(hop)]["accuracy"])
    return metrics, {"min_hop_to_any_correct": min_any, "min_hop_to_gt_margin_best": min_best}


def _bucket_hops(values: Sequence[int | None]) -> dict[str, Any]:
    counts = Counter()
    for value in values:
        if value is None:
            counts["unreachable/none"] += 1
        elif value == 0:
            counts["0"] += 1
        elif value == 1:
            counts["1"] += 1
        elif value == 2:
            counts["2"] += 1
        elif value == 3:
            counts["3"] += 1
        elif value == 4:
            counts["4"] += 1
        else:
            counts["5+"] += 1
    total = max(len(values), 1)
    return {"counts": {key: int(counts.get(key, 0)) for key in ("0", "1", "2", "3", "4", "5+", "unreachable/none")}, "proportions": {key: float(counts.get(key, 0) / total) for key in ("0", "1", "2", "3", "4", "5+", "unreachable/none")}}


def _greedy(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outputs: dict[str, list[int]] = {f"GreedyOracle-Step{step}": [] for step in range(1, 5)}
    move_rates: dict[str, int] = {name: 0 for name in outputs}
    for context in contexts:
        current = int(context["h1"])
        candidates = set(int(value) for value in context["candidates"])
        for step in range(1, 5):
            neighbors = [candidate for candidate in candidates if candidate != current and lattice_distance(current, candidate) == 1]
            options = [current] + neighbors
            next_view = _best(context, options)
            if next_view != current:
                move_rates[f"GreedyOracle-Step{step}"] += 1
            current = next_view
            outputs[f"GreedyOracle-Step{step}"].append(current)
    result: dict[str, Any] = {}
    for name, actions in outputs.items():
        result[name] = _classification_for(contexts, actions)
        result[name]["move_rate"] = float(move_rates[name] / len(contexts))
        result[name]["stay_rate"] = 1.0 - result[name]["move_rate"]
        result[name]["step"] = int(name.rsplit("Step", 1)[1])
    return result


def _basins(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_context: list[dict[str, Any]] = []
    for context in contexts:
        correct_nodes = {candidate for candidate in context["candidates"] if context["correct"][candidate]}
        components: list[list[int]] = []
        remaining = set(correct_nodes)
        while remaining:
            stack = [remaining.pop()]
            component: list[int] = []
            while stack:
                node = stack.pop(); component.append(node)
                neighbors = {other for other in remaining if lattice_distance(node, other) == 1}
                remaining.difference_update(neighbors); stack.extend(neighbors)
            components.append(sorted(component))
        sizes = sorted((len(component) for component in components), reverse=True)
        largest = sizes[0] if sizes else 0
        per_context.append({"context": str(context["row"]["episode_id"]), "correct_nodes": len(correct_nodes), "components": len(components), "largest_component_size": largest, "isolated_correct_nodes": int(sum(size == 1 for size in sizes)), "fraction_correct_in_largest": float(largest / len(correct_nodes)) if correct_nodes else None})
    with_correct = [item for item in per_context if item["correct_nodes"] > 0]
    def values(key: str) -> list[float]:
        return [float(item[key]) for item in with_correct if item[key] is not None]
    return {"contexts": len(per_context), "contexts_with_at_least_one_correct": len(with_correct), "fraction_contexts_with_correct": float(len(with_correct) / len(per_context)) if per_context else 0.0, "mean_components": float(np.mean(values("components"))) if with_correct else 0.0, "median_components": float(np.median(values("components"))) if with_correct else 0.0, "mean_largest_component_size": float(np.mean(values("largest_component_size"))) if with_correct else 0.0, "median_largest_component_size": float(np.median(values("largest_component_size"))) if with_correct else 0.0, "mean_isolated_correct_nodes": float(np.mean(values("isolated_correct_nodes"))) if with_correct else 0.0, "fraction_correct_nodes_in_largest": float(np.mean(values("fraction_correct_in_largest"))) if with_correct else 0.0, "per_context": per_context}


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    data = load_train_val()
    contexts = _contexts(data)
    khop_metrics, minimum = _khop(contexts)
    greedy = _greedy(contexts)
    greedy["PathOracle-K1-to-K4-AnyCorrect"] = {
        "definition": "exists a legal candidate on a path of length <=K; equivalent to K-hop any-correct coverage",
        "steps": {str(step): float(khop_metrics[str(step)]["any_correct_coverage"]) for step in range(1, 5)},
    }
    basins = _basins(contexts)
    minimum_output = {"min_hop_to_any_correct": _bucket_hops(minimum["min_hop_to_any_correct"]), "min_hop_to_gt_margin_best": _bucket_hops(minimum["min_hop_to_gt_margin_best"])}
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "khop_metrics.json", khop_metrics)
    _write(OUT / "minimum_hop_distribution.json", minimum_output)
    _write(OUT / "greedy_local_oracle.json", greedy)
    _write(OUT / "correct_basin_statistics.json", {key: value for key, value in basins.items() if key != "per_context"})
    result = {"experiment_id": "REDUCED12_KHOP_ORACLE_CURVE", "population": {"split": "val_moving", "contexts": len(contexts), "candidate_samples": int(sum(len(item["candidates"]) for item in contexts))}, "khop_metrics": khop_metrics, "minimum_hop_distribution": minimum_output, "greedy_local_oracle": greedy, "correct_basin_statistics": {key: value for key, value in basins.items() if key != "per_context"}, "labels": list(LABELS), "protocol": {"lattice": "4 radii x 8 azimuths; Manhattan radial+angular distance", "start": "FrozenStageCv0 H1 viewpoint", "candidate_set": "actual legal candidates only", "K0": "H1 candidate", "oracle": "GT-margin posthoc; terminal archived real ST-GCN evidence"}, "policy_test_used": False, "policy_val_used_for_diagnostic_only": True, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "new_dino_generated": False, "gt_action_used_for_posthoc_grouping_only": True, "gt_margin_used_for_privileged_diagnostic_only": True, "future_candidate_observation_used_at_inference": False, "deployable": False}
    _write(OUT / "result.json", result)
    lines = ["# Reduced12 K-hop reachable oracle curve", "", f"Moving Val contexts: {len(contexts)}", "", "## K-hop reachability oracle", "", "| Hop | Accuracy | Macro-F1 | AnyCorrect coverage | ΔAcc vs previous | Full gap |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["0", "1", "2", "3", "4", "5", "6", "Full"]:
        item = khop_metrics[name]
        lines.append(f"| {'H1' if name == '0' else 'Full' if name == 'Full' else 'K'+name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['any_correct_coverage']:.6f} | {item.get('delta_accuracy_vs_previous', 0.0):.6f} | {item.get('full_accuracy_gap', 0.0):.6f} |")
    lines.extend(["", "## Greedy local oracle", "", "| Selector | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|"])
    for name, item in greedy.items():
        if name.startswith("Greedy"):
            lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} |")
    lines.extend(["", f"Minimum hop to any correct candidate: {minimum_output['min_hop_to_any_correct']['proportions']}", f"Minimum hop to GT-margin-best candidate: {minimum_output['min_hop_to_gt_margin_best']['proportions']}", "", "## Correct-view basins", "", f"Contexts with a correct candidate: {basins['fraction_contexts_with_correct']:.6f}; mean largest correct component: {basins['mean_largest_component_size']:.4f}; mean fraction in largest component: {basins['fraction_correct_nodes_in_largest']:.6f}.", "", "The path oracle is the K-hop reachability ceiling; greedy results show whether monotonic local exploration can attain that ceiling. This is a Val-only privileged diagnostic and did not read policy Test or generate data."])
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
