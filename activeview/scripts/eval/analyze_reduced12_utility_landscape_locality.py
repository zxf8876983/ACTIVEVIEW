#!/usr/bin/env python3
"""Audit local spatial structure of reduced12 candidate HAR utility."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    angular_neighbor,
    candidate_logp,
    classification,
    correlation,
    gt_margin,
    lattice_distance,
    legal_ids,
    load_train_val,
    radial_neighbor,
    terminal_from_actions,
)

OUT = Path("experiments/reduced12_eight_placement_v1/utility_landscape_locality")


def _contexts(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return moving Val contexts with exact Stage-C prediction ordering."""
    contexts = []
    for row in data["stage_d_rows"]["val"]:
        cache = data["caches"]["val"]
        index = int(row["cache_index"])
        candidates = legal_ids(cache, index)
        if len(candidates) == 0:
            continue
        prediction = row["stage_c_prediction"]
        selected = int(row["s1_viewpoint_id"])
        if selected not in candidates:
            raise ValueError(f"H1 viewpoint {selected} is not legal for {row['episode_id']}")
        contexts.append({"row": row, "index": index, "label": int(row["label_id"]), "candidates": candidates, "h1": selected, "prediction": prediction})
    if not contexts:
        raise RuntimeError("no moving Val contexts")
    return contexts


def _pair_statistics(contexts: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values: dict[str, list[tuple[float, float, bool, bool]]] = defaultdict(list)
    for context in contexts:
        index, label = int(context["index"]), int(context["label"])
        candidates = list(context["candidates"])
        for position, left in enumerate(candidates):
            for right in candidates[position + 1 :]:
                kind = "angular" if angular_neighbor(left, right) else "radial" if radial_neighbor(left, right) else None
                if kind is None:
                    continue
                left_logp = candidate_logp(cache, index, left)
                right_logp = candidate_logp(cache, index, right)
                values[kind].append((gt_margin(left_logp, label), gt_margin(right_logp, label), bool(np.argmax(left_logp) == label), bool(np.argmax(right_logp) == label)))
    values["all_1hop"] = values["angular"] + values["radial"]
    result: dict[str, Any] = {}
    for name, pairs in values.items():
        if not pairs:
            result[name] = {"pairs": 0, "pearson": 0.0, "spearman": 0.0, "p_neighbor_correct_given_center_correct": 0.0, "p_neighbor_correct_given_center_wrong": 0.0, "mean_abs_margin_difference": 0.0}
            continue
        left = np.asarray([item[0] for item in pairs], dtype=np.float64)
        right = np.asarray([item[1] for item in pairs], dtype=np.float64)
        center_correct = np.asarray([item[2] for item in pairs], dtype=bool)
        neighbor_correct = np.asarray([item[3] for item in pairs], dtype=bool)
        result[name] = {
            "pairs": len(pairs),
            "pearson": correlation(left, right),
            "spearman": correlation(left, right, spearman=True),
            "p_neighbor_correct_given_center_correct": float(np.mean(neighbor_correct[center_correct])) if np.any(center_correct) else 0.0,
            "p_neighbor_correct_given_center_wrong": float(np.mean(neighbor_correct[~center_correct])) if np.any(~center_correct) else 0.0,
            "mean_abs_margin_difference": float(np.mean(np.abs(left - right))),
        }
    return result


def _margin(context: Mapping[str, Any], cache: Mapping[str, np.ndarray], viewpoint: int) -> float:
    return gt_margin(candidate_logp(cache, int(context["index"]), viewpoint), int(context["label"]))


def _local_actions(context: Mapping[str, Any], radius: int, cache: Mapping[str, np.ndarray]) -> int:
    h1 = int(context["h1"])
    allowed = [candidate for candidate in context["candidates"] if lattice_distance(h1, candidate) <= radius]
    return max(allowed, key=lambda candidate: (_margin(context, cache, candidate), -candidate))


def _terminal_actions(contexts: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], selector: str) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    moves = 0
    for context in contexts:
        index, label = int(context["index"]), int(context["label"])
        candidates = list(context["candidates"])
        if selector == "H1":
            selected = int(context["h1"])
        elif selector == "1-Hop Oracle":
            selected = _local_actions(context, 1, cache)
        elif selector == "2-Hop Oracle":
            selected = _local_actions(context, 2, cache)
        elif selector == "Full GT-margin Oracle":
            options = [(gt_margin(np.asarray(context["row"]["s0_feature"][256:268], dtype=np.float32), label), -1)]
            options.extend((_margin(context, cache, candidate), candidate) for candidate in candidates)
            selected = max(options, key=lambda item: (item[0], -item[1]))[1]
        elif selector == "AnyCorrect Oracle":
            s0 = np.asarray(context["row"]["s0_feature"][256:268], dtype=np.float32)
            if int(np.argmax(s0)) == label:
                selected = -1
            else:
                correct = [candidate for candidate in candidates if int(np.argmax(candidate_logp(cache, index, candidate))) == label]
                selected = correct[0] if correct else -1
        else:
            raise ValueError(selector)
        if selected == -1:
            logp = np.asarray(context["row"]["s0_feature"][256:268], dtype=np.float32)
        else:
            logp = candidate_logp(cache, index, selected)
            moves += 1
        labels.append(label)
        predictions.append(int(np.argmax(logp)))
    result = classification(labels, predictions)
    result.update({"selector": selector, "move_rate": float(moves / len(contexts)), "stay_rate": float(1.0 - moves / len(contexts))})
    return result


def _distance_audit(contexts: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    counts = {"h1_itself": 0, "one_hop": 0, "within_two_hops": 0, "over_two_hops": 0}
    distances: list[int] = []
    for context in contexts:
        best = max(context["candidates"], key=lambda candidate: (_margin(context, cache, candidate), -candidate))
        distance = lattice_distance(int(context["h1"]), int(best))
        distances.append(distance)
        if distance == 0:
            counts["h1_itself"] += 1
        elif distance == 1:
            counts["one_hop"] += 1
        elif distance <= 2:
            counts["within_two_hops"] += 1
        else:
            counts["over_two_hops"] += 1
    n = max(len(contexts), 1)
    return {"counts": counts, "proportions": {key: value / n for key, value in counts.items()}, "mean_lattice_distance": float(np.mean(distances)), "median_lattice_distance": float(np.median(distances))}


def run() -> dict[str, Any]:
    data = load_train_val()
    contexts = _contexts(data)
    cache = data["caches"]["val"]
    adjacency = _pair_statistics(contexts, cache)
    distance = _distance_audit(contexts, cache)
    metrics = {name: _terminal_actions(contexts, cache, name) for name in ("H1", "1-Hop Oracle", "2-Hop Oracle", "Full GT-margin Oracle", "AnyCorrect Oracle")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adjacency_statistics.json").write_text(json.dumps(adjacency, indent=2), encoding="utf-8")
    (OUT / "h1_to_oracle_distance.json").write_text(json.dumps(distance, indent=2), encoding="utf-8")
    (OUT / "local_oracle_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result = {
        "experiment_id": "REDUCED12_UTILITY_LANDSCAPE_LOCALITY",
        "population": {"val_moving_contexts": len(contexts)},
        "adjacency": adjacency,
        "h1_to_oracle_distance": distance,
        "local_oracle_metrics": metrics,
        "labels": list(LABELS),
        "protocol": {"utility": "GT ST-GCN true-class margin", "lattice": "4 radii x 8 azimuths; only actually legal candidates", "terminal": "real archived candidate skeleton represented by frozen ST-GCN logp"},
        "policy_test_used": False,
        "policy_train_used_for_model_training_only": True,
        "policy_val_used_for_evaluation_only": True,
        "future_candidate_observation_used_at_inference": False,
        "future_candidate_skeleton_used_as_train_target_or_terminal_eval_only": True,
        "gt_action_predicted_at_inference": False,
        "gt_action_used_for_supervision_or_posthoc_only": True,
        "scene_visibility_used_as_posthoc_only": True,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "frozen_stgcn_modified": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gap = metrics["Full GT-margin Oracle"]["accuracy"] - metrics["2-Hop Oracle"]["accuracy"]
    lines = ["# Reduced12 utility landscape locality", "", f"Moving Val contexts: {len(contexts)}", "", "## Adjacency", "", "| Pair set | Pairs | Pearson | Spearman | P(neighbor correct | center correct) | P(neighbor correct | center wrong) | Mean |margin diff| |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in adjacency.items():
        lines.append(f"| {name} | {item['pairs']} | {item['pearson']:.6f} | {item['spearman']:.6f} | {item['p_neighbor_correct_given_center_correct']:.6f} | {item['p_neighbor_correct_given_center_wrong']:.6f} | {item['mean_abs_margin_difference']:.6f} |")
    lines.extend(["", "## Local oracle ceiling", "", "| Selector | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, item in metrics.items():
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |")
    lines.extend(["", f"GT-margin-best distance: {distance['proportions']}", f"2-hop oracle to full GT-margin oracle accuracy gap: {gap * 100.0:.3f} pp.", "", "This is a Val-only oracle audit; no policy Test, training, or perception regeneration was used."])
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
