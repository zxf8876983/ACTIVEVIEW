#!/usr/bin/env python3
"""Val-only audit of spatially diverse candidate shortlists."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    candidate_logp,
    classification,
    gt_margin,
    lattice_distance,
    legal_ids,
    load_train_val,
)

OUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/diverse_topk_proposal_audit"


def _contexts(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in data["stage_d_rows"]["val"]:
        prediction = row["stage_c_prediction"]
        index = int(row["cache_index"])
        ids = legal_ids(data["caches"]["val"], index)
        scores = {int(v): float(s) for v, s in zip(prediction["candidate_viewpoint_ids"], prediction["predicted_utilities"])}
        if set(ids) != set(scores):
            raise ValueError(f"proposal candidate mismatch: {row['episode_id']}")
        output.append({"row": row, "index": index, "label": int(row["label_id"]), "ids": ids, "scores": scores})
    return output


def _ordinary(context: Mapping[str, Any], k: int) -> list[int]:
    return sorted(context["ids"], key=lambda candidate: (-context["scores"][candidate], candidate))[:k]


def _azimuth_diverse(context: Mapping[str, Any], k: int) -> list[int]:
    ordered = _ordinary(context, len(context["ids"]))
    selected: list[int] = []
    bins: set[int] = set()
    for candidate in ordered:
        bin_id = int(candidate) % 8
        if bin_id in bins:
            continue
        selected.append(candidate)
        bins.add(bin_id)
        if len(selected) == k:
            return selected
    for candidate in ordered:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == k:
            break
    return selected


def _lattice_diverse(context: Mapping[str, Any], k: int) -> list[int]:
    ordered = _ordinary(context, len(context["ids"]))
    if not ordered:
        return []
    selected = [ordered[0]]
    for candidate in ordered[1:]:
        if all(lattice_distance(candidate, chosen) >= 2 for chosen in selected):
            selected.append(candidate)
        if len(selected) == k:
            return selected
    for candidate in ordered[1:]:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == k:
            break
    return selected


def _oracle_metric(contexts: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], shortlists: Sequence[Sequence[int]], name: str) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    coverage: list[bool] = []
    for context, shortlist in zip(contexts, shortlists):
        index, label = int(context["index"]), int(context["label"])
        labels.append(label)
        candidate_predictions = [int(np.argmax(candidate_logp(cache, index, candidate))) for candidate in shortlist]
        coverage.append(label in candidate_predictions)
        options = [(gt_margin(np.asarray(context["row"]["s0_feature"][256:268]), label), -1)]
        options.extend((gt_margin(candidate_logp(cache, index, candidate), label), candidate) for candidate in shortlist)
        selected = max(options, key=lambda item: (item[0], -item[1]))[1]
        if selected == -1:
            predictions.append(int(np.argmax(context["row"]["s0_feature"][256:268])))
        else:
            predictions.append(int(np.argmax(candidate_logp(cache, index, selected))))
    result = classification(labels, predictions)
    result["shortlist"] = name
    result["any_correct_coverage"] = float(np.mean(coverage))
    return result


def _diversity(contexts: Sequence[Mapping[str, Any]], shortlists: Sequence[Sequence[int]]) -> dict[str, float]:
    azimuths: list[float] = []
    lattice: list[float] = []
    for shortlist in shortlists:
        for index, left in enumerate(shortlist):
            for right in shortlist[index + 1 :]:
                delta = abs((int(left) % 8) - (int(right) % 8))
                azimuths.append(float(min(delta, 8 - delta) * 45.0))
                lattice.append(float(lattice_distance(left, right)))
    return {"mean_pairwise_azimuth_separation_deg": float(np.mean(azimuths)) if azimuths else 0.0, "mean_pairwise_lattice_distance": float(np.mean(lattice)) if lattice else 0.0}


def run() -> dict[str, Any]:
    data = load_train_val()
    contexts = _contexts(data)
    cache = data["caches"]["val"]
    metrics: dict[str, Any] = {}
    for family, builder in (("Ordinary", _ordinary), ("Azimuth-Diverse", _azimuth_diverse), ("Lattice-Diverse", _lattice_diverse)):
        for k in (1, 3, 5):
            shortlists = [builder(context, k) for context in contexts]
            name = f"{family} Top{k}"
            item = _oracle_metric(contexts, cache, shortlists, name)
            item.update(_diversity(contexts, shortlists))
            metrics[name] = item
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": "REDUCED12_DIVERSE_TOPK_PROPOSAL_AUDIT",
        "population": {"val_moving_contexts": len(contexts)},
        "metrics": metrics,
        "baseline_score": "FrozenStageCv0 predicted_utilities from Stage-C val_predictions.jsonl",
        "labels": list(LABELS),
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
    lines = ["# Reduced12 diverse Top-K proposal audit", "", f"Moving Val contexts: {len(contexts)}", "", "| Shortlist | AnyCorrect coverage | Oracle Acc | Oracle Macro-F1 | Mean azimuth separation | Mean lattice distance |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in metrics.items():
        lines.append(f"| {name} | {item['any_correct_coverage']:.6f} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mean_pairwise_azimuth_separation_deg']:.3f} | {item['mean_pairwise_lattice_distance']:.3f} |")
    ordinary = metrics["Ordinary Top3"]["any_correct_coverage"]
    best_diverse = max(metrics[name]["any_correct_coverage"] for name in ("Azimuth-Diverse Top3", "Lattice-Diverse Top3"))
    lines.extend(["", f"Best diverse Top3 coverage delta vs ordinary: {(best_diverse - ordinary) * 100.0:+.3f} pp.", "No GT utility was used for shortlist ordering; true evidence is used only for oracle-within-shortlist evaluation.", "", "policy_test_used=false; no training or perception regeneration."])
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
