#!/usr/bin/env python3
"""Audit privileged multi-view fusion ceilings on reduced12 Val.

Only cached Train/Val artifacts are read.  Ground-truth labels are used to
select view sets for the privileged ceiling, while terminal predictions are
the fused frozen ST-GCN log-probabilities.  No model is trained or modified.
"""

from __future__ import annotations

import argparse
import itertools
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
    classification,
    candidate_logp,
    lattice_distance,
    legal_ids,
    load_train_val,
)


DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/multiview_fusion_ceiling"
FUSIONS = ("MeanLogP", "SumLogP")
BUDGETS = (1, 2, 3, 4)
HOPS = (1, 2, 3, 4)
BEAM_SIZE = 32


def _softmax(logp: Sequence[float]) -> np.ndarray:
    values = np.asarray(logp, dtype=np.float64)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def _margin(logp: np.ndarray, label: int) -> float:
    others = np.delete(np.asarray(logp, dtype=np.float64), int(label))
    return float(logp[int(label)] - np.max(others))


def _contexts(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join moving Stage-D rows with the all-legal Val true-evidence cache."""
    cache = data["caches"]["val"]
    contexts: list[dict[str, Any]] = []
    for row in data["stage_d_rows"]["val"]:
        index = int(row["cache_index"])
        candidates = sorted(legal_ids(cache, index))
        h1 = int(row["s1_viewpoint_id"])
        if not candidates or h1 not in candidates:
            raise ValueError(f"invalid moving candidate set: {row['episode_id']}")
        label = int(row["label_id"])
        evidence = {
            candidate: candidate_logp(cache, index, candidate)
            for candidate in candidates
        }
        contexts.append({
            "episode_id": str(row["episode_id"]),
            "index": index,
            "row": row,
            "label": label,
            "h1": h1,
            "candidates": candidates,
            "logp": evidence,
            "s0_logp": np.asarray(cache["current_logp"][index], dtype=np.float32),
        })
    return contexts


def _fused_logp(context: Mapping[str, Any], selected: Sequence[int], fusion: str) -> np.ndarray:
    values = np.stack([np.asarray(context["logp"][int(view)], dtype=np.float64) for view in selected])
    if fusion == "MeanLogP":
        return np.mean(values, axis=0)
    if fusion == "SumLogP":
        return np.sum(values, axis=0)
    raise ValueError(f"unsupported fusion: {fusion}")


def _selection_score(context: Mapping[str, Any], selected: Sequence[int], fusion: str) -> float:
    return _margin(_fused_logp(context, selected, fusion), int(context["label"]))


def _sort_states(states: Sequence[tuple[int, ...]], context: Mapping[str, Any], fusion: str) -> list[tuple[int, ...]]:
    return sorted(
        states,
        key=lambda state: (-_selection_score(context, state, fusion), tuple(state)),
    )


def _best_set(
    context: Mapping[str, Any], candidates: Sequence[int], budget: int, fusion: str,
) -> tuple[int, ...]:
    """Select at most ``budget`` candidates with a fixed exact/beam oracle."""
    ordered = tuple(sorted(int(value) for value in candidates))
    if not ordered:
        raise ValueError("cannot select from an empty candidate set")
    size = min(int(budget), len(ordered))
    best_single = max(
        ((value,) for value in ordered),
        key=lambda state: (_selection_score(context, state, fusion), tuple(-value for value in state)),
    )
    if size == 1:
        return best_single

    # B=2 is exact over all pairs plus the singleton choices.  For B=3/4,
    # preserve the exact singleton/pair candidates and use the fixed-width
    # beam only for larger sets.
    states: list[tuple[int, ...]] = [best_single]
    pairs = list(itertools.combinations(ordered, 2))
    states.extend(pairs)
    if size == 2:
        return _sort_states(states, context, fusion)[0]

    beam_states = pairs
    for depth in range(3, size + 1):
        expanded: list[tuple[int, ...]] = []
        for state in beam_states:
            last = state[-1]
            for value in ordered:
                if value > last:
                    expanded.append(state + (value,))
        beam_states = _sort_states(expanded, context, fusion)[:BEAM_SIZE]
        states.extend(beam_states)
    if not states:
        raise RuntimeError(f"beam produced no states for budget {size}")
    return _sort_states(states, context, fusion)[0]


def _selected_metrics(
    contexts: Sequence[Mapping[str, Any]], selections: Sequence[tuple[int, ...]], fusion: str,
) -> dict[str, Any]:
    labels = [int(context["label"]) for context in contexts]
    predictions: list[int] = []
    margins: list[float] = []
    gt_probabilities: list[float] = []
    entropies: list[float] = []
    for context, selected in zip(contexts, selections):
        fused = _fused_logp(context, selected, fusion)
        probabilities = _softmax(fused)
        predictions.append(int(np.argmax(fused)))
        margins.append(_margin(fused, int(context["label"])))
        gt_probabilities.append(float(probabilities[int(context["label"])]))
        entropies.append(float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))))
    result = classification(labels, predictions)
    result.update({
        "fusion": fusion,
        "budget": int(max(len(item) for item in selections)) if selections else 0,
        "mean_selected_view_count": float(np.mean([len(item) for item in selections])) if selections else 0.0,
        "mean_gt_probability": float(np.mean(gt_probabilities)) if gt_probabilities else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_gt_margin": float(np.mean(margins)) if margins else 0.0,
        "move_rate": 1.0,
        "stay_rate": 0.0,
    })
    return result


def _single_metrics(contexts: Sequence[Mapping[str, Any]], actions: Sequence[int], name: str) -> dict[str, Any]:
    labels = [int(context["label"]) for context in contexts]
    predictions: list[int] = []
    for context, action in zip(contexts, actions):
        if int(action) < 0:
            predictions.append(int(np.argmax(context["s0_logp"])))
        else:
            predictions.append(int(np.argmax(context["logp"][int(action)])))
    result = classification(labels, predictions)
    result.update({"method": name, "move_rate": float(np.mean(np.asarray(actions) >= 0)), "stay_rate": float(np.mean(np.asarray(actions) < 0))})
    return result


def _select_all(
    contexts: Sequence[Mapping[str, Any]], candidates_by_context: Sequence[Sequence[int]],
    fusion: str, budget: int,
) -> list[tuple[int, ...]]:
    return [_best_set(context, candidates, budget, fusion) for context, candidates in zip(contexts, candidates_by_context)]


def _oracle_sets(contexts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[tuple[int, ...]]]]:
    all_candidates = [context["candidates"] for context in contexts]
    output: dict[str, dict[str, list[tuple[int, ...]]]] = {}
    for fusion in FUSIONS:
        output[fusion] = {}
        for budget in BUDGETS:
            output[fusion][f"B{budget}"] = _select_all(contexts, all_candidates, fusion, budget)
    return output


def _hop_sets(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for hop in HOPS:
        allowed = [
            [candidate for candidate in context["candidates"] if lattice_distance(context["h1"], candidate) <= hop]
            for context in contexts
        ]
        for fusion in FUSIONS:
            for budget in (2, 3, 4):
                key = f"{fusion}-K{hop}-B{budget}"
                output[key] = _select_all(contexts, allowed, fusion, budget)
    return output


def _complementarity(
    contexts: Sequence[Mapping[str, Any]], selections: Mapping[str, Sequence[tuple[int, ...]]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total = len(contexts)
    all_single_wrong: list[bool] = []
    any_single_correct: list[bool] = []
    for context in contexts:
        correctness = [int(np.argmax(context["logp"][candidate])) == int(context["label"]) for candidate in context["candidates"]]
        all_single_wrong.append(not any(correctness))
        any_single_correct.append(any(correctness))
    for name, chosen in selections.items():
        fusion, budget_name = name.rsplit("-", 1)
        budget = int(budget_name[1:])
        fusion_correct = [int(np.argmax(_fused_logp(context, state, fusion))) == int(context["label"])
                          for context, state in zip(contexts, chosen)]
        rescue = [wrong and correct for wrong, correct in zip(all_single_wrong, fusion_correct)]
        harm = [had_correct and not correct for had_correct, correct in zip(any_single_correct, fusion_correct)]
        rows[name] = {
            "budget": budget,
            "fusion": fusion,
            "contexts": total,
            "all_single_wrong_contexts": int(sum(all_single_wrong)),
            "all_single_wrong_to_fusion_correct_count": int(sum(rescue)),
            "all_single_wrong_to_fusion_correct_fraction": float(sum(rescue) / total) if total else 0.0,
            "at_least_one_single_correct_to_fusion_wrong_count": int(sum(harm)),
            "at_least_one_single_correct_to_fusion_wrong_fraction": float(sum(harm) / total) if total else 0.0,
            "fusion_correct_count": int(sum(fusion_correct)),
        }
    return rows


def _belief_progression(
    contexts: Sequence[Mapping[str, Any]], oracle_sets: Mapping[str, Mapping[str, Sequence[tuple[int, ...]]]],
    best_fusion: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {"fusion": best_fusion, "budgets": {}}
    for budget in BUDGETS:
        selections = oracle_sets[best_fusion][f"B{budget}"]
        result = _selected_metrics(contexts, selections, best_fusion)
        output["budgets"][f"B{budget}"] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "mean_gt_probability": result["mean_gt_probability"],
            "mean_entropy": result["mean_entropy"],
            "mean_gt_margin": result["mean_gt_margin"],
        }
    return output


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _analysis(
    result: Mapping[str, Any], fusion_metrics: Mapping[str, Any],
    complementarity: Mapping[str, Any], khop: Mapping[str, Any],
) -> str:
    baseline = float(result["reference_metrics"]["BestSingle Oracle"]["accuracy"])
    b3 = max(float(fusion_metrics[f"{fusion}-B3"]["accuracy"]) for fusion in FUSIONS)
    b4 = max(float(fusion_metrics[f"{fusion}-B4"]["accuracy"]) for fusion in FUSIONS)
    unrestricted_b3 = b3
    unrestricted_b4 = b4
    k2_b2 = max(float(khop[f"{fusion}-K2-B2"]["accuracy"]) for fusion in FUSIONS)
    k3_b3 = max(float(khop[f"{fusion}-K3-B3"]["accuracy"]) for fusion in FUSIONS)
    k4_b4 = max(float(khop[f"{fusion}-K4-B4"]["accuracy"]) for fusion in FUSIONS)
    b3_delta = b3 - baseline
    b4_delta = b4 - baseline
    all_wrong = int(next(iter(complementarity.values()))["all_single_wrong_contexts"])
    b4_rescue = max(int(complementarity[f"{fusion}-B4"]["all_single_wrong_to_fusion_correct_count"]) for fusion in FUSIONS)
    b4_harm = max(int(complementarity[f"{fusion}-B4"]["at_least_one_single_correct_to_fusion_wrong_count"]) for fusion in FUSIONS)
    threshold = "supports Sequential Multi-view Active Recognition" if b3_delta >= 0.05 or b4_delta >= 0.05 else "mixed evidence" if max(b3_delta, b4_delta) >= 0.02 else "sequential policy headroom is small"
    return "\n".join([
        "# Reduced12 Multi-view Fusion Ceiling Audit",
        "",
        f"Moving Val contexts: {result['population']['contexts']}; legal candidate samples: {result['population']['candidate_samples']}.",
        "Policy Test was not read.  This is a privileged GT-label-conditioned ceiling, not a deployable method.",
        "",
        "## Reference and unrestricted fusion",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        *[f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |" for name, item in result["reference_metrics"].items()],
        *[f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |" for name, item in fusion_metrics.items()],
        "",
        "For a fixed cardinality, MeanLogP and SumLogP induce the same margin ordering; with the requested at-most budget, their different scales can select different cardinalities, so both results are retained.",
        "MeanFeature was skipped because the frozen Val cache contains no candidate-level ST-GCN feature tensor or reusable classifier-head interface.",
        "",
        "## Complementarity",
        "",
        f"Contexts where every single legal view is wrong: {all_wrong}.",
        f"Best B4 fusion rescues {b4_rescue} of these contexts (fraction {b4_rescue / max(result['population']['contexts'], 1):.6f}).",
        f"At B4, contexts with at least one correct single view but a wrong fused prediction: {b4_harm} (see all budgets in complementarity.json).",
        "",
        "## K-hop fusion",
        "",
        f"Best K2-B2 Accuracy: {k2_b2:.6f}; best K3-B3 Accuracy: {k3_b3:.6f}; unrestricted B3 Accuracy: {unrestricted_b3:.6f}.",
        f"Best K4-B4 Accuracy: {k4_b4:.6f}; unrestricted B4 Accuracy: {unrestricted_b4:.6f}.",
        "K-hop values are privileged reachability ceilings from FrozenStageCv0 H1, not learned policies.",
        "",
        "## Scientific answers",
        "",
        f"1. B=3/B=4 exceed BestSingle by {b3_delta * 100:.2f}/{b4_delta * 100:.2f} percentage points; the fixed decision category is **{threshold}**.",
        "2. Fusion-correct cases with all single views wrong are the direct evidence for multi-view complementarity; their count is reported above and in complementarity.json.",
        "3. K3/K4 capture the fraction shown above of the unrestricted B3/B4 ceiling; K-hop fusion does not assume access outside the lattice radius.",
        "4. No new policy, recognizer, fusion network or Test evaluation was started automatically.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false",
        "new_rgb_generated=false",
        "new_skeleton_generated=false",
        "frozen_stgcn_modified=false",
        "gt_label_used_for_oracle_only=true",
        "deployable=false",
        "```",
        "",
    ])


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    data = load_train_val()
    contexts = _contexts(data)
    oracle_sets = _oracle_sets(contexts)
    fusion_metrics: dict[str, Any] = {}
    for fusion in FUSIONS:
        for budget in BUDGETS:
            key = f"{fusion}-B{budget}"
            fusion_metrics[key] = _selected_metrics(contexts, oracle_sets[fusion][f"B{budget}"], fusion)

    h1_actions = [int(context["h1"]) for context in contexts]
    s0_actions = [-1] * len(contexts)
    reference_metrics = {
        "S0-only": _single_metrics(contexts, s0_actions, "S0-only"),
        "FrozenStageCv0": _single_metrics(contexts, h1_actions, "FrozenStageCv0"),
        "BestSingle Oracle": fusion_metrics["MeanLogP-B1"],
        "AnyCorrect-single-view Oracle": fusion_metrics["MeanLogP-B1"],
    }

    khop_sets = _hop_sets(contexts)
    khop_metrics: dict[str, Any] = {}
    for key, selections in khop_sets.items():
        fusion = key.split("-", 1)[0]
        khop_metrics[key] = _selected_metrics(contexts, selections, fusion)

    best_fusion = max(
        FUSIONS,
        key=lambda fusion: (float(fusion_metrics[f"{fusion}-B4"]["accuracy"]), -FUSIONS.index(fusion)),
    )
    complementarity = _complementarity(
        contexts,
        {key: oracle_sets[fusion][f"B{budget}"] for fusion in FUSIONS for budget in (2, 3, 4) for key in [f"{fusion}-B{budget}"]},
    )
    belief_progression = _belief_progression(contexts, oracle_sets, best_fusion)

    candidate_samples = int(sum(len(context["candidates"]) for context in contexts))
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_MULTIVIEW_FUSION_CEILING",
        "population": {"split": "val_moving", "contexts": len(contexts), "candidate_samples": candidate_samples},
        "labels": list(LABELS),
        "reference_metrics": reference_metrics,
        "fusion_metrics": fusion_metrics,
        "best_fusion": best_fusion,
        "mean_feature": {"status": "skipped", "reason": "no candidate-level ST-GCN feature tensor or reusable frozen classifier-head interface in existing Val artifacts"},
        "complementarity": complementarity,
        "khop_fusion": khop_metrics,
        "belief_progression": belief_progression,
        "protocol": {
            "candidate_set": "existing legal candidates only",
            "oracle_objective": "maximize fused GT-class margin",
            "B1_B2": "exact enumeration",
            "B3_B4": f"fixed beam oracle, beam_size={BEAM_SIZE}",
            "khop_start": "FrozenStageCv0 H1 s1 viewpoint",
            "fusion_definitions": {
                "MeanLogP": "mean candidate log-probability vectors",
                "SumLogP": "sum candidate log-probability vectors",
            },
            "terminal_prediction": "argmax fused frozen ST-GCN log-probability",
        },
        "policy_test_used": False,
        "training_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "new_dino_generated": False,
        "frozen_stgcn_modified": False,
        "gt_label_used_for_oracle_only": True,
        "deployable": False,
    }
    _write(output_dir / "fusion_metrics.json", fusion_metrics)
    _write(output_dir / "complementarity.json", complementarity)
    _write(output_dir / "khop_fusion.json", khop_metrics)
    _write(output_dir / "belief_progression.json", belief_progression)
    _write(output_dir / "result.json", result)
    (output_dir / "analysis.md").write_text(_analysis(result, fusion_metrics, complementarity, khop_metrics), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
