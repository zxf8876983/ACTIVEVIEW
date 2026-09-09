#!/usr/bin/env python3
"""Val-only combination of top-down LOS filtering and FrozenStageCv0 ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.analyze_reduced12_topdown_los_nbv import (
    DEFAULT_POLICY_ROOT,
    DEFAULT_TRUE_CACHE,
    LABELS,
    _collect_bundles,
    _load_npz,
    _metrics,
)

NUM_CLASSES = 12
SEED = 42
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/topdown_los_filter_frozen"
)


def _any_correct_prediction(
    s0_prediction: int,
    candidate_predictions: np.ndarray,
    label: int,
) -> tuple[int, bool]:
    if s0_prediction == label:
        return s0_prediction, False
    if np.any(candidate_predictions == label):
        return label, True
    return s0_prediction, False


def evaluate(
    data_root: Path,
    *,
    scene_root: Path,
    stage_c_path: Path,
    stage_c_predictions_path: Path,
    stage_d_path: Path,
    true_cache_path: Path,
) -> dict[str, Any]:
    stage_c_rows = load_jsonl(stage_c_path)
    stage_c_predictions = load_jsonl(stage_c_predictions_path)
    moving_rows = load_jsonl(stage_d_path)
    if not all(path.as_posix().endswith("/val.jsonl") for path in (stage_c_path, stage_d_path)):
        raise ValueError("this diagnostic is Val-only")
    if len(stage_c_rows) != len(stage_c_predictions):
        raise ValueError("Stage-C feature/prediction lengths differ")
    prediction_by_id = {str(row["episode_id"]): row for row in stage_c_predictions}
    if len(prediction_by_id) != len(stage_c_predictions):
        raise ValueError("duplicate Stage-C prediction episode_id")
    cache = _load_npz(true_cache_path)
    required = {"current_logp", "candidate_logp", "candidate_mask", "candidate_ids", "candidate_geodesic"}
    if not required.issubset(cache):
        raise ValueError(f"true cache missing keys: {sorted(required - set(cache))}")
    bundles, scene_grid = _collect_bundles(
        data_root, stage_c_rows, moving_rows, cache, scene_root
    )
    stage_c_index = {str(row["episode_id"]): i for i, row in enumerate(stage_c_rows)}
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    predictions = {
        name: np.empty(labels.size, dtype=np.int64)
        for name in (
            "S0-only", "FrozenStageCv0", "TopDown-LOS",
            "TopDownLOS-Filter+Frozen", "AnyCorrect Oracle",
        )
    }
    moved = {name: np.zeros(labels.size, dtype=bool) for name in predictions}
    has_visible: list[bool] = []
    filtered_out: list[bool] = []
    frozen_selected_los: list[int] = []
    rescue_count = harm_count = 0
    rescue_denominator = harm_denominator = 0
    frozen_los1_correct = frozen_los1_total = 0
    frozen_los0_correct = frozen_los0_total = 0
    rng = np.random.default_rng(SEED)
    # Random is not a reported method in this combination table, but advancing
    # the same fixed stream keeps any future extension deterministic.
    del rng

    for index, (row, bundle) in enumerate(zip(moving_rows, bundles)):
        episode_id = str(row["episode_id"])
        cache_index = int(bundle["cache_index"])
        stage_prediction = prediction_by_id[episode_id]
        ids = np.asarray(bundle["candidate_ids"], dtype=np.int64)
        slots = np.asarray(bundle["slots"], dtype=np.int64)
        candidate_logp = np.asarray(cache["candidate_logp"][cache_index, slots], dtype=np.float64)
        candidate_predictions = np.argmax(candidate_logp, axis=1)
        current_logp = np.asarray(cache["current_logp"][cache_index], dtype=np.float64)
        s0_prediction = int(np.argmax(current_logp))
        label = int(labels[index])
        predictions["S0-only"][index] = s0_prediction

        s1_logp = np.asarray(row["s1_feature"], dtype=np.float64)[256:256 + NUM_CLASSES]
        if s1_logp.shape != (NUM_CLASSES,):
            raise ValueError(f"invalid s1 logp schema for {episode_id}")
        frozen_prediction = int(np.argmax(s1_logp))
        predictions["FrozenStageCv0"][index] = frozen_prediction
        moved["FrozenStageCv0"][index] = True

        los = np.asarray(bundle["candidate_los"], dtype=np.int64)
        geodesic = np.asarray(bundle["candidate_geodesic"], dtype=np.float64)
        visible_indices = np.flatnonzero(los == 1)
        has_visible.append(bool(visible_indices.size))
        original_id = int(row["proposal_rank_1_id"])
        original_matches = np.flatnonzero(ids == original_id)
        if original_matches.size != 1:
            raise ValueError(f"FrozenStageCv0 action is not a legal candidate: {episode_id}")
        original_index = int(original_matches[0])
        frozen_selected_los.append(int(los[original_index]))
        if los[original_index] == 1:
            frozen_los1_total += 1
            frozen_los1_correct += int(frozen_prediction == label)
        else:
            frozen_los0_total += 1
            frozen_los0_correct += int(frozen_prediction == label)

        candidate_scores = np.asarray(stage_prediction["predicted_utilities"], dtype=np.float64)
        prediction_ids = np.asarray(stage_prediction["candidate_viewpoint_ids"], dtype=np.int64)
        if candidate_scores.shape != prediction_ids.shape or set(prediction_ids.tolist()) != set(ids.tolist()):
            raise ValueError(f"FrozenStageCv0 score/candidate identity mismatch: {episode_id}")
        score_by_id = {int(viewpoint): float(score) for viewpoint, score in zip(prediction_ids, candidate_scores)}
        if visible_indices.size:
            filtered_index = min(
                visible_indices.tolist(),
                key=lambda j: (-score_by_id[int(ids[j])], float(geodesic[j]), int(ids[j])),
            )
            predictions["TopDownLOS-Filter+Frozen"][index] = int(candidate_predictions[filtered_index])
            moved["TopDownLOS-Filter+Frozen"][index] = True
            filtered_out.append(original_index not in set(visible_indices.tolist()))
        else:
            predictions["TopDownLOS-Filter+Frozen"][index] = frozen_prediction
            moved["TopDownLOS-Filter+Frozen"][index] = True
            filtered_out.append(False)

        # The original top-down LOS selector is candidate-only, with geodesic
        # and viewpoint-id tie breaks, matching the preceding diagnostic.
        topdown_index = min(
            range(ids.size),
            key=lambda j: (-int(los[j]), float(geodesic[j]), int(ids[j])),
        )
        predictions["TopDown-LOS"][index] = int(candidate_predictions[topdown_index])
        moved["TopDown-LOS"][index] = True

        any_prediction, any_moved = _any_correct_prediction(
            s0_prediction, candidate_predictions, label
        )
        predictions["AnyCorrect Oracle"][index] = any_prediction
        moved["AnyCorrect Oracle"][index] = any_moved

        if filtered_out[-1]:
            original_correct = frozen_prediction == label
            filtered_correct = predictions["TopDownLOS-Filter+Frozen"][index] == label
            rescue_count += int(not original_correct and filtered_correct)
            harm_count += int(original_correct and not filtered_correct)
            rescue_denominator += int(not original_correct)
            harm_denominator += int(original_correct)

    methods = {name: _metrics(prediction, labels) for name, prediction in predictions.items()}
    move_rates = {name: float(np.mean(values)) for name, values in moved.items()}
    frozen = methods["FrozenStageCv0"]
    filtered = methods["TopDownLOS-Filter+Frozen"]
    diagnostics = {
        "contexts_with_los1_candidate": {
            "count": int(np.sum(has_visible)),
            "rate": float(np.mean(has_visible)),
        },
        "frozen_selection_filtered_out": {
            "count": int(np.sum(filtered_out)),
            "rate": float(np.mean(filtered_out)),
        },
        "filtered_context_outcomes": {
            "rescue_count": rescue_count,
            "harm_count": harm_count,
            "net_rescue_minus_harm": rescue_count - harm_count,
            "rescue_rate_among_frozen_wrong": float(rescue_count / rescue_denominator) if rescue_denominator else 0.0,
            "harm_rate_among_frozen_correct": float(harm_count / harm_denominator) if harm_denominator else 0.0,
        },
        "frozen_selected_correctness_by_los": {
            "los1_count": frozen_los1_total,
            "los1_correct": frozen_los1_correct,
            "p_correct_given_los1": float(frozen_los1_correct / frozen_los1_total) if frozen_los1_total else 0.0,
            "los0_count": frozen_los0_total,
            "los0_correct": frozen_los0_correct,
            "p_correct_given_los0": float(frozen_los0_correct / frozen_los0_total) if frozen_los0_total else 0.0,
        },
    }
    result = {
        "experiment_id": "REDUCED12_TOPDOWN_LOS_FILTER_FROZEN",
        "status": "COMPLETED",
        "population": {"val_moving_contexts": int(labels.size), "scene_count": len(scene_grid)},
        "labels": list(LABELS),
        "methods": methods,
        "move_rates": move_rates,
        "deltas_vs_frozen": {
            "TopDown-LOS": {
                "accuracy_pp": 100.0 * (methods["TopDown-LOS"]["accuracy"] - frozen["accuracy"]),
                "macro_f1_pp": 100.0 * (methods["TopDown-LOS"]["macro_f1"] - frozen["macro_f1"]),
            },
            "TopDownLOS-Filter+Frozen": {
                "accuracy_pp": 100.0 * (filtered["accuracy"] - frozen["accuracy"]),
                "macro_f1_pp": 100.0 * (filtered["macro_f1"] - frozen["macro_f1"]),
            },
        },
        "diagnostics": diagnostics,
        "scene_grid": scene_grid,
        "protocol": {
            "filter": "LOS=1 candidates only when non-empty",
            "frozen_score": "existing Stage-C FrozenStageCv0 predicted_utilities",
            "tie_break": "candidate geodesic then viewpoint id",
            "terminal": "selected real archived skeleton through frozen reduced12 ST-GCN cache",
            "seed": SEED,
        },
        "leakage_flags": {
            "test_used": False,
            "training_used": False,
            "habitat_gt_geometry_used_for_oracle_diagnostic": True,
            "future_candidate_skeleton_used_only_for_terminal_evaluation": True,
        },
    }
    return result


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    frozen = result["methods"]["FrozenStageCv0"]
    topdown = result["methods"]["TopDown-LOS"]
    filtered = result["methods"]["TopDownLOS-Filter+Frozen"]
    diag = result["diagnostics"]
    gain = result["deltas_vs_frozen"]["TopDownLOS-Filter+Frozen"]
    lines = [
        "# Reduced12 Top-down LOS Filter + FrozenStageCv0",
        "",
        "Val Moving only. The binary Habitat top-down LOS is a privileged geometry rejector; FrozenStageCv0's existing Stage-C predicted utility remains the only fine-ranking score. No model was trained and no Test/perception artifact was read or generated.",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("S0-only", "FrozenStageCv0", "TopDown-LOS", "TopDownLOS-Filter+Frozen", "AnyCorrect Oracle"):
        metric = result["methods"][name]
        lines.append(
            f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | "
            f"{100.0 * (metric['accuracy'] - frozen['accuracy']):+.3f} | "
            f"{100.0 * (metric['macro_f1'] - frozen['macro_f1']):+.3f} | "
            f"{result['move_rates'][name]:.6f} |"
        )
    lines.extend([
        "",
        f"Contexts with at least one LOS=1 candidate: {diag['contexts_with_los1_candidate']['rate']:.6f}; Frozen selections filtered out: {diag['frozen_selection_filtered_out']['rate']:.6f}.",
        f"Among filtered contexts, rescue={diag['filtered_context_outcomes']['rescue_count']}, harm={diag['filtered_context_outcomes']['harm_count']}, net={diag['filtered_context_outcomes']['net_rescue_minus_harm']}.",
        f"Frozen selected correctness: P(correct|LOS=1)={diag['frozen_selected_correctness_by_los']['p_correct_given_los1']:.6f}; P(correct|LOS=0)={diag['frozen_selected_correctness_by_los']['p_correct_given_los0']:.6f}.",
        "",
        "## Scientific judgment",
        "",
    ])
    if gain["accuracy_pp"] >= 1.0 or gain["macro_f1_pp"] >= 1.0:
        lines.append(f"The LOS filter materially improves FrozenStageCv0 ({gain['accuracy_pp']:+.3f} pp Accuracy, {gain['macro_f1_pp']:+.3f} pp Macro-F1), supporting a coarse-geometry rejector plus HAR fine-ranking route.")
    else:
        lines.append(f"The LOS filter does not materially exceed FrozenStageCv0 ({gain['accuracy_pp']:+.3f} pp Accuracy, {gain['macro_f1_pp']:+.3f} pp Macro-F1; threshold for material gain is 1 pp). Binary LOS is therefore not yet validated as a rejector; the next diagnostic should use multi-height/3D joint visibility rather than tune thresholds or weights.")
    if diag["frozen_selected_correctness_by_los"]["p_correct_given_los1"] > diag["frozen_selected_correctness_by_los"]["p_correct_given_los0"]:
        lines.append("LOS and Frozen ranking show some complementarity at the candidate level, but the end-to-end combination determines whether that signal is useful.")
    else:
        lines.append("LOS is not complementary to Frozen selection correctness in this split.")
    lines.extend([
        "",
        "`test_used=false`, `training_used=false`, `habitat_gt_geometry_used_for_oracle_diagnostic=true`, and `future_candidate_skeleton_used_only_for_terminal_evaluation=true`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--stage-c", type=Path, default=None)
    parser.add_argument("--stage-c-predictions", type=Path, default=None)
    parser.add_argument("--stage-d", type=Path, default=None)
    parser.add_argument("--true-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    policy_root = data_root / DEFAULT_POLICY_ROOT
    stage_c_path = (args.stage_c or policy_root / "stage_c/features/val.jsonl").resolve()
    stage_c_predictions_path = (args.stage_c_predictions or policy_root / "stage_c/predictions/val_predictions.jsonl").resolve()
    stage_d_path = (args.stage_d or policy_root / "stage_d/features/val.jsonl").resolve()
    true_cache_path = (args.true_cache or data_root / DEFAULT_TRUE_CACHE).resolve()
    result = evaluate(
        data_root,
        scene_root=args.scene_root.resolve(),
        stage_c_path=stage_c_path,
        stage_c_predictions_path=stage_c_predictions_path,
        stage_d_path=stage_d_path,
        true_cache_path=true_cache_path,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(result["diagnostics"], indent=2) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
