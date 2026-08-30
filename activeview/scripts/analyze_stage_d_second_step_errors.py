#!/usr/bin/env python3
"""Prepare/run the Val-only EXP016 second-step error decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_error_decomposition import (
    build_exp016_variant_trajectories,
    exp016_decision_diagnostics,
    summarize_variant_rows,
    validate_exp016_episode_alignment,
    validate_exp016_split,
)
from activeview.active_view.stage_d_evaluation import (
    build_fixed_first_oracle,
    build_stage_d_trajectories,
)
from activeview.active_view.utility_label_builder import file_sha256


REFERENCE_METRICS = {
    "EXP014": {
        "accuracy": 0.6582540930864375,
        "macro_f1": 0.6101526052247462,
        "mean_regret": 1.4224626188609946,
        "p90_regret": 5.515662515163418,
        "headroom_capture": 0.7833127000367449,
    },
    "FixedFirstSecondStepOracle": {
        "accuracy": 0.7715021091013083,
        "macro_f1": 0.725081382409528,
        "mean_regret": 0.5862035541860805,
        "p90_regret": 1.6999013550579545,
        "headroom_capture": 0.8908871486261597,
    },
}


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _assert_val_rows(rows: Sequence[Mapping[str, Any]], name: str) -> None:
    # EXP014's second-step prediction artifact predates the split field.  Its
    # Val scope is established by the explicit input path and the exact
    # Stage-B/v0 episode-ID alignment check below.  Explicit non-Val labels
    # remain rejected.
    invalid = sorted(
        {
            str(row["policy_split"])
            for row in rows
            if "policy_split" in row and str(row["policy_split"]) != "val"
        }
    )
    if invalid:
        raise ValueError(f"{name} contains non-Val rows: {invalid}")


def _metric_row(name: str, summary: Mapping[str, Any], gate: str, candidate: str) -> dict[str, Any]:
    regret = summary["decision_regret"]
    headroom = summary["positive_headroom_capture"]
    movement = summary["movement"]
    return {
        "variant": name,
        "gate": gate,
        "candidate": candidate,
        "accuracy": float(summary["recognition"]["accuracy"]),
        "macro_f1": float(summary["recognition"]["macro_f1"]),
        "mean_regret": float(regret["mean"]),
        "median_regret": float(regret["median"]),
        "p90_regret": float(regret["p90"]),
        "headroom_capture": float(headroom["aggregate_positive_clipped_ratio"]),
        "average_moves": float(movement["average_moves"]),
        "mean_geodesic_cost_m": float(movement["trajectory_geodesic_cost_m"]["mean"]),
    }


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def _reference_checks(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for method, references in REFERENCE_METRICS.items():
        summary = metrics[method]
        for key, expected in references.items():
            if key == "mean_regret":
                actual = float(summary["decision_regret"]["mean"])
            elif key == "p90_regret":
                actual = float(summary["decision_regret"]["p90"])
            elif key == "headroom_capture":
                actual = float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"])
            else:
                actual = float(summary["recognition"][key])
            checks[f"{method}.{key}"] = {
                "actual": actual,
                "reference": expected,
                "within_abs_tolerance_1e-5": abs(actual - expected) <= 1e-5,
            }
    return checks


def _trajectory_groups(
    trajectories: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
) -> dict[str, Any]:
    v0 = {str(row["episode_id"]): row for row in v0_prediction_rows}
    groups: dict[str, list[Mapping[str, Any]]] = {
        "A_v0_stay": [],
        "B_v0_move_learned_second_stay": [],
        "C_v0_move_learned_second_move": [],
    }
    for row in trajectories:
        first = v0[str(row["episode_id"])]
        if bool(first["predicted_stays"]):
            groups["A_v0_stay"].append(row)
        elif int(row["moves"]) == 1:
            groups["B_v0_move_learned_second_stay"].append(row)
        else:
            groups["C_v0_move_learned_second_move"].append(row)
    return {
        name: {
            "count": len(rows),
            "accuracy": float(summarize_variant_rows(rows, categories)["recognition"]["accuracy"]),
            "mean_regret": float(summarize_variant_rows(rows, categories)["decision_regret"]["mean"]),
        }
        for name, rows in groups.items()
    }


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    exp014_predictions: Path,
    v0_predictions: Path,
    label_mapping: Path,
    output_path: Path,
    split: str = "val",
) -> dict[str, Any]:
    """Run EXP016's offline decomposition for the explicitly allowed split."""
    validate_exp016_split(split)
    cache_summary_path = cache_root / "stage_d_feature_summary.json"
    cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    cache_rows = load_jsonl(Path(cache_summary["feature_files"]["val"]))
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_rows = load_jsonl(v0_predictions)
    exp014_rows = load_jsonl(exp014_predictions)
    _assert_val_rows(stage_b_rows, "Stage B Val utility")
    _assert_val_rows(v0_rows, "Stage C-v0 Val predictions")
    _assert_val_rows(cache_rows, "Stage D Val cache")
    _assert_val_rows(exp014_rows, "EXP014 Val predictions")
    alignment = validate_exp016_episode_alignment(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows,
    )

    categories = _categories(label_mapping)
    exp014_trajectories = build_stage_d_trajectories(stage_b_rows, v0_rows, cache_rows, exp014_rows)
    learned_rows, learned_counters = build_exp016_variant_trajectories(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows,
        gate="learned",
        candidate="learned",
    )
    if len(learned_rows) != len(exp014_trajectories):
        raise ValueError("EXP016 learned+learned trajectory count differs from EXP014")
    for expected, actual in zip(exp014_trajectories, learned_rows):
        if (expected["selected_viewpoint_id"], expected["moves"]) != (actual["selected_viewpoint_id"], actual["moves"]):
            raise ValueError(f"EXP014 action logic mismatch: {expected['episode_id']}")

    oracle_gate_rows, oracle_gate_counters = build_exp016_variant_trajectories(
        stage_b_rows=stage_b_rows, v0_prediction_rows=v0_rows, cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows, gate="oracle", candidate="learned",
    )
    oracle_candidate_rows, oracle_candidate_counters = build_exp016_variant_trajectories(
        stage_b_rows=stage_b_rows, v0_prediction_rows=v0_rows, cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows, gate="learned", candidate="oracle",
    )
    fixed_first_rows = build_fixed_first_oracle(stage_b_rows, v0_rows, cache_rows)
    method_rows = {
        "EXP014": exp014_trajectories,
        "OracleGate_LearnedCandidate": oracle_gate_rows,
        "LearnedGate_OracleCandidate": oracle_candidate_rows,
        "FixedFirstSecondStepOracle": fixed_first_rows,
    }
    raw_metrics = {name: summarize_variant_rows(rows, categories) for name, rows in method_rows.items()}
    table = [
        _metric_row("EXP014", raw_metrics["EXP014"], "learned", "learned"),
        _metric_row("OracleGate + LearnedCandidate", raw_metrics["OracleGate_LearnedCandidate"], "oracle", "learned"),
        _metric_row("LearnedGate + OracleCandidate", raw_metrics["LearnedGate_OracleCandidate"], "learned", "oracle"),
        _metric_row("Fixed-first Second-Step Oracle", raw_metrics["FixedFirstSecondStepOracle"], "oracle", "oracle"),
    ]
    exp = table[0]
    gate = table[1]
    candidate = table[2]
    oracle = table[3]
    joint_accuracy_gap = oracle["accuracy"] - exp["accuracy"]
    joint_regret_gap = exp["mean_regret"] - oracle["mean_regret"]
    headroom_decomposition = {
        "accuracy": {
            "delta_gate": gate["accuracy"] - exp["accuracy"],
            "delta_candidate": candidate["accuracy"] - exp["accuracy"],
            "delta_joint": joint_accuracy_gap,
            "gate_recovery_fraction": _safe_fraction(gate["accuracy"] - exp["accuracy"], joint_accuracy_gap),
            "candidate_recovery_fraction": _safe_fraction(candidate["accuracy"] - exp["accuracy"], joint_accuracy_gap),
            "descriptive_interaction": joint_accuracy_gap - (gate["accuracy"] - exp["accuracy"]) - (candidate["accuracy"] - exp["accuracy"]),
        },
        "mean_regret": {
            "total_gap": joint_regret_gap,
            "gate_reduction": exp["mean_regret"] - gate["mean_regret"],
            "candidate_reduction": exp["mean_regret"] - candidate["mean_regret"],
            "gate_recovery_fraction": _safe_fraction(exp["mean_regret"] - gate["mean_regret"], joint_regret_gap),
            "candidate_recovery_fraction": _safe_fraction(exp["mean_regret"] - candidate["mean_regret"], joint_regret_gap),
        },
    }
    reference_checks = _reference_checks(raw_metrics)
    if not all(item["within_abs_tolerance_1e-5"] for item in reference_checks.values()):
        raise ValueError("Frozen EXP014 or fixed-first oracle reference mismatch")
    result = {
        "experiment_id": "EXP016",
        "experiment_name": "second_step_error_decomposition",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": False,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "episode_count": len(stage_b_rows),
        "v0_move_eligible_episode_count": sum(not bool(row["predicted_stays"]) for row in v0_rows),
        "source_episode_counts": {
            "stage_b_val": len(stage_b_rows),
            "stage_c_v0_val_predictions": len(v0_rows),
            "stage_d_cache_val": len(cache_rows),
            "exp014_val_predictions": len(exp014_rows),
        },
        "episode_alignment": alignment,
        "policies": {
            "EXP014": "learned gate + learned candidate; corrected EXP014 decision logic",
            "OracleGate_LearnedCandidate": "oracle true-U2 Stay/Move gate + learned p2/p3 candidate",
            "LearnedGate_OracleCandidate": "learned Stay/Move gate + oracle true-U2 p2/p3 candidate",
            "FixedFirstSecondStepOracle": "fixed v0 first decision + argmax(Stay=0,true U2 p2,true U2 p3)",
        },
        "metrics_table": table,
        "headroom_decomposition": headroom_decomposition,
        "decision_diagnostics": exp016_decision_diagnostics(
            v0_prediction_rows=v0_rows, cache_rows=cache_rows, exp014_prediction_rows=exp014_rows,
        ),
        "trajectory_groups": _trajectory_groups(exp014_trajectories, v0_rows, categories),
        "decision_counters": {
            "EXP014": learned_counters,
            "OracleGate_LearnedCandidate": oracle_gate_counters,
            "LearnedGate_OracleCandidate": oracle_candidate_counters,
        },
        "reference_checks": reference_checks,
        "provenance": {
            "stage_c_v0_val_predictions": str(v0_predictions.resolve()),
            "stage_c_v0_val_predictions_sha256": file_sha256(v0_predictions),
            "exp014_val_predictions": str(exp014_predictions.resolve()),
            "exp014_val_predictions_sha256": file_sha256(exp014_predictions),
            "stage_d_cache_summary": str(cache_summary_path.resolve()),
            "stage_d_cache_summary_sha256": file_sha256(cache_summary_path),
            "stage_d_val_features": str(Path(cache_summary["feature_files"]["val"]).resolve()),
            "stage_d_val_features_sha256": file_sha256(Path(cache_summary["feature_files"]["val"])),
            "stage_b_val_utility": str((stage_b_root / "utility_labels" / "val.jsonl").resolve()),
            "stage_b_val_utility_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"),
        },
        "validity": {
            "first_step_protocol_frozen": True,
            "gt_true_u2_used_only_for_offline_oracle_branches": True,
            "future_perception_used_as_learned_input": False,
            "test_split_accepted": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--exp014-predictions", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        cache_root=args.cache_root,
        stage_b_root=args.stage_b_root,
        exp014_predictions=args.exp014_predictions,
        v0_predictions=args.v0_predictions,
        label_mapping=args.label_mapping,
        output_path=args.output,
        split=args.split,
    )
    print(json.dumps({"experiment_id": "EXP016", "split": "val", "test_used": False, "metrics_table": result["metrics_table"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
