#!/usr/bin/env python3
"""Run the Val-only EXP018 executed-candidate gate alignment audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_error_decomposition import (
    build_exp016_variant_trajectories,
)
from activeview.active_view.stage_d_evaluation import (
    build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_executed_gate import (
    build_executed_candidate_oracle_gate_trajectories,
    summarize_target_alignment,
    validate_exp018_episode_alignment,
)
from activeview.active_view.stage_d_gate_calibration import (
    build_thresholded_prediction_rows,
    load_calibration_artifact,
)
from activeview.active_view.utility_label_builder import file_sha256


REFERENCE_METRICS = {
    "EXP014": {"accuracy": 0.6582540930864375, "mean_regret": 1.4224626188609946},
    "EXP017": {"accuracy": 0.6509616072066919, "mean_regret": 1.477152994442029},
    "AnyPositiveOracleGate_LearnedCandidate": {"accuracy": 0.7200257381854579, "mean_regret": 0.9691383557931994},
}


def validate_exp018_split(split: str) -> None:
    """Accept only the explicitly authorized Val analysis split."""
    if str(split).lower() != "val":
        raise ValueError("EXP018 accepts Val only; Test is locked")


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _assert_val_rows(rows: Sequence[Mapping[str, Any]], name: str, *, allow_missing: bool = False) -> None:
    invalid = set()
    for row in rows:
        value = row.get("policy_split")
        if value is None:
            if not allow_missing:
                invalid.add("<missing>")
        elif str(value).lower() != "val":
            invalid.add(str(value))
    if invalid:
        raise ValueError(f"{name} must contain only Val rows: {sorted(invalid)}")


def _metric_row(name: str, summary: Mapping[str, Any], gate: str, candidate: str) -> dict[str, Any]:
    regret = summary["decision_regret"]
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
        "headroom_capture": float(summary["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]),
        "average_moves": float(movement["average_moves"]),
        "mean_geodesic_cost_m": float(movement["trajectory_geodesic_cost_m"]["mean"]),
    }


def _reference_checks(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for method, expected in REFERENCE_METRICS.items():
        actual = metrics[method]
        checks[method] = {
            key: {
                "actual": float(actual[key]),
                "reference": float(value),
                "within_abs_tolerance_1e-5": abs(float(actual[key]) - float(value)) <= 1e-5,
            }
            for key, value in expected.items()
        }
    return checks


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def _changed_exp017_episodes(
    *,
    exp014_rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    zero = {str(row["episode_id"]): row for row in exp014_rows}
    calibrated = {str(row["episode_id"]): row for row in calibrated_rows}
    executed = {str(row["episode_id"]): row for row in decisions}
    changed = [
        episode_id
        for episode_id, row in zero.items()
        if bool(row["predicted_stays"]) and not bool(calibrated[episode_id]["predicted_stays"])
    ]
    move_to_stay = sum(
        not bool(row["predicted_stays"]) and bool(calibrated[episode_id]["predicted_stays"])
        for episode_id, row in zero.items()
    )
    selected = [float(executed[episode_id]["executed_true_utility"]) for episode_id in changed]
    positive = sum(value > 0.0 for value in selected)
    nonpositive = len(selected) - positive
    return {
        "tau_0_stay_count": sum(bool(row["predicted_stays"]) for row in zero.values()),
        "tau_0_move_count": sum(not bool(row["predicted_stays"]) for row in zero.values()),
        "calibrated_stay_count": sum(bool(row["predicted_stays"]) for row in calibrated.values()),
        "calibrated_move_count": sum(not bool(row["predicted_stays"]) for row in calibrated.values()),
        "stay_to_move_count": len(changed),
        "move_to_stay_count": int(move_to_stay),
        "changed_executed_positive_count": int(positive),
        "changed_executed_nonpositive_count": int(nonpositive),
        "changed_executed_true_utility_mean": float(np.mean(selected)) if selected else 0.0,
        "changed_executed_true_utility_median": float(np.median(selected)) if selected else 0.0,
    }


def analyze(
    *,
    cache_root: Path,
    stage_b_root: Path,
    exp014_predictions: Path,
    exp017_calibration: Path,
    exp017_result: Path,
    v0_predictions: Path,
    label_mapping: Path,
    output_path: Path,
    runtime_output: Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    """Run EXP018 using only the frozen corrected Val artifacts."""
    validate_exp018_split(split)
    calibration = load_calibration_artifact(exp017_calibration)
    exp017_summary = json.loads(exp017_result.read_text(encoding="utf-8"))
    if exp017_summary.get("experiment_id") != "EXP017" or exp017_summary.get("test_used") is not False:
        raise ValueError("Invalid EXP017 result provenance")

    cache_summary_path = cache_root / "stage_d_feature_summary.json"
    cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    cache_rows = load_jsonl(Path(cache_summary["feature_files"]["val"]))
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_rows = load_jsonl(v0_predictions)
    exp014_rows = load_jsonl(exp014_predictions)
    _assert_val_rows(stage_b_rows, "Stage B Val utility")
    _assert_val_rows(v0_rows, "Stage C-v0 Val predictions")
    _assert_val_rows(cache_rows, "Stage D Val cache")
    _assert_val_rows(exp014_rows, "EXP014 Val predictions", allow_missing=True)
    alignment = validate_exp018_episode_alignment(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows,
    )
    categories = _categories(label_mapping)

    exp014_trajectories = build_stage_d_trajectories(stage_b_rows, v0_rows, cache_rows, exp014_rows)
    tau = float(calibration["selected_tau"])
    calibrated_rows = build_thresholded_prediction_rows(exp014_rows, cache_rows, tau)
    exp017_trajectories = build_stage_d_trajectories(stage_b_rows, v0_rows, cache_rows, calibrated_rows)
    any_oracle_trajectories, _ = build_exp016_variant_trajectories(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows,
        gate="oracle",
        candidate="learned",
    )
    executed_trajectories, decisions, executed_counters = build_executed_candidate_oracle_gate_trajectories(
        stage_b_rows=stage_b_rows,
        v0_prediction_rows=v0_rows,
        cache_rows=cache_rows,
        exp014_prediction_rows=exp014_rows,
    )
    method_rows = {
        "EXP014": exp014_trajectories,
        "EXP017": exp017_trajectories,
        "AnyPositiveOracleGate_LearnedCandidate": any_oracle_trajectories,
        "ExecutedCandidateOracleGate_LearnedCandidate": executed_trajectories,
    }
    summaries = {name: summarize_trajectory_rows(rows, categories) for name, rows in method_rows.items()}
    table = [
        _metric_row("EXP014", summaries["EXP014"], "learned tau=0", "learned"),
        _metric_row("EXP017", summaries["EXP017"], f"learned tau={tau:.16g}", "learned"),
        _metric_row("AnyPositiveOracleGate + LearnedCandidate", summaries["AnyPositiveOracleGate_LearnedCandidate"], "true max U2 > 0", "learned"),
        _metric_row("ExecutedCandidateOracleGate + LearnedCandidate", summaries["ExecutedCandidateOracleGate_LearnedCandidate"], "true U2(c_hat) > 0", "learned"),
    ]
    metrics = {key: value for key, value in zip(method_rows, table)}
    references = _reference_checks(metrics)
    if not all(item["within_abs_tolerance_1e-5"] for checks in references.values() for item in checks.values()):
        raise ValueError("Frozen EXP014, EXP017 or any-positive reference mismatch")
    target_alignment = summarize_target_alignment(decisions)
    changed = _changed_exp017_episodes(
        exp014_rows=exp014_rows,
        calibrated_rows=calibrated_rows,
        decisions=decisions,
    )
    exp_metric = metrics["EXP014"]
    any_metric = metrics["AnyPositiveOracleGate_LearnedCandidate"]
    executed_metric = metrics["ExecutedCandidateOracleGate_LearnedCandidate"]
    any_acc_gap = any_metric["accuracy"] - exp_metric["accuracy"]
    any_regret_gap = exp_metric["mean_regret"] - any_metric["mean_regret"]
    result = {
        "experiment_id": "EXP018",
        "experiment_name": "executed_candidate_gate_alignment",
        "status": "COMPLETED",
        "decision": "INCONCLUSIVE",
        "split": "val",
        "test_used": False,
        "training_performed": False,
        "perception_regenerated": False,
        "habitat_rendering_performed": False,
        "stgcn_retrained": False,
        "episode_count": len(stage_b_rows),
        "v0_move_eligible_episode_count": len(decisions),
        "episode_alignment": alignment,
        "metrics_table": table,
        "target_alignment": target_alignment,
        "executed_candidate_oracle_gate": {
            "accuracy": executed_metric["accuracy"],
            "macro_f1": executed_metric["macro_f1"],
            "mean_regret": executed_metric["mean_regret"],
            "median_regret": executed_metric["median_regret"],
            "p90_regret": executed_metric["p90_regret"],
            "headroom_capture": executed_metric["headroom_capture"],
        },
        "headroom_comparison": {
            "any_oracle_accuracy_gain": any_acc_gap,
            "executed_oracle_accuracy_gain": executed_metric["accuracy"] - exp_metric["accuracy"],
            "executable_fraction_of_any_gate_accuracy_headroom": _safe_fraction(executed_metric["accuracy"] - exp_metric["accuracy"], any_acc_gap),
            "any_oracle_regret_reduction": any_regret_gap,
            "executed_oracle_regret_reduction": exp_metric["mean_regret"] - executed_metric["mean_regret"],
            "executable_fraction_of_any_gate_regret_headroom": _safe_fraction(exp_metric["mean_regret"] - executed_metric["mean_regret"], any_regret_gap),
        },
        "exp017_changed_gate_decomposition": changed,
        "executed_gate_counters": executed_counters,
        "reference_checks": references,
        "provenance": {
            "source_commit": _git_commit(),
            "stage_c_v0_val_predictions": str(v0_predictions.resolve()),
            "stage_c_v0_val_predictions_sha256": file_sha256(v0_predictions),
            "exp014_val_predictions": str(exp014_predictions.resolve()),
            "exp014_val_predictions_sha256": file_sha256(exp014_predictions),
            "exp017_calibration": str(exp017_calibration.resolve()),
            "exp017_calibration_sha256": file_sha256(exp017_calibration),
            "exp017_result": str(exp017_result.resolve()),
            "exp017_result_sha256": file_sha256(exp017_result),
            "stage_d_cache_summary": str(cache_summary_path.resolve()),
            "stage_d_cache_summary_sha256": file_sha256(cache_summary_path),
            "stage_d_val_cache": str(Path(cache_summary["feature_files"]["val"]).resolve()),
            "stage_d_val_cache_sha256": file_sha256(Path(cache_summary["feature_files"]["val"])),
            "stage_b_val_utility": str((stage_b_root / "utility_labels" / "val.jsonl").resolve()),
            "stage_b_val_utility_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"),
        },
        "validity": {
            "first_step_protocol_frozen": True,
            "p1_proposal_frozen": True,
            "learned_candidate_ranking_frozen": True,
            "true_u2_used_only_for_offline_gate_analysis": True,
            "true_u2_used_for_candidate_identity": False,
            "test_split_accepted": False,
        },
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    if runtime_output is not None:
        runtime_output.parent.mkdir(parents=True, exist_ok=True)
        runtime_output.write_text(payload, encoding="utf-8")
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--exp014-predictions", type=Path, required=True)
    parser.add_argument("--exp017-calibration", type=Path, required=True)
    parser.add_argument("--exp017-result", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--split", choices=("val",), default="val")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        cache_root=args.cache_root,
        stage_b_root=args.stage_b_root,
        exp014_predictions=args.exp014_predictions,
        exp017_calibration=args.exp017_calibration,
        exp017_result=args.exp017_result,
        v0_predictions=args.v0_predictions,
        label_mapping=args.label_mapping,
        output_path=args.output,
        runtime_output=args.runtime_output,
        split=args.split,
    )
    print(json.dumps({"experiment_id": "EXP018", "split": "val", "test_used": False, "metrics_table": result["metrics_table"], "target_alignment": result["target_alignment"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
