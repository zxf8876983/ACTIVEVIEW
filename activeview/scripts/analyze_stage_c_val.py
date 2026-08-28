#!/usr/bin/env python3
"""Freeze a read-only Stage C-v0 Val baseline for Stage C-v1 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_failure_analysis import (
    analyze_rows,
    load_jsonl,
    prepare_aligned_rows,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def freeze_val_baseline(
    *,
    dataset_root: Path,
    stage_b_root: Path,
    stage_c_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze the frozen Set Ranker Val predictions without running a model."""
    validation_path = stage_c_root / "validation_report.json"
    validation = _load(validation_path)
    if validation.get("passed") is not True or validation.get("error_count") != 0:
        raise RuntimeError("Frozen Stage C validator is not passed")

    evaluation_path = stage_c_root / "evaluations" / "set_ranker_evaluation_summary.json"
    evaluation = _load(evaluation_path)
    if evaluation.get("model_type") != "set_ranker":
        raise RuntimeError("Expected frozen Set Ranker evaluation summary")
    prediction_path = Path(evaluation["prediction_files"]["val"])
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    stage_a_summary = _load(dataset_root / "stage_a_summary.json")
    feature_summary = _load(stage_c_root / "stage_c_feature_summary.json")
    stage_a_rows = load_jsonl(stage_a_summary["episode_files"]["val"])
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    feature_rows = load_jsonl(feature_summary["feature_files"]["val"])
    prediction_rows = load_jsonl(prediction_path)
    rows = prepare_aligned_rows(
        stage_a_rows,
        stage_b_rows,
        feature_rows,
        prediction_rows,
        expected_split="val",
    )
    categories = evaluation["categories"]
    analysis = analyze_rows(rows, categories, split="val", model="set_ranker")
    val_metrics = evaluation["metrics"]["val"]
    difficulty = analysis["candidate_set_difficulty"]
    taxonomy = analysis["failure_taxonomy"]
    baseline_metrics: Dict[str, Any] = {
        "protocol": "ACTIVEVIEW v11.5 Stage C-v1 Val baseline",
        "model_type": "set_ranker",
        "split": "val",
        "episode_count": len(rows),
        "record_count": len({str(row["record_id"]) for row in rows}),
        "recognition": {
            name: val_metrics["recognition"][name]
            for name in ("NoMove", "StageC", "CandidateOracle", "SafeOracle")
        },
        "decision_regret": val_metrics["decision_regret"],
        "positive_headroom_capture": val_metrics["positive_headroom_capture"],
        "candidate_oracle_hit_rate": val_metrics["candidate_oracle_hit_rate"],
        "safe_action_match_rate": val_metrics["safe_action_match_rate"],
        "failure_taxonomy": {
            name: {
                "count": value["count"],
                "ratio": value["ratio"],
                "mean_regret": value["regret"]["mean"],
                "p90_regret": value["regret"]["p90"],
            }
            for name, value in taxonomy.items()
        },
        "regret_groups": {
            name: value
            for name, value in analysis["regret"].items()
            if name.startswith("G")
        },
        "utility_gap_quartiles": {
            name: {
                "count": difficulty[name]["count"],
                "candidate_hit_rate": difficulty[name]["candidate_hit_rate"],
                "mean_regret": difficulty[name]["regret"]["mean"],
                "median_regret": difficulty[name]["regret"]["median"],
                "p90_regret": difficulty[name]["regret"]["p90"],
                "headroom_capture": difficulty[name]["headroom"]["aggregate_capture"],
            }
            for name in ("very_small", "small", "medium", "large")
        },
        "large_gap": {
            "exact_candidate_hit": difficulty["large"]["candidate_hit_rate"],
            "mean_regret": difficulty["large"]["regret"]["mean"],
            "p90_regret": difficulty["large"]["regret"]["p90"],
            "headroom_capture": difficulty["large"]["headroom"]["aggregate_capture"],
        },
    }
    artifact_provenance = {
        "stage_c_validator_sha256": file_sha256(validation_path),
        "stage_c_evaluation_summary_sha256": file_sha256(evaluation_path),
        "stage_c_prediction_sha256": file_sha256(prediction_path),
        "stage_a_summary_sha256": file_sha256(dataset_root / "stage_a_summary.json"),
        "stage_b_summary_sha256": file_sha256(stage_b_root / "stage_b_summary.json"),
        "stage_c_feature_summary_sha256": file_sha256(stage_c_root / "stage_c_feature_summary.json"),
    }
    baseline_metrics["artifact_provenance"] = artifact_provenance
    analysis["artifact_provenance"] = artifact_provenance
    analysis["evaluation_summary"] = {
        "path": str(evaluation_path.resolve()),
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "baseline_val_metrics.json", baseline_metrics)
    _write_json(output_dir / "baseline_val_analysis.json", analysis)
    return {
        "output_dir": str(output_dir.resolve()),
        "episode_count": len(rows),
        "record_count": len({str(row["record_id"]) for row in rows}),
        "baseline_val_metrics": str((output_dir / "baseline_val_metrics.json").resolve()),
        "baseline_val_analysis": str((output_dir / "baseline_val_analysis.json").resolve()),
        "artifact_provenance": artifact_provenance,
    }


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--stage-c-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/stage_c_v1/EXP001_gap_aware_ranking",
    )
    args = parser.parse_args()
    result = freeze_val_baseline(
        dataset_root=args.dataset_root,
        stage_b_root=args.stage_b_root,
        stage_c_root=args.stage_c_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
