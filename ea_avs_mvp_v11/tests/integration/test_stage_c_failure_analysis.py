"""Read-only integration checks for the frozen Stage C failure analysis."""

import json
from pathlib import Path

import pytest

from ea_avs_mvp_v11.active_view.stage_c_failure_analysis import (
    analyze_rows,
    load_jsonl,
    prepare_aligned_rows,
)
from ea_avs_mvp_v11.core.paths import get_data_root


DATA_ROOT = get_data_root()
DATASET_ROOT = DATA_ROOT / "datasets/policy_v11_5"
STAGE_B_ROOT = DATASET_ROOT / "stage_b"
STAGE_C_ROOT = DATASET_ROOT / "stage_c"


def test_frozen_stage_c_failure_analysis_coverage():
    """Ensure the analysis consumes the approved Test rows one-to-one."""
    validation_path = STAGE_C_ROOT / "validation_report.json"
    if not validation_path.exists():
        pytest.skip("frozen Stage C runtime artifacts are not available")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert validation["error_count"] == 0

    stage_a_summary = json.loads(
        (DATASET_ROOT / "stage_a_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads(
        (STAGE_C_ROOT / "stage_c_feature_summary.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (STAGE_C_ROOT / "evaluations/set_ranker_evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    stage_a = load_jsonl(stage_a_summary["episode_files"]["test"])
    stage_b = load_jsonl(STAGE_B_ROOT / "utility_labels/test.jsonl")
    features = load_jsonl(feature_summary["feature_files"]["test"])
    predictions = load_jsonl(evaluation["prediction_files"]["test"])

    assert len(predictions) == 13_774
    assert len({str(row["record_id"]) for row in predictions}) == 194
    rows = prepare_aligned_rows(stage_a, stage_b, features, predictions)
    summary = analyze_rows(rows, evaluation["categories"])

    assert summary["episode_count"] == 13_774
    assert summary["record_count"] == 194
    assert sum(
        item["count"] for item in summary["failure_taxonomy"].values()
    ) == summary["episode_count"]
    groups = ("G0_near_optimal", "G1_low_regret", "G2_moderate_regret", "G3_high_regret")
    assert sum(summary["regret"][name]["count"] for name in groups) == summary["episode_count"]
