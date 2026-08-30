import pytest

from activeview.active_view.stage_d_executed_gate import (
    build_executed_candidate_oracle_gate_trajectories,
    executed_candidate_decision,
    summarize_target_alignment,
)
from activeview.scripts.analyze_stage_d_executed_gate_alignment import build_parser


def _record(episode_id: str = "e") -> dict:
    return {
        "episode_id": episode_id,
        "record_id": "r",
        "policy_split": "val",
        "scene_id": "s",
        "region": "bedroom",
        "label_id": 0,
        "current": {"viewpoint_id": 0, "predicted_label_id": 0},
        "candidates": [
            {"viewpoint_id": 1, "predicted_label_id": 0, "geodesic_distance_m": 1.0, "utility": 0.0},
            {"viewpoint_id": 2, "predicted_label_id": 1, "geodesic_distance_m": 2.0, "utility": 1.0},
            {"viewpoint_id": 3, "predicted_label_id": 2, "geodesic_distance_m": 3.0, "utility": -1.0},
        ],
        "oracle": {"safe_oracle_utility": 1.0, "safe_oracle_stays": False},
    }


def test_any_positive_but_learned_candidate_is_not_executable():
    result = executed_candidate_decision(
        learned_utilities=[0.1, 0.2], true_utilities=[1.0, -1.0],
        candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert result["learned_candidate_id"] == 3
    assert result["any_positive"] is True
    assert result["executed_positive"] is False


def test_executed_gate_moves_to_exact_learned_candidate():
    result = executed_candidate_decision(
        learned_utilities=[0.1, 0.2], true_utilities=[-1.0, 1.0],
        candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert result["learned_candidate_id"] == 3
    assert result["executed_positive"] is True


def test_executed_gate_never_switches_to_true_best_candidate():
    result = executed_candidate_decision(
        learned_utilities=[0.2, 0.1], true_utilities=[1.0, 10.0],
        candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert result["learned_candidate_id"] == 2
    assert result["executed_true_utility"] == 1.0


def test_v0_stay_remains_stay_at_s0():
    rows, decisions, counters = build_executed_candidate_oracle_gate_trajectories(
        stage_b_rows=[_record()],
        v0_prediction_rows=[{"episode_id": "e", "predicted_stays": True}],
        cache_rows=[], exp014_prediction_rows=[],
    )
    assert rows[0]["moves"] == 0
    assert rows[0]["selected_viewpoint_id"] == 0
    assert decisions == []
    assert counters == {"v0_stay": 1}


def test_impossible_executed_positive_without_any_positive_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        summarize_target_alignment([{"any_positive": False, "executed_positive": True}])


def test_exp018_entry_point_rejects_test_split():
    required = [
        "--cache-root", "cache", "--stage-b-root", "stage-b",
        "--exp014-predictions", "exp014", "--exp017-calibration", "calibration",
        "--exp017-result", "result", "--v0-predictions", "v0",
        "--label-mapping", "labels", "--output", "output",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(required + ["--split", "test"])
