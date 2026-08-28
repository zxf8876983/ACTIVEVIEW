from ea_avs_mvp_v11.scripts.validate_stage_c import CANONICAL_COUNTS, FORBIDDEN, _compare, _validate_independent_decision


def test_stage_c_validator_uses_frozen_policy_counts():
    assert CANONICAL_COUNTS == {"train": 589, "val": 197, "test": 194}


def test_stage_c_feature_forbidden_fields_are_not_silent_inputs():
    assert "candidate_entropy" in FORBIDDEN
    assert "candidate_skeleton" in FORBIDDEN
    errors = []
    _compare({"accuracy": 0.5}, {"accuracy": 0.4}, "metrics", errors)
    assert errors == ["metric_value_mismatch:metrics.accuracy"]


def test_validator_recomputes_stage_c_decision_from_utilities_and_stage_b():
    stage_b = {
        "current": {"viewpoint_id": 0, "predicted_label_id": 1, "entropy": 0.9},
        "candidates": [
            {"viewpoint_id": 1, "predicted_label_id": 2, "entropy": 0.4, "utility": 0.2},
            {"viewpoint_id": 2, "predicted_label_id": 3, "entropy": 0.3, "utility": -0.1},
        ],
        "oracle": {"candidate_oracle_viewpoint_id": 1, "safe_oracle_viewpoint_id": 1, "safe_oracle_stays": False, "safe_oracle_utility": 0.2},
    }
    contract = {"candidate_geodesic": [2.0, 1.0]}
    row = {
        "episode_id": "e", "candidate_viewpoint_ids": [1, 2], "predicted_utilities": [0.5, 0.5],
        "predicted_candidate_viewpoint_id": 1, "predicted_stays": False, "predicted_action": "candidate:1",
        "selected_true_utility": 0.2, "selected_predicted_label_id": 2, "selected_entropy": 0.4,
        "candidate_oracle_viewpoint_id": 1, "candidate_oracle_predicted_label_id": 2,
        "candidate_oracle_entropy": 0.4, "safe_oracle_viewpoint_id": 1, "safe_oracle_stays": False,
        "safe_oracle_action": "candidate:1", "safe_oracle_utility": 0.2,
        "safe_oracle_predicted_label_id": 2, "safe_oracle_entropy": 0.4, "regret": 0.3,
    }
    errors = []
    _validate_independent_decision(row, stage_b, contract, "unit", "val", errors)
    assert "decision_candidate_id_mismatch:unit:val:e" in errors
    assert "selected_true_utility_mismatch:unit:val:e" in errors
