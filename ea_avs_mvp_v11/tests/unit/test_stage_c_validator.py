from ea_avs_mvp_v11.scripts.validate_stage_c import CANONICAL_COUNTS, FORBIDDEN, _compare


def test_stage_c_validator_uses_frozen_policy_counts():
    assert CANONICAL_COUNTS == {"train": 589, "val": 197, "test": 194}


def test_stage_c_feature_forbidden_fields_are_not_silent_inputs():
    assert "candidate_entropy" in FORBIDDEN
    assert "candidate_skeleton" in FORBIDDEN
    errors = []
    _compare({"accuracy": 0.5}, {"accuracy": 0.4}, "metrics", errors)
    assert errors == ["metric_value_mismatch:metrics.accuracy"]
