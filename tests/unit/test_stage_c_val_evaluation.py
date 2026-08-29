from activeview.scripts.evaluate_stage_c_val import _comparison


def _baseline():
    return {
        "large_gap": {"mean_regret": 10.0},
        "c2_wrong_high_utility_loss_rate": 0.4,
        "accuracy": 0.65,
        "macro_f1": 0.60,
        "regret": {"p90": 8.0},
        "headroom_capture": 0.70,
    }


def _analysis(mean_regret=9.0, c2_ratio=0.35):
    return {
        "candidate_set_difficulty": {
            "large": {"regret": {"mean": mean_regret}},
        },
        "failure_taxonomy": {
            "C2_wrong_high_utility_loss": {"ratio": c2_ratio},
        },
    }


def test_comparison_acceptance_checks_use_val_only_metrics():
    metrics = {
        "recognition": {"StageC": {"accuracy": 0.70, "macro_f1": 0.60}},
        "decision_regret": {"p90": 7.0},
        "positive_headroom_capture": {"aggregate_positive_clipped_ratio": 0.72},
    }

    result = _comparison(_baseline(), metrics, _analysis())

    assert result["acceptance_checks"] == {
        "large_gap_mean_regret_improved_5pct": True,
        "harmful_ranking_diagnostic_improved": True,
        "macro_f1_drop_within_0_5pp": True,
    }
    assert result["preliminary_protocol_status"] == "PASS"


def test_comparison_marks_conflicting_metrics_for_review():
    metrics = {
        "recognition": {"StageC": {"accuracy": 0.70, "macro_f1": 0.58}},
        "decision_regret": {"p90": 9.0},
        "positive_headroom_capture": {"aggregate_positive_clipped_ratio": 0.69},
    }

    result = _comparison(_baseline(), metrics, _analysis(mean_regret=9.9, c2_ratio=0.41))

    assert result["preliminary_protocol_status"] == "REVIEW"
    assert result["acceptance_checks"]["large_gap_mean_regret_improved_5pct"] is False
    assert result["acceptance_checks"]["harmful_ranking_diagnostic_improved"] is False
    assert result["acceptance_checks"]["macro_f1_drop_within_0_5pp"] is False
