import pytest

from activeview.scripts.evaluate_stage_c_val import _comparison


def _baseline(*, legacy_c2: bool = False):
    payload = {
        "accuracy": 0.65,
        "macro_f1": 0.60,
        "regret": {"mean": 1.5, "median": 0.1, "p90": 8.0},
        "headroom_capture": 0.70,
    }
    payload["c2_wrong_high_utility_loss_rate" if legacy_c2 else "c2_rate"] = 0.4
    return payload


def _metrics():
    return {
        "recognition": {"StageC": {"accuracy": 0.70, "macro_f1": 0.62}},
        "decision_regret": {"mean": 1.2, "median": 0.08, "p90": 7.0},
        "positive_headroom_capture": {
            "aggregate_positive_clipped_ratio": 0.72,
        },
    }


def _analysis(c2_ratio=0.35):
    return {
        "failure_taxonomy": {
            "C2_wrong_high_utility_loss": {"ratio": c2_ratio},
        },
        # EXP001-only diagnostics may be absent for newer experiments.
    }


def test_comparison_emits_generic_metric_deltas_only():
    result = _comparison(_baseline(), _metrics(), _analysis())

    assert result["accuracy_delta"] == pytest.approx(0.05)
    assert result["macro_f1_delta"] == pytest.approx(0.02)
    assert result["mean_regret_delta"] == pytest.approx(-0.3)
    assert result["p90_regret_delta"] == pytest.approx(-1.0)
    assert result["headroom_delta"] == pytest.approx(0.02)
    assert result["c2_rate_delta"] == pytest.approx(-0.05)
    assert result["experiment"]["median_regret"] == pytest.approx(0.08)
    assert result["baseline"]["median_regret"] == pytest.approx(0.1)
    assert "large_gap" not in result
    assert "acceptance_checks" not in result
    assert "preliminary_protocol_status" not in result


def test_comparison_supports_legacy_exp001_c2_field_without_large_gap():
    result = _comparison(_baseline(legacy_c2=True), _metrics(), _analysis())

    assert result["baseline"]["c2_rate"] == 0.4
    assert result["c2_rate_delta"] == pytest.approx(-0.05)
