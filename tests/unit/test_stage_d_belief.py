"""Focused correctness tests for EXP038--EXP040 belief helpers."""

import numpy as np
import pytest

from activeview.active_view.stage_d_belief import (
    belief_from_log_probs,
    fuse_beliefs,
    oracle_action,
    select_max_correctness,
    select_min_risk,
    top_k_belief,
)


def test_belief_fusion_is_normalized_and_top3_keeps_mass() -> None:
    first = belief_from_log_probs(np.arange(16, dtype=np.float32))
    second = belief_from_log_probs(np.arange(16, dtype=np.float32)[::-1])
    for mode in ("latest", "mean", "geometric"):
        result = fuse_beliefs([first, second], mode)
        assert result.shape == (16,)
        assert np.isclose(float(result.sum()), 1.0)
    top3 = top_k_belief(first, 3)
    assert np.isclose(float(top3.sum()), 1.0)
    assert np.count_nonzero(top3) == 3


def test_oracle_stay_tie_and_action_selectors_are_deterministic() -> None:
    assert oracle_action([-1.0, -2.0]) == 0
    assert oracle_action([1.0, 1.0]) == 1
    assert select_min_risk([0.5, 0.5, 1.0]) == 0
    assert select_max_correctness([0.5, 0.5, 0.1]) == 0


def test_belief_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        belief_from_log_probs(np.zeros(15))
    with pytest.raises(ValueError):
        fuse_beliefs([], "mean")
    with pytest.raises(ValueError):
        top_k_belief(np.ones(16), 0)
