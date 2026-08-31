"""Focused correctness tests for EXP028 pure audit helpers."""

import numpy as np
import pytest

from activeview.active_view.stage_d_predictability import (
    margin_bin_index,
    majority_action,
    neighbor_agreement,
    neighbor_entropy,
    normalized_entropy,
    oracle_action_index,
    oracle_margin,
)


def test_oracle_matches_fixed_stay_first_ties() -> None:
    assert oracle_action_index([-1.0, -2.0]) == 0
    assert oracle_action_index([1.0, 1.0]) == 1


def test_margin_is_deterministic_and_bounded_bins() -> None:
    values = oracle_margin([0.6, -0.2])
    assert values["margin_1"] == pytest.approx(0.6)
    assert values["candidate_margin"] == pytest.approx(0.8)
    assert margin_bin_index(0.05) == 1
    assert margin_bin_index(2.0) == 6


def test_majority_tie_prefers_stay_then_candidate_order() -> None:
    assert majority_action([1, 2]) == 1
    assert majority_action([2, 0]) == 0


def test_entropy_range_and_binary_reduction() -> None:
    assert normalized_entropy([25, 0, 0]) == pytest.approx(0.0)
    assert normalized_entropy([1, 1, 1]) == pytest.approx(1.0)
    values = neighbor_entropy([0] * 13 + [1] * 6 + [2] * 6)
    assert 0.0 <= values["three_way"] <= 1.0
    assert 0.0 <= values["binary"] <= 1.0


def test_neighbor_agreement_uses_train_labels_only() -> None:
    train_labels = np.asarray([0, 1, 2, 2], dtype=np.int64)
    indices = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    val_labels = np.asarray([0, 2], dtype=np.int64)
    result = neighbor_agreement(train_labels, indices, val_labels, 1)
    assert result["three_way_accuracy"] == pytest.approx(1.0)


def test_true_utility_is_not_needed_for_observable_helpers() -> None:
    # The NN helper consumes only labels and precomputed indices; no utility
    # argument exists that could leak oracle targets into representation.
    assert "true_utilities" not in neighbor_agreement.__annotations__
