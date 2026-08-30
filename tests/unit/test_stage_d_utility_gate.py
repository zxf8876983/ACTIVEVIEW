import numpy as np
import pytest
import torch

from activeview.active_view.stage_d_contextual_gate import CONTEXTUAL_GATE_INPUT_DIM
from activeview.active_view.stage_d_policy import SequentialObservationRanker
from activeview.active_view.stage_d_utility_gate import (
    UtilityExecutedGateMLP,
    apply_utility_gate_decision,
    build_utility_gate_rows,
    utility_regression_loss,
)


def _feature_row(target_values=None, split="train"):
    return {
        "episode_id": "episode",
        "policy_split": split,
        "remaining_candidate_ids": [2, 3],
        "s0_feature": [0.0] * 275,
        "s1_feature": [0.1] * 275,
        "delta_semantic": [0.2] * 19,
        "second_step_candidate_geometry": [[0.0] * 11, [1.0] * 11],
        "second_step_candidate_geodesic": [2.0, 3.0],
        "second_step_utility_targets": target_values or [-1.5, 2.25],
    }


def _prediction_row(split="train"):
    return {
        "episode_id": "episode",
        "policy_split": split,
        "remaining_candidate_ids": [2, 3],
        "predicted_utilities": [0.1, 0.2],
    }


def _rows(target_values=None, split="train"):
    vectors = {"current": ([0.0] * 275, [1.0] * 275), "delta": ([0.0] * 19, [1.0] * 19), "geometry": ([0.0] * 11, [1.0] * 11)}
    return build_utility_gate_rows(
        feature_rows=[_feature_row(target_values, split)],
        prediction_rows=[_prediction_row(split)],
        current_mean=vectors["current"][0],
        current_std=vectors["current"][1],
        delta_mean=vectors["delta"][0],
        delta_std=vectors["delta"][1],
        geometry_mean=vectors["geometry"][0],
        geometry_std=vectors["geometry"][1],
        split=split,
    )


def test_c_hat_and_raw_true_u2_target_use_frozen_ranking_only():
    row = _rows()[0]
    assert row["candidate_id"] == 3
    assert row["selected_index"] == 1
    assert row["target_regression"] == 2.25


def test_true_u2_changes_target_but_not_model_features():
    first = _rows([-1.5, 2.25])[0]
    second = _rows([9.0, -4.0])[0]
    assert first["candidate_id"] == second["candidate_id"] == 3
    assert first["target_regression"] == 2.25
    assert second["target_regression"] == -4.0
    assert np.array_equal(first["candidate_geometry"], second["candidate_geometry"])
    assert np.array_equal(first["s0_feature"], second["s0_feature"])
    assert first["predicted_utility"] == second["predicted_utility"] == 0.2


def test_utility_gate_has_fixed_129d_regressor():
    model = UtilityExecutedGateMLP()
    assert model(torch.zeros((2, CONTEXTUAL_GATE_INPUT_DIM))).shape == (2,)


def test_strict_sign_decision_moves_only_for_positive_prediction():
    move = apply_utility_gate_decision(3, 1e-9)
    stay = apply_utility_gate_decision(3, 0.0)
    negative = apply_utility_gate_decision(3, -1e-9)
    assert move["predicted_stays"] is False and move["predicted_candidate_viewpoint_id"] == 3
    assert stay["predicted_stays"] is True and stay["predicted_candidate_viewpoint_id"] is None
    assert negative["predicted_stays"] is True


def test_true_u2_is_not_an_input_to_regression_loss_model():
    model = UtilityExecutedGateMLP()
    features = torch.zeros((1, CONTEXTUAL_GATE_INPUT_DIM))
    with torch.no_grad():
        before = model(features).clone()
    _ = utility_regression_loss(model(features), torch.tensor([10.0]))
    with torch.no_grad():
        after = model(features).clone()
    assert torch.equal(before, after)


def test_test_split_is_rejected():
    with pytest.raises(ValueError, match="Test is locked"):
        _rows(split="test")
