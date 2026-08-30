import numpy as np
import pytest
import torch

from activeview.active_view.stage_d_executed_gate import executed_candidate_decision
from activeview.active_view.stage_d_executed_gate_training import (
    EXECUTED_GATE_FEATURE_DIM,
    ExecutedCandidateGateMLP,
    build_executed_gate_examples,
)


def _feature_row(episode_id: str = "e") -> dict:
    return {
        "episode_id": episode_id,
        "policy_split": "train",
        "remaining_candidate_ids": [2, 3],
        "second_step_candidate_geometry": [[0.0] * 11, [1.0] * 11],
        "second_step_candidate_geodesic": [2.0, 3.0],
        "second_step_utility_targets": [1.0, -1.0],
    }


def _prediction_row(episode_id: str = "e") -> dict:
    return {
        "episode_id": episode_id,
        "policy_split": "train",
        "remaining_candidate_ids": [2, 3],
        "predicted_utilities": [0.1, 0.2],
    }


def test_c_hat_and_y_exec_use_frozen_ranking_and_selected_true_u2_only():
    result = executed_candidate_decision(
        learned_utilities=[0.1, 0.2], true_utilities=[10.0, -1.0],
        candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert result["learned_candidate_id"] == 3
    assert result["executed_true_utility"] == -1.0
    assert result["executed_positive"] is False


def test_training_examples_have_no_true_utility_in_model_features():
    examples = build_executed_gate_examples(
        feature_rows=[_feature_row()], prediction_rows=[_prediction_row()],
        geometry_mean=[0.0] * 11, geometry_std=[1.0] * 11, split="train",
    )
    assert len(examples) == 1
    assert examples[0]["candidate_id"] == 3
    assert examples[0]["target"] == 0
    assert examples[0]["features"].shape == (EXECUTED_GATE_FEATURE_DIM,)
    assert np.allclose(examples[0]["features"][:11], 1.0)
    assert examples[0]["features"][11] == pytest.approx(0.2)
    changed_target_row = _feature_row()
    changed_target_row["second_step_utility_targets"] = [-1.0, 1.0]
    changed = build_executed_gate_examples(
        feature_rows=[changed_target_row], prediction_rows=[_prediction_row()],
        geometry_mean=[0.0] * 11, geometry_std=[1.0] * 11, split="train",
    )
    assert np.array_equal(examples[0]["features"], changed[0]["features"])
    assert examples[0]["target"] != changed[0]["target"]


def test_exp019_model_has_fixed_small_architecture():
    model = ExecutedCandidateGateMLP()
    output = model(torch.zeros((2, EXECUTED_GATE_FEATURE_DIM)))
    assert output.shape == (2,)


def test_test_split_is_rejected_for_training_examples():
    with pytest.raises(ValueError, match="Test is locked"):
        build_executed_gate_examples(
            feature_rows=[], prediction_rows=[], geometry_mean=[0.0] * 11,
            geometry_std=[1.0] * 11, split="test",
        )
