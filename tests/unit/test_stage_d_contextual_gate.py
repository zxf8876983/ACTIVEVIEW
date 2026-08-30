import numpy as np
import pytest
import torch

from activeview.active_view.stage_d_contextual_gate import (
    CONTEXTUAL_GATE_INPUT_DIM,
    CONTEXTUAL_TOKEN_DIM,
    ContextualExecutedGateMLP,
    apply_contextual_gate_decision,
    build_contextual_gate_rows,
    contextual_candidate_tokens,
    freeze_exp014_ranker,
)
from activeview.active_view.stage_d_policy import SequentialObservationRanker


def _feature_row(episode_id: str = "e", target_values=None) -> dict:
    return {
        "episode_id": episode_id,
        "policy_split": "train",
        "remaining_candidate_ids": [2, 3],
        "s0_feature": [0.0] * 275,
        "s1_feature": [0.1] * 275,
        "delta_semantic": [0.2] * 19,
        "second_step_candidate_geometry": [[0.0] * 11, [1.0] * 11],
        "second_step_candidate_geodesic": [2.0, 3.0],
        "second_step_utility_targets": target_values or [1.0, -1.0],
    }


def _prediction_row(episode_id: str = "e") -> dict:
    return {
        "episode_id": episode_id,
        "policy_split": "train",
        "remaining_candidate_ids": [2, 3],
        "predicted_utilities": [0.1, 0.2],
    }


def _stats(dim: int):
    return [0.0] * dim, [1.0] * dim


def test_contextual_token_has_128_dimensions_and_does_not_call_utility_head():
    model = freeze_exp014_ranker(SequentialObservationRanker())

    class RaiseUtilityHead(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("utility head called")

    model.utility_head = RaiseUtilityHead()
    tokens = contextual_candidate_tokens(
        model,
        torch.zeros((2, 275)),
        torch.zeros((2, 275)),
        torch.zeros((2, 19)),
        torch.zeros((2, 2, 11)),
        torch.ones((2, 2), dtype=torch.bool),
    )
    assert tokens.shape == (2, 2, CONTEXTUAL_TOKEN_DIM)


def test_all_exp014_parameters_are_frozen():
    model = freeze_exp014_ranker(SequentialObservationRanker())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.training


def test_contextual_gate_input_is_129d_and_target_only_depends_on_selected_true_u2():
    means = {"current": _stats(275), "delta": _stats(19), "geometry": _stats(11)}
    kwargs = {
        "feature_rows": [_feature_row()],
        "prediction_rows": [_prediction_row()],
        "current_mean": means["current"][0], "current_std": means["current"][1],
        "delta_mean": means["delta"][0], "delta_std": means["delta"][1],
        "geometry_mean": means["geometry"][0], "geometry_std": means["geometry"][1],
        "split": "train",
    }
    first = build_contextual_gate_rows(**kwargs)[0]
    changed = _feature_row(target_values=[-1.0, 1.0])
    second = build_contextual_gate_rows(**{**kwargs, "feature_rows": [changed]})[0]
    assert first["candidate_id"] == second["candidate_id"] == 3
    assert first["target"] == 0
    assert second["target"] == 1
    assert np.array_equal(first["candidate_geometry"], second["candidate_geometry"])
    assert first["predicted_utility"] == second["predicted_utility"] == 0.2


def test_gate_decision_always_executes_frozen_candidate():
    move = apply_contextual_gate_decision(3, 0.01)
    stay = apply_contextual_gate_decision(3, -0.01)
    assert move["predicted_stays"] is False
    assert move["predicted_candidate_viewpoint_id"] == 3
    assert stay["predicted_stays"] is True
    assert stay["predicted_candidate_viewpoint_id"] is None


def test_exp020_gate_has_fixed_129d_model():
    model = ContextualExecutedGateMLP()
    assert model(torch.zeros((2, CONTEXTUAL_GATE_INPUT_DIM))).shape == (2,)


def test_test_split_is_rejected():
    with pytest.raises(ValueError, match="Test is locked"):
        build_contextual_gate_rows(
            feature_rows=[], prediction_rows=[],
            current_mean=[0.0] * 275, current_std=[1.0] * 275,
            delta_mean=[0.0] * 19, delta_std=[1.0] * 19,
            geometry_mean=[0.0] * 11, geometry_std=[1.0] * 11,
            split="test",
        )
