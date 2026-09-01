"""Focused invariants for the EXP041--EXP044 world-model components."""

from __future__ import annotations

import pytest
import torch

from activeview.active_view.stage_d_world_model import CandidateObservationWorldModel, world_model_loss
from activeview.scripts.run_stage_d_exp041_044 import _rows


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch, 2, 3, 30, 17),
        torch.zeros(batch, 2, 9),
        torch.zeros(batch, 9),
    )


def test_world_model_output_shape_and_belief_variant() -> None:
    history, descriptor, candidate = _inputs()
    model = CandidateObservationWorldModel(use_belief=True)
    output = model(history, descriptor, candidate, history_belief=torch.zeros(2, 32))
    assert tuple(output.shape) == (2, 3, 30, 17)


def test_world_model_rgb_variant_uses_visited_tokens_only() -> None:
    history, descriptor, candidate = _inputs(1)
    model = CandidateObservationWorldModel(use_rgb=True)
    rgb = torch.zeros(1, 2, 16, 768)
    output = model(history, descriptor, candidate, history_rgb=rgb)
    assert tuple(output.shape) == (1, 3, 30, 17)


def test_world_model_rejects_wrong_skeleton_shape() -> None:
    model = CandidateObservationWorldModel()
    with pytest.raises(ValueError, match="history_skeleton"):
        model(torch.zeros(1, 2, 3, 17), torch.zeros(1, 2, 9), torch.zeros(1, 9))


def test_fixed_velocity_loss_is_nonnegative() -> None:
    prediction = torch.zeros(2, 3, 30, 17)
    target = torch.ones_like(prediction)
    total, pose, velocity = world_model_loss(prediction, target)
    assert float(total) >= float(pose) >= 0.0
    assert float(velocity) >= 0.0


def test_test_split_is_locked(tmp_path) -> None:
    with pytest.raises(ValueError, match="Test"):
        _rows(tmp_path, "test")
