import pytest
import torch

from activeview.scripts.run_stage_d_exp049_051 import (
    _JointRevision,
    _budget,
    _legal_order,
)


def _row() -> dict[str, object]:
    return {
        "episode_id": "ep",
        "scene_id": "scene",
        "region": "room",
        "record_id": "record",
        "s0_viewpoint_id": 0,
        "s1_viewpoint_id": 1,
    }


def test_legal_order_excludes_visited_and_uses_geometry_then_id() -> None:
    pairwise = {("scene", "room"): {1: {0: 1.0, 2: 2.0, 3: 2.0, 4: 1.0}}}
    azimuths = {("scene", "room"): {i: float(i * 45) for i in range(32)}}
    v0 = {"ep": {"predicted_stays": False}}
    assert _legal_order(_row(), pairwise, azimuths, v0) == [4, 2, 3]


def test_budget_prefix_and_all_legal() -> None:
    order = [5, 2, 9]
    assert _budget(order, 2) == [5, 2]
    assert _budget(order, "ALL_LEGAL") == order


def test_joint_revision_current_context_dimension_is_38() -> None:
    model = _JointRevision()
    current = torch.zeros(2, 38)
    candidates = torch.zeros(2, 3, 26)
    mask = torch.ones(2, 3, dtype=torch.bool)
    scores, posterior = model(current, candidates, mask)
    assert scores.shape == (2, 3)
    assert posterior.shape == (2, 3, 16)


def test_legal_order_rejects_v0_stay() -> None:
    pairwise = {("scene", "room"): {1: {2: 1.0}}}
    azimuths = {("scene", "room"): {1: 0.0, 2: 90.0}}
    assert _legal_order(_row(), pairwise, azimuths, {"ep": {"predicted_stays": True}}) == []


def test_no_test_split_entrypoint() -> None:
    with pytest.raises(ValueError, match="Test"):
        # The campaign deliberately exposes only fixed Train/Val row loading.
        from activeview.scripts.run_stage_d_exp041_044 import _rows
        from activeview.core.paths import get_data_root

        _rows(get_data_root(), "test")
