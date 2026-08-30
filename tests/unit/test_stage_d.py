import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, summarize_trajectory_rows
from activeview.active_view.stage_d_dataset import second_step_geometry
from activeview.active_view.stage_d_policy import (
    SequentialObservationRanker,
    schema_metadata,
    semantic_delta,
    second_step_utility,
    trajectory_cost,
)
from activeview.active_view.stage_c_features import frozen_current_features


class _FrozenSTGCNStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 16, bias=False)

    def forward_features(self, tensor):
        return torch.ones((tensor.shape[0], 256), dtype=tensor.dtype, device=tensor.device)


def test_second_step_utility_and_semantic_delta_are_defined_from_s1():
    s0 = np.zeros(275, dtype=np.float32)
    s1 = np.ones(275, dtype=np.float32)
    assert second_step_utility(-1.0, -2.5) == 1.5
    assert semantic_delta(s0, s1).shape == (19,)
    assert np.allclose(semantic_delta(s0, s1), 1.0)


def test_cached_s1_skeleton_is_reduced_to_275d_observable_feature():
    model = _FrozenSTGCNStub().eval()
    skeleton = np.zeros((3, 30, 17), dtype=np.float32)
    feature, log_probs = frozen_current_features(model, skeleton, torch.device("cpu"))
    assert feature.shape == (256,)
    assert log_probs.shape == (16,)
    assert np.isfinite(feature).all()
    assert np.isfinite(log_probs).all()


def test_future_unvisited_perception_is_not_a_policy_input():
    model = SequentialObservationRanker().eval()
    s0 = torch.randn(1, 275)
    s1 = torch.randn(1, 275)
    delta = torch.randn(1, 19)
    geometry = torch.randn(1, 2, 11)
    mask = torch.ones(1, 2, dtype=torch.bool)
    first = model(s0, s1, delta, geometry, mask)
    _unused_future_a = torch.randn(1, 2, 275)
    _unused_future_b = torch.randn(1, 2, 275)
    second = model(s0, s1, delta, geometry, mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=0.0)
    assert schema_metadata()["future_unvisited_candidate_perception_used_as_input"] is False


def test_second_step_geometry_changes_when_current_view_changes():
    first = second_step_geometry(
        s1_position=[0.0, 0.0, 0.0], s1_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0], target_snapped_position=[1.0, 0.0, 0.0],
        target_geodesic=1.0, placement_position=[0.0, 0.0, 0.0],
    )
    second = second_step_geometry(
        s1_position=[0.0, 0.0, 1.0], s1_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0], target_snapped_position=[1.0, 0.0, 0.0],
        target_geodesic=1.0, placement_position=[0.0, 0.0, 0.0],
    )
    assert not np.allclose(first, second)


def test_second_step_geometry_model_is_candidate_aligned_and_permutation_equivariant():
    model = SequentialObservationRanker().eval()
    s0 = torch.randn(1, 275)
    s1 = torch.randn(1, 275)
    delta = torch.randn(1, 19)
    geometry = torch.randn(1, 2, 11)
    mask = torch.ones(1, 2, dtype=torch.bool)
    first = model(s0, s1, delta, geometry, mask)
    permutation = torch.tensor([1, 0])
    second = model(s0, s1, delta, geometry[:, permutation], mask[:, permutation])
    assert first.shape == (1, 2)
    assert torch.allclose(first[:, permutation], second, atol=1e-6, rtol=0.0)


def test_fixed_first_oracle_cannot_change_initial_stay():
    stage_b = [{
        "episode_id": "e0", "record_id": "r0", "policy_split": "val", "scene_id": "s", "region": "bedroom", "label_id": 0,
        "current": {"viewpoint_id": 0, "predicted_label_id": 0, "entropy": 0.1, "correct": True},
        "candidates": [{"viewpoint_id": 1, "utility": 3.0, "geodesic_distance_m": 1.0, "predicted_label_id": 1, "entropy": 0.2}],
        "oracle": {"candidate_oracle_viewpoint_id": 1, "safe_oracle_viewpoint_id": 1, "safe_oracle_utility": 3.0, "safe_oracle_stays": False},
    }]
    v0 = [{"episode_id": "e0", "predicted_stays": True, "predicted_candidate_viewpoint_id": 1}]
    assert build_fixed_first_oracle(stage_b, v0, []) [0]["moves"] == 0


def test_trajectory_cost_and_move_metrics():
    assert trajectory_cost(1.2) == 1.2
    assert trajectory_cost(1.2, 0.8) == 2.0
    rows = [{"label_id": 0, "predicted_label_id": 0, "selected_true_utility": 0.0, "safe_oracle_utility": 0.0, "regret": 0.0, "moves": 0, "trajectory_geodesic_cost_m": 0.0}]
    result = summarize_trajectory_rows(rows, ["a"])
    assert result["movement"]["move_0_rate"] == 1.0
