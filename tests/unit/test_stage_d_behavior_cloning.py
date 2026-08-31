import numpy as np
import torch

from activeview.active_view.stage_d_behavior_cloning import (
    SpatialRGBBehaviorCloner,
    oracle_action_index,
    select_behavior_action,
)
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, build_stage_d_trajectories


def _record() -> dict:
    return {
        "episode_id": "e0", "record_id": "r0", "policy_split": "val", "scene_id": "s", "region": "bedroom", "label_id": 0,
        "current": {"viewpoint_id": 0, "predicted_label_id": 0, "entropy": 0.1, "correct": True},
        "candidates": [
            {"viewpoint_id": 1, "utility": 2.0, "geodesic_distance_m": 1.0, "predicted_label_id": 1, "entropy": 0.2},
            {"viewpoint_id": 2, "utility": 1.0, "geodesic_distance_m": 2.0, "predicted_label_id": 2, "entropy": 0.2},
        ],
        "oracle": {"candidate_oracle_viewpoint_id": 1, "safe_oracle_viewpoint_id": 1, "safe_oracle_utility": 2.0, "safe_oracle_stays": False},
    }


def test_oracle_label_matches_frozen_fixed_first_oracle() -> None:
    record = _record()
    cache = [{"episode_id": "e0", "remaining_candidate_ids": [1, 2], "second_step_utility_targets": [1.0, -1.0], "second_step_candidate_geodesic": [1.0, 2.0]}]
    v0 = [{"episode_id": "e0", "predicted_stays": False, "predicted_candidate_viewpoint_id": 1}]
    frozen = build_fixed_first_oracle([record], v0, cache)
    assert oracle_action_index(cache[0]["second_step_utility_targets"]) == 1
    assert frozen[0]["moves"] == 2 and frozen[0]["selected_viewpoint_id"] == 1


def test_stay_zero_and_candidate_ties_follow_cache_order() -> None:
    assert oracle_action_index([-0.1, 0.0]) == 0
    assert oracle_action_index([1.0, 1.0]) == 1


def test_behavior_model_has_trainable_stay_head_and_three_logits() -> None:
    model = SpatialRGBBehaviorCloner()
    assert any(parameter.requires_grad for parameter in model.stay_head.parameters())
    logits = model(
        torch.zeros(2, 275), torch.zeros(2, 275), torch.zeros(2, 19),
        torch.zeros(2, 2, 11), torch.ones(2, 2, dtype=torch.bool),
        torch.zeros(2, 16, 768), torch.zeros(2, 16, 768),
    )
    assert logits.shape == (2, 3)


def test_inference_is_deterministic_stay_first_and_candidates_independent() -> None:
    assert select_behavior_action([2.0, 2.0, 2.0], [8, 3]) == (0, None)
    assert select_behavior_action([0.0, 3.0, 1.0], [8, 3]) == (1, 8)
    assert select_behavior_action([0.0, 1.0, 0.0], [8]) == (1, 8)


def test_v0_stay_never_enters_second_step_trajectory() -> None:
    record = _record()
    v0 = [{"episode_id": "e0", "predicted_stays": True, "predicted_candidate_viewpoint_id": None}]
    cache = [{"episode_id": "e0", "remaining_candidate_ids": [1], "second_step_utility_targets": [5.0], "second_step_candidate_geodesic": [1.0]}]
    second = [{"episode_id": "e0", "predicted_stays": False, "predicted_candidate_viewpoint_id": 1}]
    result = build_stage_d_trajectories([record], v0, cache, second)
    assert result[0]["moves"] == 0


def test_true_utility_not_required_by_model_input_signature() -> None:
    model = SpatialRGBBehaviorCloner().eval()
    with torch.inference_mode():
        output_a = model(
            torch.randn(1, 275), torch.randn(1, 275), torch.randn(1, 19), torch.randn(1, 2, 11), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16, 768), torch.randn(1, 16, 768)
        )
        output_b = model(
            torch.randn(1, 275), torch.randn(1, 275), torch.randn(1, 19), torch.randn(1, 2, 11), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16, 768), torch.randn(1, 16, 768)
        )
    assert output_a.shape == output_b.shape == (1, 3)
