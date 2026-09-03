import numpy as np

from activeview.evaluation.evaluator import summarize_trajectory_rows
from activeview.methods.active_view.geometry import (
    order_candidates,
    semantic_delta,
    second_step_decision,
    second_step_geometry,
    second_step_utility,
    trajectory_cost,
)


def test_second_step_utility_and_semantic_delta_are_defined_from_s1():
    s0 = np.zeros(275, dtype=np.float32)
    s1 = np.ones(275, dtype=np.float32)
    assert second_step_utility(-1.0, -2.5) == 1.5
    assert semantic_delta(s0, s1).shape == (19,)
    assert np.allclose(semantic_delta(s0, s1), 1.0)


def test_candidate_order_is_deterministic():
    assert order_candidates([1.0, 1.0, 0.5], [3, 2, 1], [2.0, 1.0, 1.0]) == [2, 3, 1]


def test_second_step_geometry_uses_stage_a_radial_relative_azimuth():
    geometry = second_step_geometry(
        s1_position=[0.0, 0.0, 0.0], s1_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0], target_snapped_position=[1.0, 0.0, 0.0],
        target_geodesic=1.0, placement_position=[0.0, 0.0, 0.0], relative_azimuth_deg=90.0,
    )
    assert np.isclose(geometry[5], 1.0, atol=1e-6)
    assert np.isclose(geometry[6], 0.0, atol=1e-6)


def test_second_step_decision_strictly_uses_positive_utility():
    assert second_step_decision([0.2, 0.1], [2, 3], [1.0, 2.0]) == (False, 2, 0.2)
    assert second_step_decision([0.0, -0.1], [2, 3], [1.0, 2.0])[0] is True


def test_trajectory_cost_and_move_metrics():
    assert trajectory_cost(1.2) == 1.2
    assert trajectory_cost(1.2, 0.8) == 2.0
    rows = [{"label_id": 0, "predicted_label_id": 0, "selected_true_utility": 0.0, "safe_oracle_utility": 0.0, "regret": 0.0, "moves": 0, "trajectory_geodesic_cost_m": 0.0}]
    result = summarize_trajectory_rows(rows, ["a"])
    assert result["movement"]["move_0_rate"] == 1.0
