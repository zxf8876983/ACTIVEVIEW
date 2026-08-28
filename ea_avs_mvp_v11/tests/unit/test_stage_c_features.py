import numpy as np

from ea_avs_mvp_v11.active_view.stage_c_features import (
    CANDIDATE_GEOMETRY_DIM,
    CURRENT_FEATURE_DIM,
    candidate_geometry_features,
    current_state_features,
)


def test_current_feature_shape_and_geometry_shape():
    vector = current_state_features(np.zeros(256), np.log(np.ones(16) / 16), 0.5)
    assert vector.shape == (CURRENT_FEATURE_DIM,)
    candidate = {
        "relative_position": [1.0, 0.0, 0.0], "position": [1.0, 0.0, 0.0], "snapped_position": [1.0, 0.0, 0.0],
        "euclidean_distance_m": 1.0, "geodesic_distance_m": 1.2,
        "relative_azimuth_deg": 90.0,
    }
    geometry = candidate_geometry_features(candidate, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[1.0, 0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    assert geometry.shape == (CANDIDATE_GEOMETRY_DIM,)
    assert np.isfinite(geometry).all()


def test_geometry_uses_egocentric_delta_and_snapped_radius():
    candidate = {
        "position": [99.0, 0.0, 99.0], "snapped_position": [2.0, 0.0, 0.0],
        "euclidean_distance_m": 2.0, "geodesic_distance_m": 2.0,
        "relative_azimuth_deg": 0.0,
    }
    geometry = candidate_geometry_features(candidate, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[0.7071068, 0.0, 0.7071068, 0.0], placement_position=[0.0, 0.0, 0.0])
    assert np.allclose(geometry[:3], [0.0, 0.0, 2.0], atol=1e-5)
    assert geometry[9] == 2.0


def test_feature_constructor_ignores_labels_and_future_candidate_perception():
    current_inputs = (np.arange(256, dtype=np.float32), np.log(np.ones(16, dtype=np.float32) / 16.0), 0.7)
    first = current_state_features(*current_inputs)
    second = current_state_features(*current_inputs)
    assert np.array_equal(first, second)
    candidate = {
        "snapped_position": [1.0, 0.0, 0.0], "position": [100.0, 0.0, 100.0],
        "euclidean_distance_m": 1.0, "geodesic_distance_m": 1.0,
        "relative_azimuth_deg": 90.0, "candidate_entropy": 0.01,
        "candidate_confidence": 0.99, "candidate_utility": 10.0,
    }
    altered = {**candidate, "candidate_entropy": 9.0, "candidate_confidence": 0.01, "candidate_utility": -10.0}
    first_geometry = candidate_geometry_features(candidate, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[1.0, 0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    second_geometry = candidate_geometry_features(altered, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[1.0, 0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    assert np.array_equal(first_geometry, second_geometry)
