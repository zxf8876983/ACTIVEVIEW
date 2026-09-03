import numpy as np

from activeview.active_view.policy_features import (
    BASE_CANDIDATE_GEOMETRY_DIM,
    CANDIDATE_GEOMETRY_DIM,
    CURRENT_FEATURE_DIM,
    RELATIVE_CANDIDATE_GEOMETRY_DIM,
    RELATIVE_CANDIDATE_GEOMETRY_NAMES,
    candidate_set_relative_features,
    candidate_geometry_features,
    current_state_features,
    schema_metadata,
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
        "position": [99.0, 0.0, 99.0], "snapped_position": [2.0, 0.0, 0.0], "relative_position": [2.0, 0.0, 0.0],
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
        "snapped_position": [1.0, 0.0, 0.0], "position": [100.0, 0.0, 100.0], "relative_position": [1.0, 0.0, 0.0],
        "euclidean_distance_m": 1.0, "geodesic_distance_m": 1.0,
        "relative_azimuth_deg": 90.0, "candidate_entropy": 0.01,
        "candidate_confidence": 0.99, "candidate_utility": 10.0,
    }
    altered = {**candidate, "candidate_entropy": 9.0, "candidate_confidence": 0.01, "candidate_utility": -10.0}
    first_geometry = candidate_geometry_features(candidate, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[1.0, 0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    second_geometry = candidate_geometry_features(altered, current_position=[0.0, 0.0, 0.0], current_rotation_wxyz=[1.0, 0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    assert np.array_equal(first_geometry, second_geometry)


def _base_geometry(radii, geodesics, deltas=None):
    deltas = deltas if deltas is not None else [radius - 2.0 for radius in radii]
    rows = np.zeros((len(radii), BASE_CANDIDATE_GEOMETRY_DIM), dtype=np.float32)
    rows[:, 4] = geodesics
    rows[:, 9] = radii
    rows[:, 10] = deltas
    return rows


def test_relative_geometry_ranks_and_zscores_are_finite():
    relative = candidate_set_relative_features(
        _base_geometry([1.0, 2.0, 4.0], [3.0, 1.0, 2.0])
    )

    assert relative.shape == (3, RELATIVE_CANDIDATE_GEOMETRY_DIM)
    assert np.isfinite(relative).all()
    assert np.array_equal(np.argsort(relative[:, 1]), np.array([0, 1, 2]))
    assert np.array_equal(np.argsort(relative[:, 3]), np.array([1, 2, 0]))
    assert RELATIVE_CANDIDATE_GEOMETRY_NAMES == (
        "radius_zscore", "radius_rank", "geodesic_zscore", "geodesic_rank",
        "delta_radius_normalized",
    )


def test_relative_geometry_is_permutation_equivariant():
    base = _base_geometry([1.0, 2.0, 4.0], [3.0, 1.0, 2.0])
    permutation = np.array([2, 0, 1])
    first = candidate_set_relative_features(base)
    shuffled = candidate_set_relative_features(base[permutation])
    assert np.allclose(first[permutation], shuffled)


def test_relative_geometry_ties_share_rank_and_remain_permutation_equivariant():
    base = _base_geometry([1.5, 1.5, 3.0, 3.0], [1.0, 1.0, 2.0, 2.0])
    first = candidate_set_relative_features(base)
    permutation = np.array([3, 1, 0, 2])
    shuffled = candidate_set_relative_features(base[permutation])

    assert np.allclose(first[permutation], shuffled)
    assert first[0, 1] == first[1, 1]
    assert first[2, 1] == first[3, 1]
    assert first[0, 3] == first[1, 3]
    assert first[2, 3] == first[3, 3]


def test_relative_geometry_single_candidate_or_constant_set_is_finite():
    single = candidate_set_relative_features(_base_geometry([2.0], [1.0]))
    constant = candidate_set_relative_features(
        _base_geometry([2.0, 2.0], [1.0, 1.0])
    )
    assert np.isfinite(single).all()
    assert np.isfinite(constant).all()
    assert np.allclose(single[:, :4], 0.0)
    assert np.allclose(constant[:, [0, 2]], 0.0)
    assert np.isfinite(constant[:, [1, 3]]).all()


def test_relative_schema_dimension_matches_feature_names():
    schema = schema_metadata(include_relative_features=True)
    assert schema["candidate_geometry_dim"] == CANDIDATE_GEOMETRY_DIM + RELATIVE_CANDIDATE_GEOMETRY_DIM
    assert len(schema["candidate_geometry_names"]) == schema["candidate_geometry_dim"]
