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
        "relative_position": [1.0, 0.0, 0.0], "position": [1.0, 0.0, 0.0],
        "euclidean_distance_m": 1.0, "geodesic_distance_m": 1.2,
        "relative_azimuth_deg": 90.0,
    }
    geometry = candidate_geometry_features(candidate, current_position=[0.0, 0.0, 0.0], placement_position=[0.0, 0.0, 0.0])
    assert geometry.shape == (CANDIDATE_GEOMETRY_DIM,)
    assert np.isfinite(geometry).all()
