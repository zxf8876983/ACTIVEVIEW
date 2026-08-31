import numpy as np

from activeview.active_view.stage_d_visibility import (
    FEATURE_NAMES,
    analytic_candidate_order,
    candidate_visibility_features,
)


def test_visibility_feature_schema_and_front_projection():
    joints = np.asarray([[0.0, 1.1, -2.0], [0.2, 1.2, -2.0]], dtype=np.float32)
    points = np.asarray([[0.0, 1.1, -1.0]], dtype=np.float32)
    features = candidate_visibility_features(points, joints, candidate_position=[0.0, 0.0, 0.0], candidate_rotation_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert features.shape == (len(FEATURE_NAMES),)
    assert 0.0 <= float(features[1]) <= 1.0
    assert float(features[2]) > 0.0


def test_analytic_order_uses_visibility_then_area_then_p2_tie():
    first = np.zeros(17, dtype=np.float32); second = np.zeros(17, dtype=np.float32)
    first[0], first[1], first[13] = 0.8, 0.9, 0.1
    second[0], second[1], second[13] = 0.8, 0.9, 0.2
    assert analytic_candidate_order({2: first, 3: second}) == 3


def test_analytic_order_final_tie_prefers_lower_candidate_id():
    value = np.zeros(17, dtype=np.float32)
    assert analytic_candidate_order({3: value, 2: value}) == 2
