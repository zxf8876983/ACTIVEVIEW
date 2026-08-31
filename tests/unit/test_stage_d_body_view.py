import numpy as np
import pytest

from activeview.active_view.stage_d_body_view import FEATURE_NAMES, body_view_features
from activeview.scripts.analyze_stage_d_human_viewpoint import _assert_split


def _anchors() -> np.ndarray:
    return np.asarray([[0, 1.0, -2.0], [0.3, 1.2, -2.0], [-0.3, 1.2, -2.0], [0, 0.5, -2.0], [0.2, 0.1, -2.0]], dtype=np.float32)


def test_body_view_features_are_deterministic_and_finite():
    first = body_view_features(_anchors(), [0, 1, -2], [0, 0, 0], [1, 0, 0, 0])
    second = body_view_features(_anchors(), [0, 1, -2], [0, 0, 0], [1, 0, 0, 0])
    assert first.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_candidate_side_changes_body_view_encoding():
    left = body_view_features(_anchors(), [0, 1, -2], [-1, 0, 0], [1, 0, 0, 0])
    right = body_view_features(_anchors(), [0, 1, -2], [1, 0, 0], [1, 0, 0, 0])
    assert not np.allclose(left, right)


def test_exp031_split_validation_is_fail_closed():
    with pytest.raises(ValueError):
        _assert_split([{"policy_split": "val"}], "train", "calibration")
    with pytest.raises(ValueError):
        _assert_split([{}], "val", "evaluation")
    _assert_split([{"policy_split": "train"}], "train", "calibration")
