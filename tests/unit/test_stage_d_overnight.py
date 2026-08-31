"""Focused tests for the frozen EXP032--EXP034 overnight audit helpers."""

import numpy as np
import pytest

from activeview.scripts.analyze_stage_d_overnight import (
    _candidate_quality,
    _split_check,
    _standardize,
    _train_samples,
)


def _utility_row() -> dict:
    return {
        "current": {"entropy": 0.2, "correct": True},
        "candidates": [
            {"viewpoint_id": 2, "logp_true": -2.0, "entropy": 0.4, "correct": False},
        ],
    }


def test_split_check_requires_explicit_expected_split() -> None:
    with pytest.raises(ValueError):
        _split_check([{"episode_id": "missing"}], "train", "predictions")
    with pytest.raises(ValueError):
        _split_check([{"policy_split": "val"}], "train", "predictions")
    _split_check([{"policy_split": "TRAIN"}], "train", "predictions")


def test_candidate_quality_is_derived_from_frozen_logp_diagnostics() -> None:
    quality = _candidate_quality(_utility_row(), 2)
    np.testing.assert_allclose(quality, [-2.0 * -1, np.exp(-2.0), 0.4, 0.0])


def test_train_samples_exclude_future_quality_when_not_requested() -> None:
    row = {
        "episode_id": "e",
        "base": np.asarray([1.0, 2.0], dtype=np.float32),
        "geometry": np.asarray([[3.0, 4.0]], dtype=np.float32),
        "ids": [2],
        "targets": np.asarray([5.0], dtype=np.float32),
        "quality": np.asarray([[6.0, 7.0, 8.0, 1.0]], dtype=np.float32),
    }
    observable, target, _ = _train_samples([row], include_quality=False)
    privileged, _, _ = _train_samples([row], include_quality=True)
    assert observable.shape == (1, 4)
    assert privileged.shape == (1, 8)
    assert target.tolist() == [5.0]
    np.testing.assert_allclose(observable[0], [1.0, 2.0, 3.0, 4.0])


def test_standardize_uses_train_statistics_for_validation() -> None:
    train = np.asarray([[0.0], [2.0]], dtype=np.float32)
    validation = np.asarray([[4.0]], dtype=np.float32)
    train_scaled, validation_scaled = _standardize(train, validation)
    np.testing.assert_allclose(train_scaled.ravel(), [-1.0, 1.0])
    np.testing.assert_allclose(validation_scaled.ravel(), [3.0])
