"""Unit tests for the VideoPose3D camera-to-gravity coordinate conversion."""

from __future__ import annotations

import numpy as np

from activeview.dataset.babel_clean_dataset_generator import (
    transform_camera_sequence_to_gravity,
)
from activeview.dataset.humanoid_grounding import select_floor_height


class _Hit:
    def __init__(self, y: float) -> None:
        self.point = np.array([0.0, y, 0.0], dtype=np.float32)


def test_h36m_down_and_forward_axes_are_flipped_to_habitat_gravity() -> None:
    points = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 1, axis=0)

    converted = transform_camera_sequence_to_gravity(points, transforms)

    np.testing.assert_allclose(converted, [[[1.0, -2.0, -3.0]]])


def test_camera_yaw_does_not_change_gravity_height() -> None:
    angle = np.pi / 2.0
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    transforms = np.eye(4, dtype=np.float32)[None]
    transforms[0, :3, :3] = rotation
    points = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)

    converted = transform_camera_sequence_to_gravity(points, transforms)

    np.testing.assert_allclose(converted[0, 0, 1], -2.0)
    np.testing.assert_allclose(np.linalg.norm(converted[0, 0]), np.linalg.norm(points[0, 0]))


def test_floor_selector_ignores_overhead_and_lower_level_hits() -> None:
    hits = [_Hit(5.83), _Hit(3.10), _Hit(2.04), _Hit(0.01)]

    np.testing.assert_allclose(select_floor_height(hits, reference_y=3.16), 3.10, atol=1e-6)


def test_floor_selector_falls_back_to_reference_without_hits() -> None:
    np.testing.assert_allclose(select_floor_height([], reference_y=1.25), 1.25)
