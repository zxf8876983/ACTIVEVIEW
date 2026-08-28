"""Checks that canonical alignment removes yaw without erasing gravity pose."""

import math

import numpy as np

from activeview.perception.skeleton_definition import get_skeleton_definition
from activeview.perception.skeleton_normalizer import SkeletonNormalizer
from activeview.dataset.babel_clean_dataset_generator import compose_root_rotation


def _lying_h36m_sequence(frames: int = 9) -> np.ndarray:
    sequence = np.zeros((frames, 17, 3), dtype=np.float32)
    sequence[:, 1] = [-0.2, 0.0, 0.0]
    sequence[:, 4] = [0.2, 0.0, 0.0]
    sequence[:, 7] = [0.6, 0.1, 0.0]
    sequence[:, 8] = [1.0, 0.1, 0.0]
    sequence[:, 11] = [1.0, 0.35, 0.0]
    sequence[:, 14] = [1.0, -0.35, 0.0]
    return sequence


def _rotate_yaw(sequence: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    rotation = np.array(
        [
            [math.cos(angle), 0.0, -math.sin(angle)],
            [0.0, 1.0, 0.0],
            [math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float32,
    )
    return np.matmul(sequence, rotation.T)


def test_yaw_alignment_preserves_gravity_axis() -> None:
    normalizer = SkeletonNormalizer(
        skel_def=get_skeleton_definition(backend="h36m_17")
    )
    sequence = _rotate_yaw(_lying_h36m_sequence(), 67.0)

    aligned = normalizer.align_to_canonical_frame(sequence)

    np.testing.assert_allclose(aligned[..., 1], sequence[..., 1], atol=1e-7)
    assert float(np.mean(aligned[:, 8, 0])) > 0.9
    assert float(np.max(np.abs(aligned[:, 8, 1]))) < 0.11


def test_yaw_alignment_is_view_invariant() -> None:
    normalizer = SkeletonNormalizer(
        skel_def=get_skeleton_definition(backend="h36m_17")
    )
    reference = normalizer.align_to_canonical_frame(_lying_h36m_sequence())

    for degrees in (45.0, 90.0, 180.0, 270.0):
        rotated = _rotate_yaw(_lying_h36m_sequence(), degrees)
        aligned = normalizer.align_to_canonical_frame(rotated)
        np.testing.assert_allclose(aligned, reference, atol=1e-5)


def test_root_roll_is_preserved_when_scene_yaw_is_added() -> None:
    root_roll = np.eye(4, dtype=np.float32)
    root_roll[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    composed = compose_root_rotation(root_roll, scene_yaw_deg=73.0)
    rotated_torso = composed[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert abs(float(rotated_torso[1])) < 1e-7
    np.testing.assert_allclose(np.linalg.norm(rotated_torso), 1.0, atol=1e-7)
    np.testing.assert_allclose(composed[:3, 3], 0.0, atol=1e-7)
