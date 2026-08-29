import numpy as np

from activeview.active_view.stage_c_experiment_features import (
    transform_geometry,
    variant_geometry_names,
)
from activeview.active_view.stage_c_features import CANDIDATE_GEOMETRY_NAMES


def _base_geometry() -> np.ndarray:
    # Two mirrored directions plus two candidates with repeated radii.
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0, 2.0, 2.1, -1.0, 0.0, 1.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 3.0, 3.0, 0.0, 1.0, 1.0, 3.0, 3.1, 0.1],
            [0.0, 0.0, -1.0, 3.0, 3.2, 0.0, -1.0, 1.0, 3.0, 3.1, 0.1],
        ],
        dtype=np.float32,
    )


def test_radius_ablation_removes_direct_radius_features_and_has_expected_dim():
    transformed = transform_geometry(_base_geometry(), "radius_ablation")
    names = variant_geometry_names("radius_ablation")
    assert transformed.shape == (4, 10)
    assert names[:8] == list(CANDIDATE_GEOMETRY_NAMES[:8])
    assert names[8:] == ["geodesic_zscore", "geodesic_rank"]
    assert not any("radius" in name for name in names)


def test_direction_features_are_circular_and_permutation_equivariant():
    base = _base_geometry()
    transformed = transform_geometry(base, "direction_geometry")
    permutation = np.asarray([2, 0, 3, 1])
    shuffled = transform_geometry(base[permutation], "direction_geometry")
    assert transformed.shape == (4, 15)
    assert np.allclose(transformed[permutation], shuffled, atol=1e-6)


def test_direction_features_wrap_angles_across_pi_boundary():
    base = _base_geometry()[:2].copy()
    base[:, 5] = np.sin(np.deg2rad([179.0, -179.0]))
    base[:, 6] = np.cos(np.deg2rad([179.0, -179.0]))
    transformed = transform_geometry(base, "direction_geometry")
    # The two directions are 2 degrees apart, not 358 degrees apart.
    assert np.all(transformed[:, 13] < np.deg2rad(3.0) / np.pi)


def test_candidate_relation_features_are_permutation_equivariant_and_finite():
    base = _base_geometry()
    transformed = transform_geometry(base, "candidate_relations")
    permutation = np.asarray([1, 3, 0, 2])
    shuffled = transform_geometry(base[permutation], "candidate_relations")
    assert transformed.shape == (4, 16)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[permutation], shuffled, atol=1e-6)
