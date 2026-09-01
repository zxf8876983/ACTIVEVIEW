"""Focused invariants for EXP035--EXP037 graph/field helpers."""

import numpy as np
import pytest

from activeview.active_view.stage_d_dense_campaign import (
    VIEW_COUNT,
    fit_bayesian_linear,
    gmrf_smooth,
    graph_edges,
    graph_laplacian,
    relative_view_descriptor,
    viewpoint_azimuth,
    viewpoint_radius,
)


def test_fixed_4x8_graph_has_wrap_and_radial_edges() -> None:
    edges = set(graph_edges())
    assert (0, 7) in edges
    assert (0, 8) in edges
    assert (7, 15) in edges
    assert all(0 <= node < VIEW_COUNT for edge in edges for node in edge)


def test_graph_laplacian_is_symmetric_positive_semidefinite() -> None:
    laplacian = graph_laplacian()
    assert np.allclose(laplacian, laplacian.T)
    assert np.linalg.eigvalsh(laplacian).min() >= -1e-8
    assert np.allclose(laplacian.sum(axis=1), 0.0)


def test_gmrf_closed_form_preserves_constant_field() -> None:
    values = np.full(VIEW_COUNT, 2.0)
    np.testing.assert_allclose(gmrf_smooth(values), values)


def test_viewpoint_topology_mapping() -> None:
    assert viewpoint_radius(0) == 1.5
    assert viewpoint_radius(31) == 3.0
    assert viewpoint_azimuth(7) == 315.0
    assert viewpoint_azimuth(8) == 0.0


def test_legal_descriptor_has_no_target_fields() -> None:
    positions = np.zeros((VIEW_COUNT, 3), dtype=np.float32)
    descriptor = relative_view_descriptor(positions, np.zeros(3), 3)
    assert descriptor.shape == (9,)
    assert np.isfinite(descriptor).all()


def test_bayesian_posterior_dimensions_and_psd() -> None:
    rng = np.random.default_rng(42)
    features = np.c_[np.ones(12), rng.normal(size=(12, 3))]
    targets = rng.normal(size=12)
    posterior = fit_bayesian_linear(features, targets)
    assert posterior.weights.shape == (4,)
    assert posterior.covariance.shape == (4, 4)
    assert np.linalg.eigvalsh(posterior.covariance).min() >= -1e-8
    mean, sigma = posterior.predict(features[:2])
    assert mean.shape == sigma.shape == (2,)
    assert np.all(sigma >= 0)


def test_invalid_gmrf_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        gmrf_smooth(np.zeros(31))
