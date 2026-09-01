"""Small, deterministic helpers for the EXP035--EXP037 offline campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VIEW_COUNT = 32
RADIUS_COUNT = 4
AZIMUTH_COUNT = 8

ContextKey = tuple[str, str, str]


def context_key(row: Mapping[str, Any]) -> ContextKey:
    """Return the canonical scene/region/record identity for one row."""
    return (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))


def canonical_realpath(value: str | Path) -> str:
    """Resolve a source path without requiring it to exist."""
    return str(Path(value).expanduser().resolve(strict=False))


def index_by_context(rows: Sequence[Mapping[str, Any]], name: str = "rows") -> dict[ContextKey, Mapping[str, Any]]:
    """Index rows by context and reject duplicate context identities."""
    result: dict[ContextKey, Mapping[str, Any]] = {}
    for row in rows:
        key = context_key(row)
        if key in result:
            raise ValueError(f"Duplicate {name} context key: {key}")
        result[key] = row
    return result


def viewpoint_radius(viewpoint_id: int) -> float:
    return (1.5, 2.0, 2.5, 3.0)[int(viewpoint_id) // AZIMUTH_COUNT]


def viewpoint_azimuth(viewpoint_id: int) -> float:
    return float(int(viewpoint_id) % AZIMUTH_COUNT * 45.0)


def graph_edges() -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for node in range(VIEW_COUNT):
        radius, azimuth = node // AZIMUTH_COUNT, node % AZIMUTH_COUNT
        for other_azimuth in ((azimuth - 1) % AZIMUTH_COUNT, (azimuth + 1) % AZIMUTH_COUNT):
            other = radius * AZIMUTH_COUNT + other_azimuth
            edges.add(tuple(sorted((node, other))))
        if radius > 0:
            edges.add((node - AZIMUTH_COUNT, node))
    return sorted(edges)


def graph_laplacian() -> np.ndarray:
    adjacency = np.zeros((VIEW_COUNT, VIEW_COUNT), dtype=np.float64)
    for left, right in graph_edges():
        adjacency[left, right] = adjacency[right, left] = 1.0
    return np.diag(adjacency.sum(axis=1)) - adjacency


def gmrf_smooth(values: Sequence[float], lam: float = 0.25) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (VIEW_COUNT,):
        raise ValueError("GMRF values must have shape [32]")
    return np.linalg.solve(np.eye(VIEW_COUNT) + lam * graph_laplacian(), array)


def relative_view_descriptor(
    positions: np.ndarray, current_position: np.ndarray, viewpoint_id: int,
) -> np.ndarray:
    """Legal geometry descriptor derived from saved camera positions."""
    delta = np.asarray(positions[int(viewpoint_id)] - current_position, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    azimuth = np.deg2rad(viewpoint_azimuth(viewpoint_id))
    return np.asarray(
        [viewpoint_radius(viewpoint_id) / 3.0, np.sin(azimuth), np.cos(azimuth),
         float(delta[0]), float(delta[1]), float(delta[2]), distance,
         np.sin(np.arctan2(float(delta[0]), float(delta[2]))),
         np.cos(np.arctan2(float(delta[0]), float(delta[2])))],
        dtype=np.float32,
    )


def dense_regression_model(input_dim: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.GELU(),
        nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
    )


def train_dense_regressor(
    train_x: np.ndarray, train_y: np.ndarray, epochs: int = 20,
) -> tuple[object, float]:
    import torch

    torch.manual_seed(42)
    model = dense_regression_model(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss()
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.float32).reshape(-1, 1)
    final_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), 1024):
            index = order[start : start + 1024]
            loss = criterion(model(x[index]), y[index])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(index)
        final_loss = total / len(x)
    return model, final_loss


def predict_model(model: object, values: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        return model(torch.as_tensor(values, dtype=torch.float32)).reshape(-1).numpy()


def train_bradley_terry(
    train_x: np.ndarray, pair_left: np.ndarray, pair_right: np.ndarray,
    labels: np.ndarray, epochs: int = 20,
) -> tuple[object, float]:
    import torch
    from torch import nn

    torch.manual_seed(42)
    model = dense_regression_model(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    x = torch.as_tensor(train_x, dtype=torch.float32)
    left = torch.as_tensor(pair_left, dtype=torch.long)
    right = torch.as_tensor(pair_right, dtype=torch.long)
    target = torch.as_tensor(labels, dtype=torch.float32)
    final_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(left)); total = 0.0
        for start in range(0, len(left), 1024):
            idx = order[start : start + 1024]
            logits = model(x[left[idx]])[:, 0] - model(x[right[idx]])[:, 0]
            loss = criterion(logits, target[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(idx)
        final_loss = total / len(left)
    return model, final_loss


@dataclass(frozen=True)
class BayesianLinear:
    weights: np.ndarray
    covariance: np.ndarray
    residual_variance: float

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = features @ self.weights
        variance = np.einsum("ij,jk,ik->i", features, self.covariance, features)
        return mean, np.sqrt(np.maximum(variance * self.residual_variance, 0.0))


def fit_bayesian_linear(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> BayesianLinear:
    gram = features.T @ features + alpha * np.eye(features.shape[1])
    covariance = np.linalg.inv(gram)
    weights = covariance @ features.T @ targets
    residual = float(np.mean((features @ weights - targets) ** 2))
    return BayesianLinear(weights, covariance, max(residual, 1e-8))


def deterministic_oracle_action(values: Sequence[float]) -> int:
    return int(np.argmax(np.asarray([0.0, *values], dtype=np.float64)))


def binary_metrics(predicted: Iterable[bool], truth: Iterable[bool]) -> dict[str, float | int]:
    pred = np.asarray(list(predicted), dtype=bool); target = np.asarray(list(truth), dtype=bool)
    tp = int(np.sum(pred & target)); tn = int(np.sum(~pred & ~target)); fp = int(np.sum(pred & ~target)); fn = int(np.sum(~pred & target))
    return {"accuracy": float(np.mean(pred == target)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None}
