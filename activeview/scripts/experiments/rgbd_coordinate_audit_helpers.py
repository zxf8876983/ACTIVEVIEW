"""Small, deterministic helpers for the RGB-D coordinate sanity audit."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.scripts.experiments.rgbd_human_state_helpers import (
    HFOV_DEG,
    IMAGE_SIZE,
    SENSOR_HEIGHT,
    camera_backproject,
    rotation_wxyz_to_matrix,
)


def camera_project(
    world: Sequence[float], position: np.ndarray, rotation_wxyz: np.ndarray
) -> tuple[np.ndarray, float]:
    """Project a world point with the exact inverse of ``camera_backproject``."""
    focal = 0.5 * IMAGE_SIZE / math.tan(math.radians(HFOV_DEG) * 0.5)
    sensor = np.asarray(position, dtype=np.float64) + np.asarray([0.0, SENSOR_HEIGHT, 0.0])
    camera = rotation_wxyz_to_matrix(rotation_wxyz).T @ (np.asarray(world, dtype=np.float64) - sensor)
    depth = float(-camera[2])
    if not np.isfinite(depth) or depth <= 0.0:
        return np.asarray([np.nan, np.nan], dtype=np.float64), depth
    pixel = np.asarray(
        [0.5 * IMAGE_SIZE + focal * camera[0] / depth,
         0.5 * IMAGE_SIZE - focal * camera[1] / depth],
        dtype=np.float64,
    )
    return pixel, depth


def project_backproject_error(
    world: Sequence[float], position: np.ndarray, rotation_wxyz: np.ndarray
) -> float:
    pixel, depth = camera_project(world, position, rotation_wxyz)
    if not np.isfinite(pixel).all():
        return float("nan")
    _, reconstructed = camera_backproject(pixel, depth, position, rotation_wxyz)
    return float(np.linalg.norm(reconstructed.astype(np.float64) - np.asarray(world, dtype=np.float64)))


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def classification_metrics(labels: Sequence[int], predictions: Sequence[int], num_classes: int = 12) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    matrix = np.bincount(target * num_classes + pred, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    f1_values: list[float] = []
    per_class: dict[str, Any] = {}
    for cls in range(num_classes):
        tp = float(matrix[cls, cls]); support = float(matrix[cls].sum()); predicted = float(matrix[:, cls].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(cls)] = {"support": int(support), "recall": recall, "f1": f1}
    return {"n": int(target.size), "accuracy": float(np.mean(target == pred)) if target.size else 0.0, "macro_f1": float(np.mean(f1_values)), "confusion_matrix": matrix.tolist(), "per_class": per_class}


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(rx, ry)[0, 1])


def forward_basis(rotation_wxyz: Sequence[float]) -> dict[str, list[float]]:
    """Return Habitat camera basis vectors in world coordinates."""
    matrix = rotation_wxyz_to_matrix(rotation_wxyz)
    return {"forward_minus_z": (matrix @ np.asarray([0.0, 0.0, -1.0])).tolist(), "right_plus_x": (matrix @ np.asarray([1.0, 0.0, 0.0])).tolist(), "up_plus_y": (matrix @ np.asarray([0.0, 1.0, 0.0])).tolist()}


def yaw_wrap(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))
