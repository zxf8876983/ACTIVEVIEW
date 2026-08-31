"""Pure helpers for the EXP029 observed semantic BEV audit."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


BEV_CHANNELS = {
    "observed": 0,
    "occupied": 1,
    "free": 2,
    "human": 3,
    "wall": 4,
    "table": 5,
    "chair": 6,
    "bed": 7,
    "couch": 8,
    "cabinet": 9,
    "other_object": 10,
    "s0_position": 11,
    "s1_position": 12,
    "p2_position": 13,
    "p3_position": 14,
}
BEV_SHAPE = (15, 80, 80)
CELL_SIZE_M = 0.1
MAP_SIZE_M = 8.0


def normalize_category(name: str) -> str:
    value = " ".join(str(name).strip().lower().replace("_", " ").split())
    aliases = {"sofa": "couch", "kitchen cabinet": "cabinet"}
    value = aliases.get(value, value)
    return value if value in BEV_CHANNELS and value not in {"observed", "occupied", "free", "human"} else "other_object"


def world_to_s1(world: Sequence[float], s1_position: Sequence[float], rotation_matrix: np.ndarray) -> np.ndarray:
    point = np.asarray(world, dtype=np.float64) - np.asarray(s1_position, dtype=np.float64)
    return np.asarray(rotation_matrix, dtype=np.float64).T @ point


def bev_cell(egocentric_xyz: Sequence[float]) -> tuple[int, int] | None:
    x, z = float(egocentric_xyz[0]), float(egocentric_xyz[2])
    col = int(math.floor((x + MAP_SIZE_M / 2.0) / CELL_SIZE_M))
    row = int(math.floor((z + MAP_SIZE_M / 2.0) / CELL_SIZE_M))
    return (row, col) if 0 <= row < BEV_SHAPE[1] and 0 <= col < BEV_SHAPE[2] else None


def bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start; r1, c1 = end
    points: list[tuple[int, int]] = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    error = dc - dr
    while True:
        points.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return points
        twice = 2 * error
        if twice > -dr: error -= dr; c0 += sc
        if twice < dc: error += dc; r0 += sr


def project_depth_semantic(
    depth: np.ndarray,
    semantic: np.ndarray,
    *,
    render_camera: Any,
    s1_position: Sequence[float],
    s1_rotation_matrix: np.ndarray,
    semantic_channels: Mapping[int, int],
    pixel_stride: int = 8,
) -> np.ndarray:
    """Project observed rays into the fixed s1-centered BEV.

    Only depth endpoints are marked occupied.  Intermediate ray cells are
    free, and occupied wins when multiple rays collide in one cell.
    """
    import magnum as mn

    values = np.asarray(depth)
    labels = np.asarray(semantic)
    if values.shape != labels.shape or values.ndim != 2:
        raise ValueError("depth and semantic must be aligned 2-D arrays")
    bev = np.zeros(BEV_SHAPE, dtype=np.uint8)
    center = render_camera.unproject(mn.Vector2i(values.shape[1] // 2, values.shape[0] // 2), normalized=False)
    camera_cell = bev_cell(world_to_s1(np.asarray(center.origin), s1_position, s1_rotation_matrix))
    if camera_cell is None:
        # A valid s0 camera can lie outside the s1-centered 8 m window.  Clip
        # its ray origin to the nearest boundary so in-window traversed cells
        # remain observed/free while preserving endpoint semantics.
        camera_xyz = world_to_s1(np.asarray(center.origin), s1_position, s1_rotation_matrix)
        camera_cell = (int(np.clip((camera_xyz[2] + MAP_SIZE_M / 2) / CELL_SIZE_M, 0, 79)), int(np.clip((camera_xyz[0] + MAP_SIZE_M / 2) / CELL_SIZE_M, 0, 79)))
    occupied: set[tuple[int, int]] = set()
    free: set[tuple[int, int]] = set()
    semantic_at: dict[tuple[int, int], int] = {}
    for y in range(0, values.shape[0], pixel_stride):
        for x in range(0, values.shape[1], pixel_stride):
            distance = float(values[y, x])
            if not np.isfinite(distance) or distance <= 0.0:
                continue
            ray = render_camera.unproject(mn.Vector2i(x, y), normalized=False)
            endpoint = np.asarray(ray.origin, dtype=np.float64) + np.asarray(ray.direction, dtype=np.float64) * distance
            cell = bev_cell(world_to_s1(endpoint, s1_position, s1_rotation_matrix))
            if cell is None:
                continue
            traversed = bresenham(camera_cell, cell)
            free.update(traversed[:-1])
            semantic_id = int(labels[y, x])
            if semantic_id > 0:
                occupied.add(cell)
                semantic_at[cell] = int(semantic_channels.get(semantic_id, BEV_CHANNELS["other_object"]))
            else:
                free.add(cell)
    free.difference_update(occupied)
    for row, col in free | occupied:
        bev[BEV_CHANNELS["observed"], row, col] = 1
    for row, col in free:
        bev[BEV_CHANNELS["free"], row, col] = 1
    for row, col in occupied:
        bev[BEV_CHANNELS["occupied"], row, col] = 1
        bev[BEV_CHANNELS["free"], row, col] = 0
        bev[semantic_at.get((row, col), BEV_CHANNELS["other_object"]), row, col] = 1
    if np.any((bev[BEV_CHANNELS["occupied"]] > 0) & (bev[BEV_CHANNELS["free"]] > 0)):
        raise ValueError("occupied/free BEV conflict")
    return bev


def project_world_samples(
    points_world: np.ndarray,
    semantic: np.ndarray,
    *,
    camera_world: Sequence[float],
    s1_position: Sequence[float],
    s1_rotation_matrix: np.ndarray,
    semantic_channels: Mapping[int, int],
) -> np.ndarray:
    """Project cached Habitat-unprojected world samples into one episode BEV."""
    points = np.asarray(points_world, dtype=np.float64)
    labels = np.asarray(semantic)
    if points.ndim != 3 or points.shape[-1] != 3 or labels.shape != points.shape[:2]:
        raise ValueError("cached world samples and semantic labels are misaligned")
    bev = np.zeros(BEV_SHAPE, dtype=np.uint8)
    camera_cell = bev_cell(world_to_s1(camera_world, s1_position, s1_rotation_matrix))
    if camera_cell is None:
        xyz = world_to_s1(camera_world, s1_position, s1_rotation_matrix)
        camera_cell = (int(np.clip((xyz[2] + MAP_SIZE_M / 2) / CELL_SIZE_M, 0, 79)), int(np.clip((xyz[0] + MAP_SIZE_M / 2) / CELL_SIZE_M, 0, 79)))
    occupied: set[tuple[int, int]] = set(); free: set[tuple[int, int]] = set(); semantic_at: dict[tuple[int, int], int] = {}
    for index in np.ndindex(points.shape[:2]):
        point = points[index]
        if not np.isfinite(point).all():
            continue
        cell = bev_cell(world_to_s1(point, s1_position, s1_rotation_matrix))
        if cell is None:
            continue
        traversed = bresenham(camera_cell, cell); free.update(traversed[:-1])
        semantic_id = int(labels[index])
        if semantic_id > 0:
            occupied.add(cell); semantic_at[cell] = int(semantic_channels.get(semantic_id, BEV_CHANNELS["other_object"]))
        else:
            free.add(cell)
    free.difference_update(occupied)
    for row, col in free:
        bev[0, row, col] = 1; bev[2, row, col] = 1
    for row, col in occupied:
        bev[0, row, col] = 1; bev[1, row, col] = 1; bev[semantic_at.get((row, col), 10), row, col] = 1
    if np.any(bev[1] & bev[2]):
        raise ValueError("occupied/free BEV conflict")
    return bev


def add_markers(
    bev: np.ndarray,
    positions: Mapping[str, Sequence[float]],
    *,
    s1_position: Sequence[float],
    s1_rotation_matrix: np.ndarray,
    human_position: Sequence[float] | None = None,
) -> dict[str, tuple[int, int]]:
    channels = {"s0": 11, "s1": 12, "p2": 13, "p3": 14}
    cells: dict[str, tuple[int, int]] = {}
    for name, position in positions.items():
        # The finite local map is centered on s1. Valid episode markers may
        # lie outside its window; omit them rather than clamping to a false
        # boundary location.
        cell = bev_cell(world_to_s1(position, s1_position, s1_rotation_matrix))
        if cell is not None:
            cells[name] = cell; bev[channels[name], cell[0], cell[1]] = 1
    if human_position is not None:
        cell = bev_cell(world_to_s1(human_position, s1_position, s1_rotation_matrix))
        if cell is not None:
            bev[BEV_CHANNELS["human"], cell[0], cell[1]] = 1
            cells["human"] = cell
    return cells


def validate_bev(bev: np.ndarray) -> dict[str, Any]:
    values = np.asarray(bev)
    if values.shape != BEV_SHAPE or values.dtype != np.uint8:
        raise ValueError(f"invalid BEV schema: {values.shape} {values.dtype}")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("BEV channels must be binary")
    conflict = int(np.count_nonzero(values[1] & values[2]))
    if conflict:
        raise ValueError(f"occupied/free conflict count={conflict}")
    if int(values[0].sum()) <= 0:
        raise ValueError("BEV has no observed cells")
    return {"shape": list(values.shape), "dtype": str(values.dtype), "observed_cells": int(values[0].sum()), "occupied_cells": int(values[1].sum()), "free_cells": int(values[2].sum()), "occupied_free_conflict_count": conflict}


def pool_bev(bev: np.ndarray, grid: int = 10) -> np.ndarray:
    values = np.asarray(bev, dtype=np.float32)
    if values.shape != BEV_SHAPE or BEV_SHAPE[1] % grid or BEV_SHAPE[2] % grid:
        raise ValueError("BEV must be [15,80,80] and divisible by pooling grid")
    factor = BEV_SHAPE[1] // grid
    return values.reshape(BEV_SHAPE[0], grid, factor, grid, factor).mean(axis=(2, 4)).reshape(-1)
