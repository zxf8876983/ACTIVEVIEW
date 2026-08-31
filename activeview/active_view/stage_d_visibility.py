"""Small deterministic visibility geometry helpers for EXP030.

The module deliberately contains no Habitat or learning code.  It converts
observed depth endpoints and candidate poses into candidate-conditioned
visibility statistics, and provides a separate full-scene raycast helper for
the privileged diagnostic upper bound.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "visible_joint_fraction", "inframe_joint_fraction", "blocked_joint_fraction",
    "upper_body_visible_fraction", "lower_body_visible_fraction",
    "left_body_visible_fraction", "right_body_visible_fraction",
    "mean_ray_clearance", "min_ray_clearance", "mean_occluder_gap",
    "target_distance", "projected_bbox_width", "projected_bbox_height",
    "normalized_bbox_area", "target_center_offset", "candidate_azimuth",
    "candidate_elevation",
)
RAY_BLOCK_RADIUS_M = 0.12
IMAGE_SIZE = 256
HFOV_DEG = 75.0
SENSOR_HEIGHT_M = 1.1


def rotation_matrix(rotation_wxyz: Sequence[float]) -> np.ndarray:
    """Return a Habitat quaternion's 3x3 active rotation matrix."""
    values = np.asarray(rotation_wxyz, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("rotation_wxyz must be a finite [w,x,y,z] vector")
    try:
        import quaternion
    except ImportError:
        w, x, y, z = values / max(float(np.linalg.norm(values)), 1e-12)
        return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)
    return np.asarray(quaternion.as_rotation_matrix(quaternion.from_float_array(values)), dtype=np.float64)


def camera_origin(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64) + np.asarray(rotation, dtype=np.float64) @ np.array([0.0, SENSOR_HEIGHT_M, 0.0])


def _project_joints(
    joints_world: np.ndarray,
    candidate_position: Sequence[float],
    candidate_rotation_wxyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = rotation_matrix(candidate_rotation_wxyz)
    origin = camera_origin(candidate_position, rotation)
    camera = (rotation.T @ (np.asarray(joints_world, dtype=np.float64) - origin).T).T
    depth = -camera[:, 2]
    focal = IMAGE_SIZE / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
    safe = np.maximum(depth, 1e-9)
    u = focal * camera[:, 0] / safe + IMAGE_SIZE / 2.0
    v = IMAGE_SIZE / 2.0 - focal * camera[:, 1] / safe
    inframe = (depth > 0.0) & (u >= 0.0) & (u < IMAGE_SIZE) & (v >= 0.0) & (v < IMAGE_SIZE)
    return origin, camera, np.stack([u, v], axis=1), inframe


def _ray_clearance_and_gap(
    origin: np.ndarray,
    joint: np.ndarray,
    observed_points: np.ndarray,
    radius: float,
) -> tuple[float, float | None, bool]:
    vector = joint - origin
    length = float(np.linalg.norm(vector))
    if length <= 1e-9 or observed_points.size == 0:
        return float("inf"), None, False
    direction = vector / length
    relative = observed_points - origin
    projection = relative @ direction
    valid = (projection > 0.0) & (projection < length)
    if not np.any(valid):
        return float("inf"), None, False
    perpendicular = np.linalg.norm(relative[valid] - projection[valid, None] * direction, axis=1)
    clearance = float(np.min(perpendicular))
    blockers = perpendicular <= radius
    if not np.any(blockers):
        return clearance, None, False
    blocker_distance = float(np.min(projection[valid][blockers]))
    return clearance, length - blocker_distance, True


def candidate_visibility_features(
    observed_points_world: np.ndarray,
    human_anchors_world: np.ndarray,
    *,
    candidate_position: Sequence[float],
    candidate_rotation_wxyz: Sequence[float],
    radius: float = RAY_BLOCK_RADIUS_M,
) -> np.ndarray:
    """Compute the fixed 17-D observed-only visibility feature vector."""
    points = np.asarray(observed_points_world, dtype=np.float64).reshape(-1, 3)
    joints = np.asarray(human_anchors_world, dtype=np.float64).reshape(-1, 3)
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0:
        raise ValueError("human_anchors_world must be a non-empty [J,3] array")
    origin, camera, projected, inframe = _project_joints(joints, candidate_position, candidate_rotation_wxyz)
    clearances: list[float] = []
    gaps: list[float] = []
    blocked = np.zeros(len(joints), dtype=bool)
    for index, joint in enumerate(joints):
        clearance, gap, is_blocked = _ray_clearance_and_gap(origin, joint, points, radius)
        clearances.append(clearance if np.isfinite(clearance) else 10.0)
        if gap is not None:
            gaps.append(gap)
        blocked[index] = bool(is_blocked)
    # The anchor order is deterministic (head/upper body first in the
    # canonical extractor).  Splits are intentionally positional and used
    # only as a descriptive aggregate, not as a learned feature contract.
    half = max(1, len(joints) // 2)
    upper = np.arange(len(joints)) < half
    lower = ~upper
    left = camera[:, 0] < 0.0
    right = ~left
    visible = inframe & ~blocked
    u, v = projected[:, 0], projected[:, 1]
    inframe_u = np.clip(u, 0.0, IMAGE_SIZE - 1.0)
    inframe_v = np.clip(v, 0.0, IMAGE_SIZE - 1.0)
    bbox_width = float(np.ptp(inframe_u[inframe])) / IMAGE_SIZE if np.any(inframe) else 0.0
    bbox_height = float(np.ptp(inframe_v[inframe])) / IMAGE_SIZE if np.any(inframe) else 0.0
    area = bbox_width * bbox_height
    center_offset = float(np.linalg.norm(np.mean(projected[inframe], axis=0) / IMAGE_SIZE - 0.5)) if np.any(inframe) else 1.0
    target_distance = float(np.mean(np.linalg.norm(joints - origin, axis=1)))
    horizontal = float(np.mean(joints[:, 0] - origin[0]))
    elevation = float(np.mean(joints[:, 1] - origin[1]))
    azimuth = math.degrees(math.atan2(horizontal, -float(np.mean(joints[:, 2] - origin[2]))))

    def fraction(mask: np.ndarray, values: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    return np.asarray([
        float(np.mean(visible)), float(np.mean(inframe)), float(np.mean(blocked)),
        fraction(upper, visible), fraction(lower, visible), fraction(left, visible),
        fraction(right, visible), float(np.mean(clearances)), float(np.min(clearances)),
        float(np.mean(gaps)) if gaps else target_distance, target_distance, bbox_width,
        bbox_height, area, center_offset, azimuth / 180.0, math.degrees(math.atan2(elevation, target_distance)) / 90.0,
    ], dtype=np.float32)


def analytic_candidate_order(features: Mapping[int, Sequence[float]]) -> int:
    """Method A0 deterministic visibility-only winner (p2 wins final ties)."""
    if not features:
        raise ValueError("features cannot be empty")
    def key(item: tuple[int, Sequence[float]]) -> tuple[float, float, float, int]:
        values = np.asarray(item[1], dtype=np.float64)
        return (float(values[0]), float(values[1]), float(values[13]), -int(item[0]))
    return int(max(features.items(), key=key)[0])


def full_scene_visibility(
    simulator: Any,
    candidate_position: Sequence[float],
    candidate_rotation_wxyz: Sequence[float],
    human_anchors_world: np.ndarray,
) -> np.ndarray:
    """Return in-frame/visible flags using Habitat collision raycasts."""
    from habitat_sim.geo import Ray

    joints = np.asarray(human_anchors_world, dtype=np.float64)
    rotation = rotation_matrix(candidate_rotation_wxyz)
    origin = camera_origin(candidate_position, rotation)
    _origin, _camera, _projected, inframe = _project_joints(joints, candidate_position, candidate_rotation_wxyz)
    visible = np.zeros(len(joints), dtype=np.float32)
    for index, joint in enumerate(joints):
        vector = joint - origin
        distance = float(np.linalg.norm(vector))
        if not inframe[index] or distance <= 1e-9:
            continue
        result = simulator.cast_ray(Ray(origin, vector / distance), max_distance=distance)
        hits = getattr(result, "hits", ())
        first = float(hits[0].ray_distance) if hits else distance
        visible[index] = float(first >= distance - 0.12)
    return visible


def extract_human_anchors(human: Any, max_anchors: int = 17) -> np.ndarray:
    """Read deterministic body-link world positions from a posed humanoid."""
    positions: list[np.ndarray] = []
    for link_id in sorted(int(value) for value in human.get_link_ids()):
        try:
            node = human.get_link_scene_node(link_id)
            # Habitat's Python binding exposes ``absolute_transformation`` as
            # a method, while ``absolute_translation`` is a property.  Use
            # the latter when available and retain a small fallback for older
            # bindings.  Reading ``node.translation`` would silently produce
            # link-local coordinates and break the world-space ray audit.
            absolute_translation = getattr(node, "absolute_translation", None)
            if absolute_translation is not None:
                value = absolute_translation() if callable(absolute_translation) else absolute_translation
            else:
                transformation = node.absolute_transformation()
                value = transformation.translation
            positions.append(np.asarray(value, dtype=np.float64))
        except (AttributeError, RuntimeError, TypeError):
            continue
    if not positions:
        positions = [np.asarray(human.translation, dtype=np.float64)]
    values = np.asarray(positions, dtype=np.float64)
    if len(values) > max_anchors:
        indices = np.linspace(0, len(values) - 1, max_anchors, dtype=np.int64)
        values = values[indices]
    return values.astype(np.float32)
