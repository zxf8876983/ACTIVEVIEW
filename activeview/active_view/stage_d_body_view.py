"""Deterministic human-centric candidate viewpoint geometry for EXP031."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from activeview.active_view.stage_d_visibility import IMAGE_SIZE, HFOV_DEG, rotation_matrix


FEATURE_NAMES = (
    "candidate_human_distance", "candidate_human_azimuth", "candidate_human_elevation",
    "camera_forward_root_cosine", "shoulder_plane_view_angle", "hip_plane_view_angle",
    "body_facing_cosine", "front_view", "back_view", "left_side_view", "right_side_view",
    "projected_bbox_width", "projected_bbox_height", "projected_bbox_area", "projected_aspect_ratio",
    "mean_projected_bone_length", "min_projected_bone_length", "foreshortening_mean",
    "pairwise_projected_overlap", "left_right_limb_overlap", "upper_lower_overlap",
    "joint_inframe_fraction", "image_center_offset",
)


def _angle_to_plane(anchors: np.ndarray, direction: np.ndarray) -> float:
    if len(anchors) < 3:
        return 0.0
    centered = anchors - anchors.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return float(abs(np.dot(normal, direction)))


def body_view_features(
    anchors_world: np.ndarray,
    root_world: Sequence[float],
    candidate_position: Sequence[float],
    candidate_rotation_wxyz: Sequence[float],
) -> np.ndarray:
    """Return fixed human-centric geometry features (no environment pixels)."""
    anchors = np.asarray(anchors_world, dtype=np.float64).reshape(-1, 3)
    root = np.asarray(root_world, dtype=np.float64)
    candidate = np.asarray(candidate_position, dtype=np.float64)
    if anchors.shape[0] < 4 or root.shape != (3,) or candidate.shape != (3,):
        raise ValueError("invalid human/candidate geometry")
    rotation = rotation_matrix(candidate_rotation_wxyz)
    origin = candidate + rotation @ np.array([0.0, 1.1, 0.0])
    vector = anchors.mean(axis=0) - origin
    distance = float(np.linalg.norm(vector)); direction = vector / max(distance, 1e-9)
    camera = (rotation.T @ (anchors - origin).T).T
    depth = -camera[:, 2]; focal = IMAGE_SIZE / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
    safe = np.maximum(depth, 1e-6)
    uv = np.column_stack((focal * camera[:, 0] / safe + IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0 - focal * camera[:, 1] / safe))
    inframe = (depth > 0.0) & (uv[:, 0] >= 0.0) & (uv[:, 0] < IMAGE_SIZE) & (uv[:, 1] >= 0.0) & (uv[:, 1] < IMAGE_SIZE)
    projected = uv[inframe]
    width = float(np.ptp(projected[:, 0]) / IMAGE_SIZE) if len(projected) > 1 else 0.0
    height = float(np.ptp(projected[:, 1]) / IMAGE_SIZE) if len(projected) > 1 else 0.0
    area = width * height
    aspect = width / max(height, 1e-6)
    bones = np.linalg.norm(np.diff(camera, axis=0), axis=1) if len(camera) > 1 else np.zeros(1)
    projected_bones = np.linalg.norm(np.diff(uv, axis=0), axis=1) / IMAGE_SIZE if len(uv) > 1 else np.zeros(1)
    body_forward = anchors[min(len(anchors) - 1, len(anchors) // 3)] - root
    body_forward[1] = 0.0; body_forward /= max(np.linalg.norm(body_forward), 1e-9)
    horizontal = direction.copy(); horizontal[1] = 0.0; horizontal /= max(np.linalg.norm(horizontal), 1e-9)
    facing = float(np.dot(body_forward, horizontal)); forward_camera = rotation[:, 2]
    shoulder = anchors[: max(3, len(anchors) // 3)]; hip = anchors[len(anchors) // 3 : 2 * len(anchors) // 3]
    overlap = float(np.mean(np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=-1)[np.triu_indices(len(uv), 1)] < 0.02 * IMAGE_SIZE)) if len(uv) > 1 else 0.0
    left_overlap = float(np.mean(projected_bones[: max(1, len(projected_bones) // 2)] < 0.02)) if len(projected_bones) else 0.0
    lower_overlap = float(np.mean(projected_bones[max(1, len(projected_bones) // 2) :] < 0.02)) if len(projected_bones) else 0.0
    values = np.asarray([
        distance, math.atan2(horizontal[0], -horizontal[2]) / math.pi, math.atan2(vector[1], max(distance, 1e-9)) / (math.pi / 2),
        float(np.dot(forward_camera, direction)), _angle_to_plane(shoulder, direction), _angle_to_plane(hip, direction), facing,
        max(facing, 0.0), max(-facing, 0.0), max(horizontal[0], 0.0), max(-horizontal[0], 0.0), width, height, area, aspect,
        float(np.mean(projected_bones)) if len(projected_bones) else 0.0, float(np.min(projected_bones)) if len(projected_bones) else 0.0,
        float(np.mean(np.abs(camera[:-1, 2]) / np.maximum(bones, 1e-6))) if len(bones) else 0.0, overlap, left_overlap, lower_overlap, float(np.mean(inframe)),
        float(np.linalg.norm(projected.mean(axis=0) / IMAGE_SIZE - 0.5)) if len(projected) else 1.0,
    ], dtype=np.float32)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("non-finite body-view feature")
    return values


def candidate_body_view_features(*args: object, **kwargs: object) -> np.ndarray:
    return body_view_features(*args, **kwargs)
