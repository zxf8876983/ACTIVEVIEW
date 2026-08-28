"""Geometry-aware grounding for the Habitat humanoid URDF."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class VisualBox:
    """One URDF visual box expressed in its link-local frame."""

    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    size: Tuple[float, float, float]


def _parse_vector(value: str, *, length: int = 3) -> Tuple[float, ...]:
    values = tuple(float(item) for item in value.split())
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {value!r}")
    return values


@lru_cache(maxsize=4)
def _load_visual_boxes(urdf_path: str) -> Mapping[str, Tuple[VisualBox, ...]]:
    root = ET.parse(urdf_path).getroot()
    boxes: Dict[str, Tuple[VisualBox, ...]] = {}
    for link in root.findall("link"):
        link_name = str(link.attrib["name"])
        link_boxes = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            box = geometry.find("box") if geometry is not None else None
            if box is None:
                raise ValueError(
                    f"Unsupported non-box visual in {urdf_path}: link={link_name}"
                )
            origin = visual.find("origin")
            link_boxes.append(
                VisualBox(
                    xyz=_parse_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0),
                    rpy=_parse_vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0),
                    size=_parse_vector(box.attrib["size"]),
                )
            )
        boxes[link_name] = tuple(link_boxes)
    return boxes


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float32,
    )


def _box_corners(box: VisualBox) -> np.ndarray:
    size = np.asarray(box.size, dtype=np.float32)
    corners = np.asarray(
        [
            [sx * size[0] / 2.0, sy * size[1] / 2.0, sz * size[2] / 2.0]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )
    return corners @ _rpy_matrix(box.rpy).T + np.asarray(box.xyz, dtype=np.float32)


def _world_points(local_points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [local_points, np.ones((len(local_points), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3]


def _object_transform(human: Any) -> np.ndarray:
    rotation = np.asarray(human.transformation, dtype=np.float32)
    if rotation.shape != (4, 4):
        raise ValueError(f"Unexpected humanoid transformation shape: {rotation.shape}")
    transform = rotation.copy()
    transform[:3, 3] = np.asarray(human.translation, dtype=np.float32)
    return transform


def humanoid_geometry_y_bounds(human: Any, urdf_path: Path) -> Tuple[float, float]:
    """Return the actual rendered URDF geometry's world-space y bounds.

    Habitat-Sim 0.3.3 exposes link origins but not the articulated visual
    bounding boxes through ``ArticulatedObject``.  The active humanoid URDF is
    composed exclusively of box visuals, so we transform each URDF box corner
    with the current link scene-node transform.  This is more faithful than
    using the lowest joint/link origin, especially for lying and falling poses.
    """

    boxes_by_link = _load_visual_boxes(str(urdf_path.resolve()))
    world_points = []

    # The pelvis is the articulated object's root and is not returned by
    # get_link_scene_node() in Habitat-Sim 0.3.3.
    root_boxes = boxes_by_link.get("pelvis", ())
    root_transform = _object_transform(human)
    for box in root_boxes:
        world_points.append(_world_points(_box_corners(box), root_transform))

    for index in range(int(human.num_links)):
        link_name = str(human.get_link_name(index))
        node = human.get_link_scene_node(index)
        link_transform = np.asarray(node.absolute_transformation(), dtype=np.float32)
        for box in boxes_by_link.get(link_name, ()):
            world_points.append(_world_points(_box_corners(box), link_transform))

    if not world_points:
        link_y = [
            float(human.get_link_scene_node(index).absolute_translation[1])
            for index in range(int(human.num_links))
        ]
        return min(link_y), max(link_y)

    points = np.concatenate(world_points, axis=0)
    return float(np.min(points[:, 1])), float(np.max(points[:, 1]))


def ground_humanoid_to_floor(
    human: Any,
    *,
    base_position: Sequence[float],
    floor_y: float,
    urdf_path: Path,
) -> float:
    """Translate an already posed humanoid so its rendered geometry touches ``floor_y``."""

    base = np.asarray(base_position, dtype=np.float32)
    if base.shape != (3,):
        raise ValueError("base_position must have shape (3,)")
    human.translation = np.asarray([base[0], 0.0, base[2]], dtype=np.float32)
    min_y, _ = humanoid_geometry_y_bounds(human, urdf_path)
    grounded_y = float(floor_y) - min_y
    human.translation = np.asarray([base[0], grounded_y, base[2]], dtype=np.float32)
    return grounded_y


def select_floor_height(
    ray_hits: Sequence[Any],
    *,
    reference_y: float,
    max_navmesh_gap: float = 0.25,
) -> float:
    """Select the local floor from a vertical ray with multiple intersections.

    HM3D rays can cross ceilings, furniture, intermediate floors, and the
    actual floor. ``hits[0]`` is therefore not guaranteed to be the ground.
    Given a navigable-point height, choose the highest hit at or just below the
    point (the closest supporting surface), avoiding overhead geometry.
    """

    heights = [float(hit.point[1]) for hit in ray_hits]
    if not heights:
        return float(reference_y)
    below = [height for height in heights if height <= float(reference_y) + float(max_navmesh_gap)]
    return max(below) if below else min(heights)
