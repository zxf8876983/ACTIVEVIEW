#!/usr/bin/env python3
"""Render a ParaHome sequence without SMPL-X-to-Habitat retargeting.

ParaHome's ``joint_positions.pkl`` and object transforms share the capture
world frame (Z-up). This diagnostic renderer keeps those source coordinates
and draws the annotated body skeleton together with the recorded objects. It
is intentionally separate from the Habitat humanoid renderer: it is used to
decide whether an apparent raised arm is present in the source motion or is
introduced by the Habitat retargeting path.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


FPS = 30.0
BODY_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 15), (0, 19), (1, 2), (2, 4), (4, 5), (4, 7),
    (4, 11), (5, 6), (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
    (13, 14), (15, 16), (16, 17), (17, 18), (19, 20), (20, 21),
    (21, 22),
)
BODY_NAMES: Tuple[str, ...] = (
    "hip", "spine_l5", "spine_l4", "spine_l1", "thorax", "neck",
    "head", "right_t4_shoulder", "right_shoulder", "right_elbow",
    "right_wrist", "left_t4_shoulder", "left_shoulder", "left_elbow",
    "left_wrist", "right_hip", "right_knee", "right_ankle",
    "right_ball_foot", "left_hip", "left_knee", "left_ankle",
    "left_ball_foot",
)


@dataclass(frozen=True)
class MeshTemplate:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: Tuple[float, float, float]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_mesh(path: Path, name: str, color: Tuple[float, float, float]) -> MeshTemplate:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not contain a triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    # A few scanned objects are dense.  Deterministic face thinning keeps the
    # video responsive while retaining the object silhouette.
    max_faces = 2400
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)]
    return MeshTemplate(name, vertices, faces, color)


def _transform_vertices(template: MeshTemplate, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [template.vertices, np.ones((len(template.vertices), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3]


def _object_color(name: str) -> Tuple[float, float, float]:
    palette = {
        "laptop": (0.18, 0.34, 0.50),
        "chair": (0.46, 0.25, 0.14),
        "diningtable": (0.52, 0.34, 0.18),
        "desk": (0.52, 0.34, 0.18),
        "cup": (0.82, 0.82, 0.86),
        "kettle": (0.25, 0.27, 0.30),
    }
    return palette.get(name, (0.52, 0.52, 0.55))


def _select_objects(
    sequence_root: Path,
    object_transforms: Sequence[Mapping[str, Any]],
    names: Iterable[str],
) -> List[MeshTemplate]:
    first_frame = object_transforms[0]
    selected: List[MeshTemplate] = []
    for name in names:
        base_key = f"{name}_base"
        mesh_path = sequence_root.parent.parent / "scan" / name / "simplified" / "base.obj"
        if base_key not in first_frame or not mesh_path.is_file():
            continue
        try:
            selected.append(_load_mesh(mesh_path, name, _object_color(name)))
        except (OSError, ValueError, ImportError):
            continue
    return selected


def _annotation_text(sequence_root: Path) -> Dict[int, str]:
    path = sequence_root / "text_annotation.json"
    if not path.is_file():
        path = sequence_root / "text_annotations.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    values: Dict[int, str] = {}
    for interval, text in raw.items():
        parts = str(interval).split()
        if len(parts) != 2:
            continue
        start, end = int(parts[0]), int(parts[1])
        for frame_id in range(start, end):
            values[frame_id] = str(text)
    return values


def _frame_title(frame_id: int, text_by_frame: Mapping[int, str]) -> str:
    action = text_by_frame.get(frame_id, "unannotated")
    return f"ParaHome source-faithful replay | frame {frame_id} | {action}"


def _draw_frame(
    axis: Any,
    joints: np.ndarray,
    object_transforms: Mapping[str, Any],
    meshes: Sequence[MeshTemplate],
    title: str,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> None:
    axis.clear()
    for parent, child in BODY_EDGES:
        segment = joints[[parent, child]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#d62728", linewidth=3.0)
    axis.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="#ffcc33", s=18, depthshade=True)
    # Highlight the two shoulder-to-wrist chains so an arm pose can be checked
    # visually without relying on a retargeted skinned mesh.
    for chain in ((12, 13, 14), (8, 9, 10)):
        segment = joints[list(chain)]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#00d5ff", linewidth=5.0)
    for mesh in meshes:
        key = f"{mesh.name}_base"
        if key not in object_transforms:
            continue
        transform = np.asarray(object_transforms[key], dtype=np.float32)
        vertices = _transform_vertices(mesh, transform)
        triangles = vertices[mesh.faces][:, :, [0, 2, 1]]
        axis.add_collection3d(
            Poly3DCollection(triangles, facecolor=mesh.color, edgecolor="none", alpha=0.82)
        )
    minimum, maximum = bounds
    axis.set_xlim(float(minimum[0]), float(maximum[0]))
    axis.set_ylim(float(minimum[1]), float(maximum[1]))
    axis.set_zlim(float(minimum[2]), float(maximum[2]))
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z (up)")
    axis.view_init(elev=16.0, azim=-122.0)
    axis.set_title(title, fontsize=9)
    axis.grid(False)


def render_video(
    parahome_root: Path,
    sequence_id: str,
    output_path: Path,
    frame_start: int,
    frame_end: int,
    stride: int,
    fps: float,
) -> Dict[str, Any]:
    sequence_root = parahome_root / "data" / "seq" / sequence_id
    joints_all = np.asarray(_load_pickle(sequence_root / "joint_positions.pkl"), dtype=np.float32)
    object_transforms = _load_pickle(sequence_root / "object_transformations.pkl")
    if joints_all.ndim != 3 or joints_all.shape[1:] != (73, 3):
        raise ValueError(f"joint_positions.pkl must have shape (T,73,3), got {joints_all.shape}")
    if len(object_transforms) != len(joints_all):
        raise ValueError("object_transformations and joint_positions frame counts differ")
    start = max(0, int(frame_start))
    end = min(len(joints_all), int(frame_end))
    frame_ids = list(range(start, end, max(1, int(stride))))
    if not frame_ids:
        raise ValueError("empty frame range")
    present = json.loads((sequence_root / "object_in_scene.json").read_text(encoding="utf-8"))
    annotated = _annotation_text(sequence_root)
    preferred = [
        name for name in ("desk", "chair", "laptop", "cup", "kettle")
        if bool(present.get(name, False))
    ]
    meshes = _select_objects(sequence_root, object_transforms, preferred)
    used_points = [joints_all[frame_ids].reshape(-1, 3)]
    for mesh in meshes:
        for frame_id in frame_ids[:: max(1, len(frame_ids) // 12)]:
            key = f"{mesh.name}_base"
            if key in object_transforms[frame_id]:
                used_points.append(_transform_vertices(mesh, np.asarray(object_transforms[frame_id][key])))
    points = np.concatenate(used_points, axis=0).reshape(-1, 3)
    # Keep the visual body and the interaction area large in frame.  ParaHome
    # stores a complete room coordinate frame; showing that entire extent
    # makes the person unreadably small, so we crop only the local interaction
    # neighborhood around the observed body.
    body_center = joints_all[frame_ids, :23].reshape(-1, 3).mean(axis=0)
    minimum = body_center - np.asarray([1.15, 0.85, 1.05], dtype=np.float32)
    maximum = body_center + np.asarray([1.15, 0.85, 1.05], dtype=np.float32)
    # Include the selected furniture only when it intersects this local crop.
    # This retains the laptop/chair context without letting a distant scan
    # vertex determine the camera scale.
    minimum = np.minimum(minimum, np.maximum(points.min(axis=0), body_center - 1.25))
    maximum = np.maximum(maximum, np.minimum(points.max(axis=0), body_center + 1.25))
    bounds = (minimum, maximum)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(6.4, 6.4), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    metadata = {"sequence_id": sequence_id, "frame_ids": frame_ids, "objects": [m.name for m in meshes]}
    with imageio.get_writer(str(output_path), fps=float(fps), codec="libx264", quality=8) as writer:
        for frame_id in frame_ids:
            _draw_frame(axis, joints_all[frame_id, :23], object_transforms[frame_id], meshes, _frame_title(frame_id, annotated), bounds)
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(image)
    plt.close(figure)
    metadata.update({"frame_count": len(frame_ids), "fps": float(fps), "coordinate_frame": "ParaHome capture world (Z-up)", "renderer": "raw joint_positions + recorded object transforms; no Habitat SMPL-X retargeting"})
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parahome-root", type=Path, required=True)
    parser.add_argument("--sequence", default="s78")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=445)
    parser.add_argument("--frame-end", type=int, default=690)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    metadata = render_video(args.parahome_root.resolve(), args.sequence, args.output.resolve(), args.frame_start, args.frame_end, args.stride, args.fps)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
