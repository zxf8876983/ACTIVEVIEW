"""Offline before/after render of the retargeted Habitat humanoid (CPU only).

Habitat cannot run in this session (no CUDA/EGL device), so this script applies
the same linear-blend skinning that Habitat performs: the released ``male_0.glb``
skin rest world matrices are driven by the URDF-relative world rotations solved
by ``ParaHomeSkeletonRetargeter``.  The GLB rest skeleton matches the URDF rest
skeleton up to one constant root offset (verified separately), so the deformed
mesh is directly comparable with a Habitat render.

Run with the Habitat Conda interpreter.
"""

from __future__ import annotations

import json
import pickle
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

def _repo_root() -> Path:
    candidate = Path(__file__).resolve()
    while candidate != candidate.parent and not (candidate / "activeview" / "core" / "paths.py").is_file():
        candidate = candidate.parent
    return candidate


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_humanoid_asset_root, get_humanoid_urdf_path  # noqa: E402
from activeview.data.motion.parahome_retarget import (  # noqa: E402
    PARAHOME_BODY_INDEX,
    PARAHOME_CHILDREN,
    PARAHOME_TO_HABITAT,
    ParaHomeSkeletonRetargeter,
    _align_vectors,
)

PARAHOME_ROOT = Path("/home/zxf/MG08/robot/ParaHome")
SEQUENCE = "s78"
GLB = get_humanoid_asset_root("male_0") / "male_0.glb"
OUTPUT = REPO_ROOT / "experiments" / "parahome_feasibility_v1" / "retarget_pose_fidelity" / "renders"
FRAME_FRACTIONS = (0.25, 0.5)
MAX_FACES = 6000
COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SHAPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class AbsoluteRetargeter(ParaHomeSkeletonRetargeter):
    """Pre-fix solver: one independent world-space alignment per joint."""

    def _solve_global_rotations(self, desired):
        return {
            parent: _align_vectors(
                [self._rest_offsets[child] for child in children],
                [desired[child] - desired[parent] for child in children],
            )
            for parent, children in PARAHOME_CHILDREN.items()
        }


class SkinnedHumanoid:
    """Linear-blend skinning of male_0.glb using URDF-relative joint frames."""

    def __init__(self, glb_path: Path, retargeter: ParaHomeSkeletonRetargeter) -> None:
        self.gltf, self.binary = _load_glb(glb_path)
        primitive = self.gltf["meshes"][0]["primitives"][0]
        self.skin = self.gltf["skins"][0]
        self.positions = _read_accessor(self.gltf, self.binary, primitive["attributes"]["POSITION"])
        self.joints = _read_accessor(self.gltf, self.binary, primitive["attributes"]["JOINTS_0"]).astype(np.int64)
        self.weights = _read_accessor(self.gltf, self.binary, primitive["attributes"]["WEIGHTS_0"])
        self.weights /= np.maximum(self.weights.sum(axis=1, keepdims=True), 1e-9)
        self.faces = _read_accessor(self.gltf, self.binary, primitive["indices"]).astype(np.int64).reshape(-1, 3)
        self.bind_inverse = _read_accessor(self.gltf, self.binary, self.skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.rest_worlds = _glb_rest_worlds(self.gltf)
        self.joint_names = [self.gltf["nodes"][joint].get("name") for joint in self.skin["joints"]]
        self.offset = self._root_offset(retargeter)
        self.homogeneous = np.concatenate([self.positions, np.ones((len(self.positions), 1))], axis=1)

    def _root_offset(self, retargeter: ParaHomeSkeletonRetargeter) -> np.ndarray:
        rest = retargeter._forward_body(np.zeros(3), {name: np.eye(3) for name in PARAHOME_BODY_INDEX})
        deltas = [
            self.rest_worlds[joint][:3, 3] - rest[name]
            for joint, name in zip(self.skin["joints"], self.joint_names)
            if name in rest
        ]
        spread = float(np.max(np.linalg.norm(np.asarray(deltas) - np.mean(deltas, axis=0), axis=1)))
        if spread > 1e-3:
            raise RuntimeError(f"GLB and URDF rest skeletons diverge by {spread:.4f} m")
        return np.mean(deltas, axis=0)

    def deform(self, positions: Dict[str, np.ndarray], rotations: Dict[str, np.ndarray]) -> np.ndarray:
        matrices = []
        for joint, name in zip(self.skin["joints"], self.joint_names):
            matrix = np.eye(4)
            if name in rotations and name in positions:
                matrix[:3, :3] = rotations[name] @ self.rest_worlds[joint][:3, :3]
                matrix[:3, 3] = positions[name] + self.offset
            else:
                # GLB-only helper joints (e.g. ``root``) carry no URDF motion.
                matrix = self.rest_worlds[joint]
            matrices.append(matrix)
        matrices = np.stack(matrices)
        skinned = matrices[self.joints] @ self.bind_inverse[self.joints]
        blend = np.einsum("vk,vkab->vab", self.weights, skinned)
        deformed = np.einsum("vab,vb->va", blend, self.homogeneous)[:, :3]
        return deformed - self.offset


def _load_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    offset = 12
    chunks: Dict[bytes, bytes] = {}
    while offset < len(data):
        length, kind = struct.unpack("<II", data[offset:offset + 8])
        chunks[kind.to_bytes(4, "little")] = data[offset + 8:offset + 8 + length]
        offset += 8 + length
    return json.loads(chunks[b"JSON"].decode("utf-8")), chunks[b"BIN\x00"]


def _read_accessor(gltf: Dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]])
    width = TYPE_SHAPES[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * accessor["count"], offset=start)
    return raw.reshape(accessor["count"], stride)[:, : dtype.itemsize * width].copy().view(dtype).reshape(accessor["count"], width).astype(np.float64)


def _node_local_matrix(node: Dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    if "translation" in node:
        matrix[:3, 3] = node["translation"]
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        matrix[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return matrix


def _glb_rest_worlds(gltf: Dict[str, Any]) -> Dict[int, np.ndarray]:
    parents: Dict[int, int] = {}
    for index, node in enumerate(gltf["nodes"]):
        for child in node.get("children", []):
            parents[child] = index
    worlds: Dict[int, np.ndarray] = {}

    def resolve(index: int) -> np.ndarray:
        if index not in worlds:
            local = _node_local_matrix(gltf["nodes"][index])
            worlds[index] = local if index not in parents else resolve(parents[index]) @ local
        return worlds[index]

    for index in range(len(gltf["nodes"])):
        resolve(index)
    return worlds


def _fk_all(
    retargeter: ParaHomeSkeletonRetargeter,
    joints: np.ndarray,
    root_rotation: np.ndarray,
    root_position: np.ndarray,
    offsets: Mapping[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """World position and rotation of every articulated joint, finger joints included."""
    quaternions = joints.reshape(-1, 4)
    locals_ = {
        name: Rotation.from_quat(quaternions[index]).as_matrix()
        for index, name in enumerate(retargeter._joint_names)
    }
    table = retargeter._rest_offsets if offsets is None else offsets
    rotations: Dict[str, np.ndarray] = {"pelvis": np.asarray(root_rotation, dtype=np.float64)}
    positions: Dict[str, np.ndarray] = {"pelvis": np.asarray(root_position, dtype=np.float64)}
    remaining = list(retargeter._joint_names)
    while remaining:
        progressed = False
        for name in list(remaining):
            parent = retargeter._parent_names[name]
            if parent not in rotations:
                continue
            rotations[name] = rotations[parent] @ locals_[name]
            positions[name] = positions[parent] + rotations[parent] @ table[name]
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError(f"unresolved joints {remaining}")
    return positions, rotations


def _facing_arrows(positions: Dict[str, np.ndarray], rotations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Body facing arrows in the rendered frame.

    A Habitat render of the rest pose shows ``male_0`` facing +Z, so
    ``R_joint @ (0, 0, 1)`` is that joint's current facing direction.
    """
    forward = np.asarray([0.0, 0.0, 1.0])
    colors = {"pelvis": "#c2007a", "spine1": "#e07b00", "spine3": "#00b3d9", "head": "#1240c9"}
    return {
        name: (positions[name], rotations[name] @ forward, colors[name])
        for name in colors
    }


def _to_plot(vector: np.ndarray) -> np.ndarray:
    """Habitat (Y up, -Z forward) -> matplotlib Z-up plotting frame."""
    return np.stack([vector[..., 0], -vector[..., 2], vector[..., 1]], axis=-1)


def _plot_frame(mesh: np.ndarray, faces: np.ndarray, center: np.ndarray, arrows: Dict[str, np.ndarray], azimuth: float, title: str, path: Path, view: str) -> None:
    plot = _to_plot(mesh)
    hub = _to_plot(np.asarray(center, dtype=np.float64))
    figure = plt.figure(figsize=(6.0, 6.0), dpi=110)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_trisurf(plot[:, 0], plot[:, 1], plot[:, 2], triangles=faces, color="#a9bad0", shade=True, linewidth=0, alpha=0.95)
    for name, (origin, direction, color) in arrows.items():
        start = _to_plot(np.asarray(origin, dtype=np.float64))
        vector = _to_plot(np.asarray(direction, dtype=np.float64)) * 0.40
        axis.quiver(start[0], start[1], start[2], vector[0], vector[1], vector[2], color=color, linewidth=3.0, arrow_length_ratio=0.35)
    span = 0.55
    axis.set_xlim(hub[0] - span, hub[0] + span)
    axis.set_ylim(hub[1] - span, hub[1] + span)
    axis.set_zlim(hub[2] - 0.62, hub[2] + 0.85)
    axis.set_box_aspect((2 * span, 2 * span, 1.47))
    axis.view_init(elev=78.0 if view == "top" else 12.0, azim=float(azimuth))
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    figure.savefig(path)
    plt.close(figure)


def render_variant(variant: str, retargeter: ParaHomeSkeletonRetargeter, humanoid: SkinnedHumanoid) -> Dict[str, Any]:
    sequence_path = PARAHOME_ROOT / "data" / "seq" / SEQUENCE
    with (sequence_path / "joint_positions.pkl").open("rb") as handle:
        source_joints = np.asarray(pickle.load(handle), dtype=np.float32)
    converted, diagnostics = retargeter.retarget(source_joints, fps=30.0)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    images: List[str] = []
    faces = humanoid.faces
    for fraction in FRAME_FRACTIONS:
        frame_id = int(round(fraction * (len(joints) - 1)))
        source_frame = source_joints[frame_id]
        desired = {
            name: PARAHOME_TO_HABITAT @ source_frame[index]
            for name, index in PARAHOME_BODY_INDEX.items()
        }
        positions, rotations = _fk_all(
            retargeter,
            joints[frame_id],
            roots[frame_id, :3, :3],
            roots[frame_id, :3, 3],
            offsets=retargeter._frame_offsets(desired),
        )
        mesh = humanoid.deform(positions, rotations)
        # Camera azimuth relative to the body facing so every frame is read the same way.
        facing = rotations["pelvis"] @ np.asarray([0.0, 0.0, 1.0])
        azimuth = np.degrees(np.arctan2(facing[2], facing[0])) + 35.0
        arrows = _facing_arrows(positions, rotations)
        for view in ("top", "iso"):
            path = OUTPUT / f"{variant}_frame_{frame_id:04d}_{view}.png"
            _plot_frame(mesh, faces, positions["pelvis"], arrows, azimuth, f"retarget {variant} | s78 frame {frame_id} | {view}", path, view)
            images.append(str(path.relative_to(OUTPUT)))
    return {"variant": variant, "retarget_mean_joint_rmse_m": diagnostics.mean_joint_rmse_m, "images": images}


def main() -> None:
    urdf = get_humanoid_urdf_path("male_0")
    humanoid = SkinnedHumanoid(GLB, ParaHomeSkeletonRetargeter(urdf))
    report = [
        render_variant("before", AbsoluteRetargeter(urdf), humanoid),
        render_variant("after", ParaHomeSkeletonRetargeter(urdf), humanoid),
    ]
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
