"""Retarget ParaHome world-space body joints to the Habitat SMPL-X humanoid.

ParaHome's released SMPL-X fits cannot be copied directly to Habitat's
``male_0`` URDF: the two rigs use incompatible rest/joint frames.  This module
therefore retargets the released 23-joint body skeleton geometrically.  It
matches body-segment directions hierarchically, inheriting the parent frame so
the body twist is preserved, and never changes the generic AMASS/BABEL
``MotionConverter`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .joint_mapping import HABITAT_HUMANOID_QUAT_DIM, validate_motion_quaternions


PARAHOME_BODY_INDEX: Mapping[str, int] = {
    "pelvis": 0,
    "spine1": 1,
    "spine2": 2,
    "spine3": 4,
    "neck": 5,
    "head": 6,
    "right_collar": 7,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "left_collar": 11,
    "left_shoulder": 12,
    "left_elbow": 13,
    "left_wrist": 14,
    "right_hip": 15,
    "right_knee": 16,
    "right_ankle": 17,
    "right_foot": 18,
    "left_hip": 19,
    "left_knee": 20,
    "left_ankle": 21,
    "left_foot": 22,
}

PARAHOME_CHILDREN: Mapping[str, Tuple[str, ...]] = {
    "pelvis": ("left_hip", "right_hip", "spine1"),
    "left_hip": ("left_knee",),
    "left_knee": ("left_ankle",),
    "left_ankle": ("left_foot",),
    "right_hip": ("right_knee",),
    "right_knee": ("right_ankle",),
    "right_ankle": ("right_foot",),
    "spine1": ("spine2",),
    "spine2": ("spine3",),
    "spine3": ("neck", "left_collar", "right_collar"),
    "neck": ("head",),
    "left_collar": ("left_shoulder",),
    "left_shoulder": ("left_elbow",),
    "left_elbow": ("left_wrist",),
    "right_collar": ("right_shoulder",),
    "right_shoulder": ("right_elbow",),
    "right_elbow": ("right_wrist",),
}

# ParaHome world: X forward, Y left, Z up (right-handed, clipped SMPL-X
# convention).  Habitat: -Z forward, +Y up, +X right for a character whose
# rest facing is +Z (verified by rendering the male_0 rest pose).
#
# The mapping must be a *proper* rotation (det = +1).  An earlier version used
# [[0, 1, 0], [0, 0, 1], [-1, 0, 0]], which maps forward/up correctly but has
# det = -1: it mirrors the skeleton.  Because every solved joint rotation is a
# proper rotation, a mirrored target cannot be fitted, and for the joints with
# several children (pelvis, spine3) the least-squares fit flips the body frame
# by ~180 deg (chest facing . recorded facing = -0.91 ... -0.99) while the arm
# and hand *positions* stay faithful.  The rendered humanoid then reads as a
# face and torso turned against its own arms.
PARAHOME_TO_HABITAT = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)

# Joints that keep the rig's own rest curvature instead of following the
# recorded bone direction.  The ParaHome ``spine1``/``spine2`` joints are lumbar
# vertebrae (L5/L4) whose bones are only 0.05-0.07 m long, while the ``male_0``
# spine1/spine2 links are much higher up the spine (0.125/0.164 m) and carry a
# ~20 deg lordotic rest offset.  Forcing the rig's long lordotic lumbar onto the
# recorded short straight bones swings the abdomen mesh forward (the belly
# "bulges") and moves the hands away from the recorded objects.  The trunk's
# overall bend is still preserved because ``spine3`` is solved in world space
# from the neck and collar directions; measured on s50/s78/s3, keeping the rest
# curvature returns the abdomen to its rest shape and *reduces* the hand-to-cup /
# hand-to-laptop distance.
REST_SHAPE_JOINTS: Tuple[str, ...] = ("spine1", "spine2")

# Habitat-Sim exposes ``male_0`` articulated state in this link order.  It is
# not the PyBullet/URDF declaration order used by MotionConverterSMPLX (neck
# precedes the arms there).  Writing PyBullet-order quaternions into Habitat is
# the direct cause of the systematic tilted-head/raised-arm artifact.
HABITAT_MALE_0_JOINT_ORDER: Tuple[str, ...] = (
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "spine1", "spine2", "spine3",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "neck", "head", "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
)


@dataclass(frozen=True)
class RetargetDiagnostics:
    """Pose-fidelity diagnostics measured against source segment directions."""

    mean_joint_rmse_m: float
    mean_bone_angle_error_deg: Mapping[str, float]


def _align_vectors(rest_vectors: Sequence[np.ndarray], target_vectors: Sequence[np.ndarray]) -> np.ndarray:
    rest = np.stack(rest_vectors).astype(np.float64)
    target = np.stack(target_vectors).astype(np.float64)
    rest /= np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    rotation, _ = Rotation.align_vectors(target, rest)
    return rotation.as_matrix()


def _bone_angle_degrees(source: np.ndarray, target: np.ndarray) -> float:
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(source, target), -1.0, 1.0))))


class ParaHomeSkeletonRetargeter:
    """Convert ParaHome 23-joint positions into Habitat humanoid motion."""

    def __init__(self, urdf_path: Union[str, Path], *, match_hip_width: bool = True) -> None:
        self._match_hip_width = bool(match_hip_width)
        try:
            import pybullet as pybullet
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ParaHome retargeting requires pybullet in the Habitat environment") from exc

        resolved = Path(urdf_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Humanoid URDF not found: {resolved}")
        client = pybullet.connect(pybullet.DIRECT)
        try:
            humanoid = pybullet.loadURDF(str(resolved), useFixedBase=True)
            urdf_joint_names = []
            self._parent_names: Dict[str, str] = {}
            self._rest_offsets: Dict[str, np.ndarray] = {}
            for joint_id in range(pybullet.getNumJoints(humanoid)):
                info = pybullet.getJointInfo(humanoid, joint_id)
                name = info[1].decode("utf-8")
                parent_id = int(info[16])
                parent_name = "pelvis" if parent_id < 0 else urdf_joint_names[parent_id]
                urdf_joint_names.append(name)
                self._parent_names[name] = parent_name
                self._rest_offsets[name] = np.asarray(info[14], dtype=np.float64)
        finally:
            pybullet.disconnect(client)

        if set(urdf_joint_names) != set(HABITAT_MALE_0_JOINT_ORDER):
            raise ValueError("URDF joint set does not match the supported Habitat male_0 rig")
        self._joint_names = HABITAT_MALE_0_JOINT_ORDER
        if len(self._joint_names) * 4 != HABITAT_HUMANOID_QUAT_DIM:
            raise ValueError(f"Expected 54 articulated joints, found {len(self._joint_names)}")
        missing = set(PARAHOME_BODY_INDEX) - {"pelvis"} - set(self._joint_names)
        if missing:
            raise ValueError(f"URDF is missing required body joints: {sorted(missing)}")
        hierarchy: list[str] = []
        frontier = ["pelvis"]
        while frontier:
            name = frontier.pop(0)
            hierarchy.append(name)
            frontier.extend(child for child in PARAHOME_CHILDREN[name] if child in PARAHOME_CHILDREN)
        self._hierarchy_order: Tuple[str, ...] = tuple(hierarchy)
        # The rig's own rest stance: how far apart the knees are when the rig
        # stands untouched.  Used as the minimum stance width (see _frame_offsets).
        self._rest_knee_separation: float = float(
            np.linalg.norm(
                self._rest_offsets["left_hip"] + self._rest_offsets["left_knee"]
                - self._rest_offsets["right_hip"] - self._rest_offsets["right_knee"]
            )
        )

    def retarget(
        self,
        joint_positions: np.ndarray,
        *,
        fps: float = 30.0,
    ) -> Tuple[Dict[str, object], RetargetDiagnostics]:
        """Retarget a ``(T, >=23, 3)`` ParaHome world-space skeleton sequence."""
        source = np.asarray(joint_positions, dtype=np.float64)
        if source.ndim != 3 or source.shape[1] < 23 or source.shape[2] != 3:
            raise ValueError("joint_positions must have shape (T, >=23, 3)")
        if not np.isfinite(source).all():
            raise ValueError("joint_positions must be finite")
        if fps <= 0.0:
            raise ValueError("fps must be positive")

        joints_frames = []
        transform_frames = []
        frame_rmse = []
        angle_errors: Dict[str, list[float]] = {
            f"{parent}-{child}": []
            for parent, children in PARAHOME_CHILDREN.items()
            for child in children
        }
        for frame in source:
            desired = {
                name: PARAHOME_TO_HABITAT @ frame[index]
                for name, index in PARAHOME_BODY_INDEX.items()
            }
            global_rotations = self._solve_global_rotations(desired)
            root_rotation = global_rotations["pelvis"]
            offsets = self._frame_offsets(desired)
            joints_frames.append(self._local_quaternions(global_rotations, root_rotation))
            root_transform = np.eye(4, dtype=np.float32)
            root_transform[:3, :3] = root_rotation.astype(np.float32)
            root_transform[:3, 3] = desired["pelvis"].astype(np.float32)
            transform_frames.append(root_transform)

            reproduced = self._forward_body(desired["pelvis"], global_rotations, offsets)
            names = tuple(PARAHOME_BODY_INDEX)
            error = np.stack([reproduced[name] - desired[name] for name in names])
            frame_rmse.append(float(np.sqrt(np.mean(np.sum(error * error, axis=1)))))
            for parent, children in PARAHOME_CHILDREN.items():
                for child in children:
                    key = f"{parent}-{child}"
                    angle_errors[key].append(
                        _bone_angle_degrees(
                            desired[child] - desired[parent],
                            reproduced[child] - reproduced[parent],
                        )
                    )

        joints_array = np.stack(joints_frames).astype(np.float32)
        transform_array = np.stack(transform_frames).astype(np.float32)
        validate_motion_quaternions(joints_array)
        result: Dict[str, object] = {
            "pose_motion": {
                "joints_array": joints_array,
                "transform_array": transform_array,
                "displacement": None,
                "fps": float(fps),
            },
            "metadata": {
                "source": "ParaHome joint_positions.pkl",
                "retargeting": "hierarchical segment-direction matching",
                "num_frames": int(source.shape[0]),
                "fps": float(fps),
            },
        }
        diagnostics = RetargetDiagnostics(
            mean_joint_rmse_m=float(np.mean(frame_rmse)),
            mean_bone_angle_error_deg={
                name: float(np.mean(values)) for name, values in angle_errors.items()
            },
        )
        return result, diagnostics

    def _solve_global_rotations(self, desired: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Solve one world rotation per body joint, top-down along the rig.

        A segment direction never fixes the rotation *about* that segment, so a
        joint whose bone is (nearly) parallel to its rest direction is only
        determined up to a twist.  Solving every joint independently against the
        world rest frame resolves that freedom with a minimal world-space
        rotation, which silently discards the twist the parent chain carries:
        ``spine1``/``spine2``/``neck`` then collapse towards the rest facing
        while the shoulder girdle follows the real body yaw.  The rendered
        humanoid consequently reads as a torso/head turned against its own arms.

        Instead, each joint inherits its parent's solved frame and only adds the
        minimal rotation required by its own segment(s), so the root twist
        propagates down the whole chain.  Joints with several children are still
        determined by a joint alignment of all of them; expressing that
        alignment in the parent frame leaves their solution unchanged because
        the least-squares residual is invariant under the parent rotation.

        ``REST_SHAPE_JOINTS`` keep the rig's own rest curvature; see
        ``_constraints`` for why the recorded lumbar bones must not drive them.
        """
        rotations: Dict[str, np.ndarray] = {}
        for name in self._hierarchy_order:
            parent_name = self._parent_names.get(name, "pelvis")
            parent_frame = np.eye(3, dtype=np.float64) if name == "pelvis" else rotations[parent_name]
            if name in REST_SHAPE_JOINTS:
                rotations[name] = parent_frame
                continue
            rest, target = self._constraints(name, desired)
            target = [parent_frame.T @ vector for vector in target]
            rotations[name] = parent_frame @ _align_vectors(rest, target)
        return rotations

    def _constraints(self, name: str, desired: Mapping[str, np.ndarray]) -> Tuple[list, list]:
        """Rest/target direction pairs that determine one joint's world rotation.

        ``pelvis`` is the only joint whose URDF children are not recorded bones.
        The ParaHome ``hip`` joint is the hip *centre*, so its two hips are
        antiparallel (measured 178.3 deg apart), while the ``male_0`` pelvis
        link sits ~10 cm *above* the hip joints, so its hip offsets point
        down-lateral (60.5 deg apart).  Fitting the two raw hip offsets lets
        that definition mismatch dominate the pelvis rotation - the fit residual
        reaches ~60 deg, the pelvis tilts, and the whole trunk (and the abdomen
        mesh with it) leans into the wrong shape.  The derivation-invariant hip
        line ``left_hip - right_hip`` means the same lateral axis in both rigs,
        so the pelvis is instead solved from the hip line plus the spine axis.

        ``spine1``/``spine2`` never reach this method: see
        ``REST_SHAPE_JOINTS``.
        """
        if name == "pelvis":
            rest = [
                self._rest_offsets["left_hip"] - self._rest_offsets["right_hip"],
                self._rest_offsets["spine1"],
            ]
            target = [
                desired["left_hip"] - desired["right_hip"],
                desired["spine1"] - desired["pelvis"],
            ]
            return rest, target
        children = PARAHOME_CHILDREN[name]
        return (
            [self._rest_offsets[child] for child in children],
            [desired[child] - desired[name] for child in children],
        )

    def _local_quaternions(
        self,
        global_rotations: Mapping[str, np.ndarray],
        root_rotation: np.ndarray,
    ) -> np.ndarray:
        quaternions = []
        identity = np.eye(3, dtype=np.float64)
        for name in self._joint_names:
            current = global_rotations.get(name)
            if current is None:
                local = identity
            else:
                parent_name = self._parent_names[name]
                parent = root_rotation if parent_name == "pelvis" else global_rotations[parent_name]
                local = parent.T @ current
            quaternions.extend(Rotation.from_matrix(local).as_quat().tolist())
        return np.asarray(quaternions, dtype=np.float64)

    def _frame_offsets(self, desired: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Rig offsets for one frame, optionally matching the recorded hip width.

        The ``male_0`` hip joints are 12.5 cm apart while ParaHome's hips (hip
        *centre* convention) are 18.8-21.9 cm apart depending on the subject.
        Joint directions are matched but not joint spacing, so the avatar's legs
        stayed ~8 cm narrower than the recording and read as pressed together.
        Scaling the lateral component of the two hip offsets by the recorded/rig
        separation ratio reproduces the recorded stance; the legs simply
        translate outward, so the mesh is only stretched in the hip crease.

        The recorded stance is also narrower than the rig's own rest stance
        (knees 19-20 cm apart versus 24.4 cm at rest), which makes the thick
        ``male_0`` legs interpenetrate (measured knee mesh gap +1.1 cm and
        -5.9 cm on s3).  The target separation therefore never drops below the
        rig's rest knee separation, so the avatar is never asked to stand in a
        pose its own body cannot hold.
        """
        if not self._match_hip_width:
            return self._rest_offsets
        separation = np.linalg.norm(self._rest_offsets["left_hip"] - self._rest_offsets["right_hip"])
        recorded = np.linalg.norm(desired["left_hip"] - desired["right_hip"])
        # The recorded separation is a per-subject constant (18.8-21.9 cm on the
        # audited sequences); clamp so a bad fit cannot tear the mesh.
        target = max(float(recorded), self._rest_knee_separation)
        scale = float(np.clip(target / max(separation, 1e-9), 0.8, 2.0))
        offsets = dict(self._rest_offsets)
        axis = self._rest_offsets["left_hip"] - self._rest_offsets["right_hip"]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        for name in ("left_hip", "right_hip"):
            offset = self._rest_offsets[name]
            offsets[name] = offset + (scale - 1.0) * float(offset @ axis) * axis
        return offsets

    def _forward_body(
        self,
        root_position: np.ndarray,
        global_rotations: Mapping[str, np.ndarray],
        offsets: Mapping[str, np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        table = self._rest_offsets if offsets is None else offsets
        positions = {"pelvis": np.asarray(root_position, dtype=np.float64)}
        unresolved = set(PARAHOME_BODY_INDEX) - {"pelvis"}
        while unresolved:
            progressed = False
            for name in tuple(unresolved):
                parent = self._parent_names[name]
                if parent not in positions:
                    continue
                positions[name] = positions[parent] + global_rotations[parent] @ table[name]
                unresolved.remove(name)
                progressed = True
            if not progressed:
                raise RuntimeError(f"Cannot resolve retarget hierarchy: {sorted(unresolved)}")
        return positions
