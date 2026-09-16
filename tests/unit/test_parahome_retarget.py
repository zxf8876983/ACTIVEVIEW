from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from activeview.core.paths import get_humanoid_urdf_path
from activeview.data.motion.parahome_retarget import (
    PARAHOME_BODY_INDEX,
    PARAHOME_CHILDREN,
    PARAHOME_TO_HABITAT,
    ParaHomeSkeletonRetargeter,
)


pybullet = pytest.importorskip("pybullet")


def _synthetic_parahome_skeleton(
    retargeter: ParaHomeSkeletonRetargeter,
    rotations: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build a ParaHome-frame skeleton whose Habitat FK is ``rotations``."""
    habitat_positions = {"pelvis": np.zeros(3, dtype=np.float64)}
    unresolved = set(PARAHOME_BODY_INDEX) - {"pelvis"}
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            parent = retargeter._parent_names[name]
            if parent not in habitat_positions:
                continue
            frame = np.eye(3) if rotations is None else rotations[parent]
            habitat_positions[name] = habitat_positions[parent] + frame @ retargeter._rest_offsets[name]
            unresolved.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError(f"Unresolved synthetic hierarchy: {sorted(unresolved)}")

    source = np.zeros((23, 3), dtype=np.float64)
    inverse_basis = PARAHOME_TO_HABITAT.T
    for name, index in PARAHOME_BODY_INDEX.items():
        source[index] = inverse_basis @ habitat_positions[name]
    return source


def _fk_world_rotations(
    retargeter: ParaHomeSkeletonRetargeter,
    joints: np.ndarray,
    root_rotation: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compose the retargeted local quaternions exactly like Habitat FK does."""
    quaternions = joints.reshape(-1, 4)
    rotations: Dict[str, np.ndarray] = {"pelvis": np.asarray(root_rotation, dtype=np.float64)}
    for index, name in enumerate(retargeter._joint_names):
        local = Rotation.from_quat(quaternions[index]).as_matrix()
        parent = retargeter._parent_names[name]
        base = root_rotation if parent == "pelvis" else rotations[parent]
        rotations[name] = base @ local
    return rotations


def _rotation_error_deg(expected: np.ndarray, actual: np.ndarray) -> float:
    cosine = (np.trace(expected.T @ actual) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def test_neutral_skeleton_retargets_without_pose_distortion() -> None:
    # Hip-width matching is a stance adjustment, not part of the rotation solve,
    # so it is disabled here to isolate the neutral-pose reproduction.
    retargeter = ParaHomeSkeletonRetargeter(get_humanoid_urdf_path("male_0"), match_hip_width=False)
    skeleton = _synthetic_parahome_skeleton(retargeter)

    motion, diagnostics = retargeter.retarget(skeleton[None], fps=30.0)

    joints = np.asarray(motion["pose_motion"]["joints_array"])
    transforms = np.asarray(motion["pose_motion"]["transform_array"])
    assert joints.shape == (1, 216)
    assert transforms.shape == (1, 4, 4)
    assert np.allclose(joints.reshape(-1, 4)[:, :3], 0.0, atol=1e-6)
    assert np.allclose(np.abs(joints.reshape(-1, 4)[:, 3]), 1.0, atol=1e-6)
    assert np.allclose(transforms[0], np.eye(4), atol=1e-6)
    assert diagnostics.mean_joint_rmse_m < 1e-7


def test_parahome_to_habitat_is_a_proper_rotation() -> None:
    """The coordinate mapping must not mirror the skeleton.

    A reflection (det = -1) cannot be fitted by proper joint rotations: for the
    joints with several children the least-squares fit flips the body frame by
    ~180 deg, which renders as a face and torso turned against the arms while
    the hand positions stay faithful.
    """
    assert np.isclose(np.linalg.det(PARAHOME_TO_HABITAT), 1.0, atol=1e-9)
    assert np.allclose(PARAHOME_TO_HABITAT @ PARAHOME_TO_HABITAT.T, np.eye(3), atol=1e-9)
    # ParaHome X forward, Y left, Z up -> Habitat -Z forward, +Y up, -X left.
    assert np.allclose(PARAHOME_TO_HABITAT @ np.asarray([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-9)
    assert np.allclose(PARAHOME_TO_HABITAT @ np.asarray([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(PARAHOME_TO_HABITAT @ np.asarray([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-9)


def test_hip_width_matching_reproduces_a_wider_recorded_stance() -> None:
    """The recorded stance is reproduced, and never narrower than the rig's own.

    ``male_0``'s hip joints are 12.5 cm apart while the recorded person's are
    ~20 cm apart; without the correction the whole leg chain stays ~8 cm too
    narrow and the thick male_0 legs interpenetrate.
    """
    urdf = get_humanoid_urdf_path("male_0")
    matched = ParaHomeSkeletonRetargeter(urdf, match_hip_width=True)
    unmatched = ParaHomeSkeletonRetargeter(urdf, match_hip_width=False)

    skeleton = _synthetic_parahome_skeleton(matched)
    axis = PARAHOME_TO_HABITAT.T @ (matched._rest_offsets["left_hip"] - matched._rest_offsets["right_hip"])
    axis /= np.linalg.norm(axis)
    for offset in (1.6, 1.0):  # a 20 cm recorded stance and the rig's own 12.5 cm
        widened = skeleton.copy()
        for name, sign in (("left_hip", 1.0), ("right_hip", -1.0)):
            index = PARAHOME_BODY_INDEX[name]
            widened[index] = widened[index] + sign * (offset - 1.0) * 0.0625 * axis

        for retargeter, expected in ((matched, max(0.20, matched._rest_knee_separation)), (unmatched, 0.125)):
            motion, _ = retargeter.retarget(widened[None], fps=30.0)
            positions = retargeter._forward_body(
                np.asarray(motion["pose_motion"]["transform_array"])[0, :3, 3],
                matched._solve_global_rotations(
                    {name: PARAHOME_TO_HABITAT @ widened[index] for name, index in PARAHOME_BODY_INDEX.items()}
                ),
                retargeter._frame_offsets(
                    {name: PARAHOME_TO_HABITAT @ widened[index] for name, index in PARAHOME_BODY_INDEX.items()}
                ),
            )
            separation = float(np.linalg.norm(positions["left_hip"] - positions["right_hip"]))
            assert abs(separation - expected) < 0.005, (separation, expected)
    assert matched._rest_knee_separation > 0.20


def test_retarget_rejects_non_finite_joint_positions() -> None:
    retargeter = ParaHomeSkeletonRetargeter(get_humanoid_urdf_path("male_0"))
    skeleton = _synthetic_parahome_skeleton(retargeter)[None]
    skeleton[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        retargeter.retarget(skeleton)


def test_retarget_inherits_root_twist_along_the_spine_chain() -> None:
    """A global turn must rotate the torso with the root, not towards the rest frame.

    A bone direction leaves the rotation about that bone free.  Solving each
    joint independently in world space used to resolve that freedom with a
    minimal rotation away from the rest facing, so ``spine1``/``spine2``/``neck``
    collapsed towards the rest frame while the shoulder girdle and arms
    followed the real body yaw: the rendered humanoid read as a head and torso
    turned against its own hands.  Inheriting the parent frame keeps them
    together.
    """
    retargeter = ParaHomeSkeletonRetargeter(get_humanoid_urdf_path("male_0"))
    yaw = Rotation.from_euler("y", 180.0).as_matrix()
    ground_truth = {name: yaw for name in PARAHOME_CHILDREN}
    skeleton = _synthetic_parahome_skeleton(retargeter, ground_truth)

    motion, _ = retargeter.retarget(skeleton[None], fps=30.0)
    joints = np.asarray(motion["pose_motion"]["joints_array"])[0]
    root_rotation = np.asarray(motion["pose_motion"]["transform_array"])[0, :3, :3]
    reproduced = _fk_world_rotations(retargeter, joints, root_rotation)

    for name in PARAHOME_CHILDREN:
        error = _rotation_error_deg(ground_truth[name], reproduced[name])
        if name == "neck":
            # The head bone is tilted ~12 deg off the world up axis, so the true
            # twist about it is not observable from joint positions; the swing
            # solution is the best a positions-only retarget can do.
            assert error < 15.0, f"{name} twist error {error:.1f} deg"
        else:
            assert error < 1e-4, f"{name} orientation error {error:.3f} deg"
