"""Before/after validation of the ParaHome -> Habitat retarget pose fidelity fix.

Checks (all with the pre-fix world-space solver vs the shipped hierarchical
solver):

1. synthetic ground truth: a known Habitat pose is converted to a ParaHome
   skeleton, retargeted, and the recovered joint orientations are compared with
   the exact ground truth;
2. real ParaHome sequences: the retargeted body facing is compared with the
   observed shoulder-line geometry, the head motion is compared with the
   observed head-tip motion, and the fitted segment-direction error is reported
   to show the fix changes only the free twist.

Run with the Habitat Conda interpreter.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

def _repo_root() -> Path:
    candidate = Path(__file__).resolve()
    while candidate != candidate.parent and not (candidate / "activeview" / "core" / "paths.py").is_file():
        candidate = candidate.parent
    return candidate


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_humanoid_urdf_path  # noqa: E402
from activeview.data.motion.parahome_retarget import (  # noqa: E402
    PARAHOME_BODY_INDEX,
    PARAHOME_CHILDREN,
    PARAHOME_TO_HABITAT,
    ParaHomeSkeletonRetargeter,
    _align_vectors,
)

PARAHOME_ROOT = Path("/home/zxf/MG08/robot/ParaHome")
SEQUENCES = ("s78", "s50")
CHIRALITY_SEQUENCES = ("s3", "s50", "s78", "s99", "s121", "s162")
# The pre-fix mapping: forward/up mapping identical, but det = -1 (a mirror).
MIRROR_MAPPING = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
FACING_JOINTS = ("pelvis", "spine1", "spine2", "spine3", "neck", "head")
# Bones whose endpoints are *different anatomical joints* in the two rigs, so a
# direction match is not a meaningful error: the ParaHome hip joint is the hip
# centre (male_0 hips hang ~10 cm below its pelvis link) and the ParaHome
# spine1/spine2 are lumbar vertebrae whose rest curvature the rig keeps.
DEFINITIONAL_MISMATCH = ("pelvis-left_hip", "pelvis-right_hip", "spine1-spine2", "spine2-spine3")
OUTPUT = REPO_ROOT / "experiments" / "parahome_feasibility_v1" / "retarget_pose_fidelity"


class PreFixRetargeter(ParaHomeSkeletonRetargeter):
    """Pre-fix solver: one independent world-space alignment per joint."""

    def _solve_global_rotations(self, desired):
        return {
            parent: _align_vectors(
                [self._rest_offsets[child] for child in children],
                [desired[child] - desired[parent] for child in children],
            )
            for parent, children in PARAHOME_CHILDREN.items()
        }


def world_rotations(retargeter: ParaHomeSkeletonRetargeter, joints: np.ndarray, root_rotation: np.ndarray) -> Dict[str, np.ndarray]:
    """Habitat FK composition of the retargeted local quaternions."""
    quaternions = joints.reshape(-1, 4)
    rotations: Dict[str, np.ndarray] = {"pelvis": np.asarray(root_rotation, dtype=np.float64)}
    for index, name in enumerate(retargeter._joint_names):
        local = Rotation.from_quat(quaternions[index]).as_matrix()
        parent = retargeter._parent_names[name]
        rotations[name] = (root_rotation if parent == "pelvis" else rotations[parent]) @ local
    return rotations


def world_positions(retargeter: ParaHomeSkeletonRetargeter, rotations: Dict[str, np.ndarray], root_position: np.ndarray) -> Dict[str, np.ndarray]:
    positions = {"pelvis": np.asarray(root_position, dtype=np.float64)}
    remaining = list(retargeter._joint_names)
    while remaining:
        progressed = False
        for name in list(remaining):
            parent = retargeter._parent_names[name]
            if parent not in rotations:
                continue
            positions[name] = positions[parent] + rotations[parent] @ retargeter._rest_offsets[name]
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError(f"unresolved joints {remaining}")
    return positions


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def synthetic_skeleton(retargeter: ParaHomeSkeletonRetargeter, rotations: Dict[str, np.ndarray]) -> np.ndarray:
    positions = {"pelvis": np.zeros(3, dtype=np.float64)}
    unresolved = set(PARAHOME_BODY_INDEX) - {"pelvis"}
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            parent = retargeter._parent_names[name]
            if parent not in positions:
                continue
            positions[name] = positions[parent] + rotations[parent] @ retargeter._rest_offsets[name]
            unresolved.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError(f"unresolved synthetic hierarchy {sorted(unresolved)}")
    source = np.zeros((23, 3), dtype=np.float64)
    for name, index in PARAHOME_BODY_INDEX.items():
        source[index] = PARAHOME_TO_HABITAT.T @ positions[name]
    return source


def synthetic_case(retargeter: ParaHomeSkeletonRetargeter) -> Dict[str, float]:
    yaw = Rotation.from_euler("y", 180.0).as_matrix()
    head_pitch = yaw @ Rotation.from_euler("x", 20.0).as_matrix()
    ground_truth = {name: yaw for name in PARAHOME_CHILDREN}
    ground_truth["neck"] = head_pitch
    ground_truth["head"] = head_pitch
    skeleton = synthetic_skeleton(retargeter, ground_truth)[None]
    desired = {name: PARAHOME_TO_HABITAT @ skeleton[0, index] for name, index in PARAHOME_BODY_INDEX.items()}
    solved = retargeter._solve_global_rotations(desired)

    def recovered(name: str) -> np.ndarray:
        while name not in solved:
            name = retargeter._parent_names[name]
        return solved[name]

    return {name: angle_deg(ground_truth[name], recovered(name)) for name in FACING_JOINTS}


def real_sequence_metrics(root: Path, sequence: str, retargeter: ParaHomeSkeletonRetargeter) -> Dict[str, Dict[str, float]]:
    sequence_path = root / "data" / "seq" / sequence
    with (sequence_path / "joint_positions.pkl").open("rb") as handle:
        positions = np.asarray(pickle.load(handle), dtype=np.float64)
    with (sequence_path / "head_tips.pkl").open("rb") as handle:
        tips = np.asarray(pickle.load(handle), dtype=np.float64)
    frame_ids = np.linspace(0, len(positions) - 1, 150, dtype=np.int64)
    converted, _ = retargeter.retarget(positions.astype(np.float32), fps=30.0)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)

    facing: Dict[str, List[float]] = {name: [] for name in ("spine1", "spine3", "head")}
    relative: List[float] = []
    tip_drift: List[np.ndarray] = []
    bone_error: List[float] = []
    bone_error_matched: List[float] = []
    for frame_id in frame_ids:
        mapped = positions[frame_id] @ PARAHOME_TO_HABITAT.T
        observed = np.cross(
            mapped[PARAHOME_BODY_INDEX["left_shoulder"]] - mapped[PARAHOME_BODY_INDEX["right_shoulder"]],
            mapped[PARAHOME_BODY_INDEX["neck"]] - mapped[PARAHOME_BODY_INDEX["pelvis"]],
        )
        observed /= np.linalg.norm(observed)
        rotations = world_rotations(retargeter, joints[frame_id], roots[frame_id, :3, :3])
        for name in facing:
            facing[name].append(vector_angle_deg(rotations[name] @ np.asarray([0.0, 0.0, 1.0]), observed))
        relative.append(angle_deg(rotations["spine3"], rotations["head"]))
        tip = PARAHOME_TO_HABITAT @ (tips[frame_id] - positions[frame_id, PARAHOME_BODY_INDEX["head"]])
        tip_drift.append(rotations["head"].T @ (tip / np.linalg.norm(tip)))
        desired = {name: PARAHOME_TO_HABITAT @ positions[frame_id, index] for name, index in PARAHOME_BODY_INDEX.items()}
        solved = retargeter._solve_global_rotations(desired)
        for parent, children in PARAHOME_CHILDREN.items():
            for child in children:
                produced = solved[parent] @ retargeter._rest_offsets[child]
                value = vector_angle_deg(desired[child] - desired[parent], produced)
                bone_error.append(value)
                if f"{parent}-{child}" not in DEFINITIONAL_MISMATCH:
                    bone_error_matched.append(value)
    drift = np.asarray(tip_drift)
    mean = drift.mean(axis=0) / np.linalg.norm(drift.mean(axis=0))
    drift_angle = np.degrees(np.arccos(np.clip(drift @ mean, -1, 1)))
    return {
        "facing_vs_observed_shoulder_geometry_deg": {name: float(np.mean(values)) for name, values in facing.items()},
        "head_vs_chest_relative_rotation_deg": {
            "mean": float(np.mean(relative)),
            "max": float(np.max(relative)),
            "std": float(np.std(relative)),
        },
        "head_tip_drift_in_head_frame_deg": {"mean": float(drift_angle.mean()), "max": float(drift_angle.max())},
        "segment_direction_fit_error_deg": {"mean": float(np.mean(bone_error)), "max": float(np.max(bone_error))},
        "segment_direction_fit_error_matched_joints_deg": {
            "mean": float(np.mean(bone_error_matched)),
            "max": float(np.max(bone_error_matched)),
        },
    }


def chirality_check(root: Path, sequence: str, retargeter: ParaHomeSkeletonRetargeter) -> Dict[str, float]:
    """Body facing against the recorded facing, for both coordinate mappings.

    A proper-rotation solver cannot fit a mirrored target, so the reflection
    makes the fitted body frame face backwards while the hand positions stay
    correct.  Measured for a few frames spread over the sequence.
    """
    sequence_path = root / "data" / "seq" / sequence
    with (sequence_path / "joint_positions.pkl").open("rb") as handle:
        positions = np.asarray(pickle.load(handle), dtype=np.float64)
    frame_ids = np.linspace(0, len(positions) - 1, 25, dtype=np.int64)
    results: Dict[str, float] = {}
    import activeview.data.motion.parahome_retarget as module

    original = module.PARAHOME_TO_HABITAT
    try:
        for label, mapping in (("proper_det_plus_1", original), ("reflection_det_minus_1", MIRROR_MAPPING)):
            module.PARAHOME_TO_HABITAT = mapping
            converter = ParaHomeSkeletonRetargeter(get_humanoid_urdf_path("male_0"))
            # The solve is per-frame, so only the sampled frames need retargeting.
            converted, _ = converter.retarget(positions[frame_ids].astype(np.float32), fps=30.0)
            joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
            roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
            dots = []
            for index, frame_id in enumerate(frame_ids):
                source = positions[frame_id]
                recorded = np.cross(
                    source[PARAHOME_BODY_INDEX["left_shoulder"]] - source[PARAHOME_BODY_INDEX["right_shoulder"]],
                    source[PARAHOME_BODY_INDEX["neck"]] - source[PARAHOME_BODY_INDEX["pelvis"]],
                )
                recorded /= np.linalg.norm(recorded)
                rotations = world_rotations(converter, joints[index], roots[index, :3, :3])
                chest = rotations["spine3"] @ np.asarray([0.0, 0.0, 1.0])
                dots.append(float(chest @ (mapping @ recorded)))
            results[label] = float(np.mean(dots))
    finally:
        module.PARAHOME_TO_HABITAT = original
    return results


def main() -> None:
    urdf = get_humanoid_urdf_path("male_0")
    report: Dict[str, object] = {"sequence_metrics": {}, "synthetic_whole_body_yaw_180_deg": {}}

    for label, factory in (("pre_fix_world_space", PreFixRetargeter), ("shipped_hierarchical", ParaHomeSkeletonRetargeter)):
        retargeter = factory(urdf)
        report["synthetic_whole_body_yaw_180_deg"][label] = synthetic_case(retargeter)
        for sequence in SEQUENCES:
            report["sequence_metrics"].setdefault(sequence, {})[label] = real_sequence_metrics(PARAHOME_ROOT, sequence, retargeter)

    report["chirality_across_sequences"] = {
        sequence: chirality_check(PARAHOME_ROOT, sequence, ParaHomeSkeletonRetargeter(urdf))
        for sequence in CHIRALITY_SEQUENCES
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("synthetic ground truth (whole-body yaw 180 deg + 20 deg head pitch), orientation error [deg]")
    for label, values in report["synthetic_whole_body_yaw_180_deg"].items():
        print(f"  {label:22s} " + "  ".join(f"{name}={value:6.1f}" for name, value in values.items()))
    for sequence, variants in report["sequence_metrics"].items():
        print(f"real sequence {sequence}")
        for label, values in variants.items():
            print(f"  {label:22s} facing(spine1/spine3/head)=" +
                  "/".join(f"{values['facing_vs_observed_shoulder_geometry_deg'][name]:5.1f}" for name in ("spine1", "spine3", "head")) +
                  f"  head-vs-chest mean/max/std={values['head_vs_chest_relative_rotation_deg']['mean']:.1f}/"
                  f"{values['head_vs_chest_relative_rotation_deg']['max']:.1f}/{values['head_vs_chest_relative_rotation_deg']['std']:.1f}"
                  f"  head-tip drift={values['head_tip_drift_in_head_frame_deg']['mean']:.1f}/"
                  f"{values['head_tip_drift_in_head_frame_deg']['max']:.1f}"
                  f"  bone fit(all)={values['segment_direction_fit_error_deg']['mean']:.2f}/"
                  f"{values['segment_direction_fit_error_deg']['max']:.2f}"
                  f"  bone fit(matched joints)={values['segment_direction_fit_error_matched_joints_deg']['mean']:.2f}/"
                  f"{values['segment_direction_fit_error_matched_joints_deg']['max']:.2f}")
    print("body facing . recorded facing (mean over 25 frames), proper vs reflected mapping")
    for sequence, values in report["chirality_across_sequences"].items():
        print(f"  {sequence:5s} proper={values['proper_det_plus_1']:+.2f}  reflection={values['reflection_det_minus_1']:+.2f}")
    print(f"wrote {OUTPUT / 'validation.json'}")


if __name__ == "__main__":
    main()
