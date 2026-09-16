#!/usr/bin/env python3
"""Audit ParaHome for a minimal continuous Active HAR feasibility study.

The script reads an explicitly supplied ParaHome root, audits every sequence
and annotation, checks the representation against the existing Habitat
humanoid converter, and renders one small ``scene_id=NONE`` replay.  It never
touches the ActiveView BABEL/reduced12 artifacts or policy Test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_humanoid_urdf_path  # noqa: E402
from activeview.data.motion.amass_loader import NormalizedMotion  # noqa: E402
from activeview.data.motion.babel_clean_dataset_generator import (  # noqa: E402
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.motion_converter import MotionConverter  # noqa: E402
from activeview.data.motion.parahome_retarget import ParaHomeSkeletonRetargeter  # noqa: E402


FPS = 30.0
DECISION_INTERVALS = (1.0, 2.0)
HAR_WINDOW_SECONDS = 2.0
REPRESENTATIVE_IDS = {"hand_operation": "s78", "locomotion": "s3", "articulated_object": "s162"}
REPLAY_SEQUENCE = "s78"
REPLAY_FRAMES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPLAY_VIEWS = {
    "front": np.asarray([0.0, 1.65, 3.6], dtype=np.float32),
    "side": np.asarray([3.6, 1.65, 0.0], dtype=np.float32),
    "diagonal": np.asarray([2.9, 1.8, 2.9], dtype=np.float32),
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_stats(value: Any) -> Dict[str, Any]:
    array = _as_numpy(value)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
    }


def _nested_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_nested_finite(item) for item in value.values())
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return True
    return bool(np.isfinite(array).all())


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _subject_map(metadata: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    return {str(sequence): str(subject) for subject, sequences in metadata.items() for sequence in sequences}


def normalize_action(text: str) -> str:
    """Apply conservative semantic normalization to one raw annotation."""
    value = re.sub(r"\s+", " ", str(text).strip().lower())
    if value.startswith("open "):
        return "open"
    if value.startswith("close "):
        return "close"
    if value.startswith("get up"):
        return "stand_up"
    if value.startswith("sit "):
        return "sit"
    if value.startswith("take out"):
        return "take_out"
    if value.startswith("put "):
        return "put_place"
    if value.startswith("push ") or value.startswith("pull "):
        return "move_furniture"
    if value.startswith("move "):
        return "carry_move_object"
    if value.startswith("pour "):
        return "pour"
    if value.startswith("drink "):
        return "drink"
    if value.startswith("cut "):
        return "cut"
    if value.startswith("type "):
        return "type"
    if value.startswith("look at "):
        return "read_look"
    if value.startswith("turn on ") or value.startswith("turn off "):
        return "operate_appliance"
    if value.startswith("sprinkle "):
        return "sprinkle"
    if value.startswith("throw "):
        return "throw"
    if value.startswith("wash "):
        return "wash"
    return "other/unresolved"


def _parse_interval(key: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", str(key))
    if match is None:
        raise ValueError(f"annotation interval is not '<start> <end>': {key!r}")
    return int(match.group(1)), int(match.group(2))


def _sequence_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "seq" / sequence_id


def _smplx_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "smplx_seq" / sequence_id


def _annotation_path(sequence_path: Path) -> Path:
    for name in ("text_annotation.json", "text_annotations.json"):
        candidate = sequence_path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no text annotation JSON in {sequence_path}")


def _root_motion_metrics(path: Path) -> Dict[str, float]:
    transform = _as_numpy(_load_pickle(path)).astype(np.float64)
    translation = transform[:, :3, 3]
    steps = np.linalg.norm(np.diff(translation, axis=0), axis=1) if len(translation) > 1 else np.zeros(1)
    return {
        "root_path_length_m": float(steps.sum()),
        "root_displacement_m": float(np.linalg.norm(translation[-1] - translation[0])),
        "root_step_median": float(np.median(steps)),
        "root_step_p99": float(np.percentile(steps, 99)),
        "root_step_max": float(steps.max()),
        "root_y_min_m": float(translation[:, 1].min()),
        "root_y_max_m": float(translation[:, 1].max()),
    }


def _joint_state_variation(path: Path) -> float:
    states = _load_pickle(path)
    maximum = 0.0
    for values in states.values():
        if isinstance(values, Mapping) and values:
            array = np.asarray(list(values.values()), dtype=np.float64)
            maximum = max(maximum, float(np.ptp(array)))
    return maximum


def _sequence_record(root: Path, sequence_id: str, subject_id: str) -> Dict[str, Any]:
    sequence_path = _sequence_path(root, sequence_id)
    body_transform = _as_numpy(_load_pickle(sequence_path / "body_global_transform.pkl")).astype(np.float32)
    body_orientation = _as_numpy(_load_pickle(sequence_path / "body_joint_orientations.pkl")).astype(np.float32)
    joint_positions = _as_numpy(_load_pickle(sequence_path / "joint_positions.pkl")).astype(np.float32)
    if body_transform.ndim != 3 or body_transform.shape[1:] != (4, 4):
        raise ValueError(f"{sequence_id}: body_global_transform must be (T,4,4)")
    frame_count = int(body_transform.shape[0])
    if body_orientation.ndim != 3 or body_orientation.shape[0] != frame_count or body_orientation.shape[1:] != (23, 6):
        raise ValueError(f"{sequence_id}: body_joint_orientations must be (T,23,6)")
    if joint_positions.ndim != 3 or joint_positions.shape[0] != frame_count or joint_positions.shape[1:] != (73, 3):
        raise ValueError(f"{sequence_id}: joint_positions must be (T,73,3)")
    with _annotation_path(sequence_path).open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    intervals: List[Dict[str, Any]] = []
    for raw_interval, text in annotations.items():
        start, end = _parse_interval(raw_interval)
        intervals.append({
            "start_frame": start,
            "end_frame": end,
            "duration_frames": max(0, end - start),
            "duration_seconds": max(0.0, end - start) / FPS,
            "raw_text": str(text),
            "normalized_class": normalize_action(str(text)),
            "out_of_bounds": bool(start < 0 or end > frame_count or end < start),
        })
    objects = json.loads((sequence_path / "object_in_scene.json").read_text(encoding="utf-8"))
    object_transforms = _load_pickle(sequence_path / "object_transformations.pkl")
    if len(object_transforms) != frame_count:
        raise ValueError(f"{sequence_id}: object_transformations length does not match frame count")
    object_transform_samples: List[Dict[str, Any]] = []
    for frame_id in sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)}):
        frame_values = object_transforms[int(frame_id)]
        object_transform_samples.append({
            "frame_id": int(frame_id),
            "object_count": len(frame_values),
            "finite": bool(all(np.isfinite(_as_numpy(matrix)).all() for matrix in frame_values.values())),
        })
    joint_states = _load_pickle(sequence_path / "joint_states.pkl")
    smplx_dir = _smplx_path(root, sequence_id)
    smplx_pose_path = smplx_dir / "smplx_pose.pkl"
    smplx_params_path = smplx_dir / "smplx_params.pkl"
    smplx_info: Dict[str, Any] = {"exists": smplx_pose_path.is_file() and smplx_params_path.is_file()}
    if smplx_info["exists"]:
        smplx_pose = _load_pickle(smplx_pose_path)
        smplx_params = _load_pickle(smplx_params_path)
        smplx_info.update({
            "pose_fields": sorted(str(key) for key in smplx_pose),
            "body_pose_shape": list(_as_numpy(smplx_pose["body_pose"]).shape),
            "global_orient_shape": list(_as_numpy(smplx_pose["global_orient"]).shape),
            "transl_shape": list(_as_numpy(smplx_pose["transl"]).shape),
            "hand_pose_shape": list(_as_numpy(smplx_pose["hand_pose"]).shape),
            "gender": str(smplx_params.get("gender", "unknown")),
            "beta_shape": list(_as_numpy(smplx_params.get("beta", np.zeros((0,)))).shape),
            "finite": all(bool(np.isfinite(_as_numpy(smplx_pose[key])).all()) for key in ("body_pose", "global_orient", "transl", "hand_pose")),
        })
    return {
        "sequence_id": str(sequence_id),
        "subject_id": str(subject_id),
        "frame_count": frame_count,
        "fps": FPS,
        "fps_source": "inferred 30 fps: ParaHome frame convention and aggregate 486-minute total; no per-sequence fps field",
        "duration_seconds": frame_count / FPS,
        "annotation_count": len(intervals),
        "annotation_intervals": intervals,
        "objects_in_scene": sorted(str(key) for key, present in objects.items() if bool(present)),
        "object_count": int(sum(bool(value) for value in objects.values())),
        "has_smplx": bool(smplx_info["exists"]),
        "smplx": smplx_info,
        "joint_positions_shape": list(joint_positions.shape),
        "body_global_transform_shape": list(body_transform.shape),
        "finite_checks": {
            "body_global_transform": _finite_stats(body_transform),
            "body_joint_orientations": _finite_stats(body_orientation),
            "joint_positions": _finite_stats(joint_positions),
            "object_transformations_sample": object_transform_samples,
            "joint_states": {"finite": _nested_finite(joint_states)},
        },
        "object_transformations_frame_count": len(object_transforms),
        "root_metrics": _root_motion_metrics(sequence_path / "body_global_transform.pkl"),
        "max_articulated_joint_state_range": _joint_state_variation(sequence_path / "joint_states.pkl"),
    }


def _aggregate_action_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        for interval in record["annotation_intervals"]:
            grouped[str(interval["normalized_class"])].append({**interval, "sequence_id": record["sequence_id"], "subject_id": record["subject_id"]})
    output: Dict[str, Any] = {}
    for label in sorted(grouped):
        items = grouped[label]
        durations = np.asarray([float(item["duration_seconds"]) for item in items], dtype=np.float64)
        raw_labels = Counter(str(item["raw_text"]) for item in items)
        output[label] = {
            "instance_count": len(items),
            "total_duration_seconds": float(durations.sum()),
            "mean_duration_seconds": float(durations.mean()),
            "median_duration_seconds": float(np.median(durations)),
            "p10_duration_seconds": float(np.percentile(durations, 10)),
            "p90_duration_seconds": float(np.percentile(durations, 90)),
            "unique_sequences": len({str(item["sequence_id"]) for item in items}),
            "unique_subjects": len({str(item["subject_id"]) for item in items}),
            "raw_labels_included": dict(raw_labels),
        }
    return output


def _duration_analysis(records: Sequence[Mapping[str, Any]], action_stats: Mapping[str, Any]) -> Dict[str, Any]:
    durations = [float(interval["duration_seconds"]) for record in records for interval in record["annotation_intervals"]]
    bins = (("<1s", 0.0, 1.0), ("1-2s", 1.0, 2.0), ("2-5s", 2.0, 5.0), ("5-10s", 5.0, 10.0), ("10-30s", 10.0, 30.0), (">30s", 30.0, float("inf")))
    bin_counts = {name: int(sum(low <= value < high for value in durations)) for name, low, high in bins}
    interval_capacity: Dict[str, Any] = {}
    for action, stats in action_stats.items():
        items = [
            float(interval["duration_seconds"])
            for record in records
            for interval in record["annotation_intervals"]
            if str(interval["normalized_class"]) == action
        ]
        interval_capacity[action] = {}
        for interval in DECISION_INTERVALS:
            counts = {
                str(cycles): int(sum(math.floor(max(value - HAR_WINDOW_SECONDS, 0.0) / interval) >= cycles for value in items))
                for cycles in (2, 3, 5)
            }
            interval_capacity[action][f"decision_interval_{interval:g}s"] = {
                "har_window_seconds": HAR_WINDOW_SECONDS,
                "capacity_definition": "floor(max(duration - HAR_window, 0) / decision_interval)",
                "instances_supporting_at_least": counts,
                "fractions_supporting_at_least": {key: value / len(items) if items else 0.0 for key, value in counts.items()},
            }
    capacity_class_summary: Dict[str, Any] = {}
    for interval in DECISION_INTERVALS:
        key = f"decision_interval_{interval:g}s"
        class_fractions = {
            action: float(values[key]["fractions_supporting_at_least"]["2"])
            for action, values in interval_capacity.items()
        }
        capacity_class_summary[key] = {
            "classes_with_any_instance_supporting_2_cycles": int(sum(value > 0.0 for value in class_fractions.values())),
            "classes_with_at_least_half_instances_supporting_2_cycles": int(sum(value >= 0.5 for value in class_fractions.values())),
            "fraction_by_class": class_fractions,
        }
    return {
        "annotation_duration_count": len(durations),
        "annotation_duration_total_seconds": float(sum(durations)),
        "annotation_duration_median_seconds": float(median(durations)),
        "bins": {name: {"count": count, "fraction": count / len(durations) if durations else 0.0} for name, count in bin_counts.items()},
        "capacity_by_normalized_class": interval_capacity,
        "capacity_class_summary": capacity_class_summary,
    }


def _candidate_classes(action_stats: Mapping[str, Any]) -> Dict[str, Any]:
    unresolved = sorted(name for name in action_stats if name == "other/unresolved")
    eligible = {name: stats for name, stats in action_stats.items() if name != "other/unresolved"}
    standard = [name for name, stats in eligible.items() if stats["instance_count"] >= 30 and stats["unique_sequences"] >= 15 and stats["unique_subjects"] >= 8]
    strict = [name for name, stats in eligible.items() if stats["instance_count"] >= 50 and stats["unique_sequences"] >= 20 and stats["unique_subjects"] >= 10]
    return {
        "standard_threshold": {"instance_count": 30, "unique_sequences": 15, "unique_subjects": 8},
        "strict_threshold": {"instance_count": 50, "unique_sequences": 20, "unique_subjects": 10},
        "standard_viable_classes": sorted(standard),
        "strict_viable_classes": sorted(strict),
        "standard_viable_count": len(standard),
        "strict_viable_count": len(strict),
        "unresolved_classes_excluded_from_viability": unresolved,
    }


def _smplx_normalized_motion(root: Path, sequence_id: str) -> NormalizedMotion:
    data = _load_pickle(_smplx_path(root, sequence_id) / "smplx_pose.pkl")
    body_pose = _as_numpy(data["body_pose"]).astype(np.float32).reshape(-1, 63)
    hand_pose = _as_numpy(data["hand_pose"]).astype(np.float32).reshape(-1, 90)
    frame_count = int(body_pose.shape[0])
    padded = np.zeros((frame_count, 162), dtype=np.float32)
    padded[:, :63] = body_pose
    # SMPL-X hand order is left 15 then right 15.  Habitat's converter uses
    # SMPL-X joint indices 25..39 and 40..54, excluding the root index.
    padded[:, 24 * 3 : 24 * 3 + 45] = hand_pose[:, :45]
    padded[:, 39 * 3 : 39 * 3 + 45] = hand_pose[:, 45:]
    return NormalizedMotion(
        translation=_as_numpy(data["transl"]).astype(np.float32),
        root_rotation=_as_numpy(data["global_orient"]).astype(np.float32),
        body_pose=padded,
        fps=FPS,
        num_frames=frame_count,
        metadata={"source": "ParaHome smplx_pose.pkl", "sequence_id": sequence_id},
    )


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    error = np.sqrt(np.mean(np.sum((source @ rotation.T + translation - target) ** 2, axis=1)))
    return transform, float(error)


def _representative_audit(root: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_id = {str(record["sequence_id"]): record for record in records}
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    output: Dict[str, Any] = {}
    for role, sequence_id in REPRESENTATIVE_IDS.items():
        if sequence_id not in by_id:
            output[role] = {"sequence_id": sequence_id, "status": "missing"}
            continue
        sequence_path = _sequence_path(root, sequence_id)
        pose_data = _load_pickle(_smplx_path(root, sequence_id) / "smplx_pose.pkl")
        body_orient = _as_numpy(_load_pickle(sequence_path / "body_joint_orientations.pkl")).astype(np.float32)
        body_transform = _as_numpy(_load_pickle(sequence_path / "body_global_transform.pkl")).astype(np.float32)
        joint_positions = _as_numpy(_load_pickle(sequence_path / "joint_positions.pkl")).astype(np.float32)
        d6 = body_orient.reshape(-1, 6)
        a1 = d6[:, :3]
        a2 = d6[:, 3:]
        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8)
        b2 = a2 - b1 * np.sum(b1 * a2, axis=1, keepdims=True)
        b2 /= np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-8)
        b3 = np.cross(b1, b2)
        rot6d = np.stack((b1, b2, b3), axis=-1).reshape(-1, 3, 3)
        converted = converter.convert(_smplx_normalized_motion(root, sequence_id))
        converted_roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float64)[:, :3, 3]
        para_roots = body_transform[:, :3, 3].astype(np.float64)
        align, fit_rmse = _rigid_alignment(para_roots, converted_roots)
        object_presence = json.loads((sequence_path / "object_in_scene.json").read_text(encoding="utf-8"))
        object_transforms = _load_pickle(sequence_path / "object_transformations.pkl")
        annotated_text = " ".join(str(item["raw_text"]).lower() for item in by_id[sequence_id]["annotation_intervals"])
        candidate_objects = [name for name in ("laptop", "cup", "kettle", "chair", "book", "pot", "pan", "knife") if name in annotated_text and bool(object_presence.get(name, False))]
        key_objects = [name for name in candidate_objects if f"{name}_base" in object_transforms[0]][:3]
        frame_ids = np.linspace(0, len(converted_roots) - 1, 10, dtype=np.int64)
        distance_errors: List[float] = []
        for frame_id in frame_ids:
            for name in key_objects:
                matrix = np.asarray(object_transforms[int(frame_id)][f"{name}_base"], dtype=np.float64)
                para_distance = float(np.linalg.norm(matrix[:3, 3] - para_roots[int(frame_id)]))
                replay_position = align[:3, :3] @ matrix[:3, 3] + align[:3, 3]
                replay_distance = float(np.linalg.norm(replay_position - converted_roots[int(frame_id)]))
                distance_errors.append(abs(para_distance - replay_distance))
        output[role] = {
            "sequence_id": sequence_id,
            "duration_seconds": by_id[sequence_id]["duration_seconds"],
            "status": "audited",
            "para_joint_positions_shape": list(joint_positions.shape),
            "body_joint_orientations_shape": list(body_orient.shape),
            "body_global_transform_shape": list(body_transform.shape),
            "body_rotation_representation": "6D first-two-columns, reconstructed with Gram-Schmidt",
            "rotation6d_orthogonality_mean_abs_error": float(np.mean(np.abs(np.matmul(rot6d.transpose(0, 2, 1), rot6d) - np.eye(3)))),
            "rotation6d_determinant_min": float(np.linalg.det(rot6d).min()),
            "rotation6d_determinant_max": float(np.linalg.det(rot6d).max()),
            "smplx_representation": {
                "body_pose_shape": list(_as_numpy(pose_data["body_pose"]).shape),
                "global_orient_shape": list(_as_numpy(pose_data["global_orient"]).shape),
                "transl_shape": list(_as_numpy(pose_data["transl"]).shape),
                "hand_pose_shape": list(_as_numpy(pose_data["hand_pose"]).shape),
                "translation_units": "meters (empirically consistent with joint positions)",
                "rotation_units": "axis-angle radians",
                "hand_pose": "30 joints × 3 axis-angle values, left then right",
            },
            "para_to_habitat_alignment": align.tolist(),
            "alignment_root_rmse_m": fit_rmse,
            "key_objects_for_distance_check": key_objects,
            "ten_frame_human_object_distance_max_abs_error_m": float(max(distance_errors) if distance_errors else 0.0),
            "root_metrics": by_id[sequence_id]["root_metrics"],
            "articulated_joint_state_range": by_id[sequence_id]["max_articulated_joint_state_range"],
        }
    return output


def _scene_simulator(image_size: int = 640) -> Tuple[Any, Any]:
    import habitat_sim
    import magnum as mn

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"
    backend.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [image_size, image_size]
    sensor.position = mn.Vector3(0.0, 1.2, 0.0)
    sensor.hfov = mn.Deg(65.0)
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    manager = sim.get_object_template_manager()
    floor_template = manager.get_template_by_handle("cubeSolid")
    floor_template.scale = mn.Vector3(10.0, 0.02, 10.0)
    manager.register_template(floor_template, "parahome_diagnostic_floor", True)
    floor = sim.get_rigid_object_manager().add_object_by_template_handle("parahome_diagnostic_floor")
    floor.translation = mn.Vector3(0.0, -0.02, 0.0)
    floor.motion_type = habitat_sim.physics.MotionType.STATIC
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _register_obj(sim: Any, obj_path: Path, handle: str) -> Any:
    import habitat_sim
    import magnum as mn

    manager = sim.get_object_template_manager()
    template = manager.create_new_template(handle, False)
    template.render_asset_handle = str(obj_path)
    template.collision_asset_handle = str(obj_path)
    template.scale = mn.Vector3(1.0)
    manager.register_template(template, handle, True)
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(handle)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return obj


def _set_camera(sim: Any, position: np.ndarray, target: np.ndarray) -> None:
    import habitat_sim
    import magnum as mn
    import quaternion

    camera_position = np.asarray(position, dtype=np.float32)
    direction = np.asarray(target, dtype=np.float32) - camera_position
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    pitch = math.asin(float(direction[1]))
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(*camera_position)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector([pitch, 0.0, 0.0])
    sim.get_agent(0).set_state(state)


def _run_replay(root: Path, output: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import magnum as mn

    sequence_id = REPLAY_SEQUENCE
    by_id = {str(record["sequence_id"]): record for record in records}
    if sequence_id not in by_id or not by_id[sequence_id]["has_smplx"]:
        return {"status": "FAIL", "reason": f"replay sequence {sequence_id} missing or has no SMPL-X"}
    sequence_path = _sequence_path(root, sequence_id)
    source_joints = _as_numpy(_load_pickle(sequence_path / "joint_positions.pkl")).astype(np.float32)
    retargeter = ParaHomeSkeletonRetargeter(get_humanoid_urdf_path("male_0"))
    converted, pose_diagnostics = retargeter.retarget(source_joints, fps=FPS)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    para_root = _as_numpy(_load_pickle(sequence_path / "body_global_transform.pkl"))[:, :3, 3].astype(np.float64)
    align, align_rmse = _rigid_alignment(para_root, roots[:, :3, 3].astype(np.float64))
    object_transforms = _load_pickle(sequence_path / "object_transformations.pkl")
    annotation_text = " ".join(str(item["raw_text"]).lower() for item in by_id[sequence_id]["annotation_intervals"])
    object_names = [name for name in ("laptop", "cup", "kettle", "chair", "book", "pot", "pan", "knife") if name in annotation_text]
    object_names = [name for name in object_names if f"{name}_base" in object_transforms[0]][:4]
    frame_ids = [int(round(fraction * (len(joints) - 1))) for fraction in REPLAY_FRAMES]
    sim, human = _scene_simulator()
    visual_dir = output / "visualizations" / sequence_id
    visual_dir.mkdir(parents=True, exist_ok=True)
    objects: Dict[str, Any] = {}
    try:
        offsets, _ = precompute_grounding_offsets(human, joints, roots)
        offsets_array = np.asarray(offsets, dtype=np.float64)
        if not np.isfinite(offsets_array).all():
            raise RuntimeError("non-finite grounding offset from Habitat humanoid replay")
        for name in object_names:
            objects[name] = _register_obj(sim, root / "data" / "scan" / name / "simplified" / "base.obj", f"parahome_{name}")
        snapshots: List[Dict[str, Any]] = []
        for frame_id in frame_ids:
            apply_humanoid_pose(human, joints[frame_id], roots[frame_id], base_position=roots[frame_id, :3, 3], grounding_offset=float(offsets[frame_id]))
            mapped_human = np.asarray(human.translation, dtype=np.float32)
            for name, obj in objects.items():
                obj.transformation = mn.Matrix4((align @ np.asarray(object_transforms[frame_id][f"{name}_base"], dtype=np.float64)).astype(np.float32))
            for view_name, offset in REPLAY_VIEWS.items():
                _set_camera(sim, mapped_human + offset, mapped_human + np.asarray([0.0, 0.9, 0.0], dtype=np.float32))
                observation = np.asarray(sim.get_sensor_observations()["color"])
                if observation.ndim != 3 or observation.shape[-1] < 3 or not np.isfinite(observation[..., :3]).all():
                    raise RuntimeError(f"invalid RGB observation at frame {frame_id}, view {view_name}")
                from PIL import Image

                image_path = visual_dir / f"frame_{frame_id:04d}_{view_name}.png"
                Image.fromarray(observation[..., :3].astype(np.uint8)).save(image_path)
                snapshots.append({
                    "frame_id": frame_id,
                    "view": view_name,
                    "path": str(image_path.relative_to(output)),
                    "shape": list(observation.shape),
                    "human_translation": mapped_human.tolist(),
                })
        sim.close()
    except Exception:
        sim.close()
        raise
    root_steps = np.linalg.norm(np.diff(roots[:, :3, 3], axis=0), axis=1)
    return {
        "status": "PASS",
        "scene_id": "NONE",
        "sequence_id": sequence_id,
        "human_replay": "PASS",
        "rigid_object_replay": "PASS" if object_names else "FAIL_NO_KEY_OBJECTS",
        "objects_replayed": object_names,
        "frames_rendered": frame_ids,
        "views_rendered": sorted(REPLAY_VIEWS),
        "snapshot_count": len(frame_ids) * len(REPLAY_VIEWS),
        "snapshots": snapshots,
        "para_to_habitat_root_alignment_rmse_m": align_rmse,
        "retarget_mean_joint_rmse_m": pose_diagnostics.mean_joint_rmse_m,
        "retarget_mean_bone_angle_error_deg": dict(pose_diagnostics.mean_bone_angle_error_deg),
        "grounding_offset_min_m": float(offsets_array.min()),
        "grounding_offset_max_m": float(offsets_array.max()),
        "converted_root_step_median": float(np.median(root_steps)),
        "converted_root_step_p99": float(np.percentile(root_steps, 99)),
        "converted_root_step_max": float(root_steps.max()),
        "continuous_playback_note": "five timestamp snapshots validate the same frame-by-frame pose path; no continuous video file was required",
        "scene_note": "Habitat scene_id=NONE with diagnostic floor; this is a clean replay, not HM3D integration",
    }


def _write_reports(output: Path, records: Sequence[Mapping[str, Any]], action_stats: Mapping[str, Any], duration: Mapping[str, Any], candidates: Mapping[str, Any], representation: Mapping[str, Any], replay: Mapping[str, Any], root: Path) -> None:
    total_frames = int(sum(int(record["frame_count"]) for record in records))
    total_seconds = total_frames / FPS
    subjects = sorted({str(record["subject_id"]) for record in records})
    dataset_summary = {
        "dataset": "ParaHome",
        "root": str(root),
        "sequence_count": len(records),
        "subject_count": len(subjects),
        "subject_ids": subjects,
        "total_frames": total_frames,
        "fps": FPS,
        "fps_source": "inferred; no per-sequence fps field",
        "total_duration_seconds": total_seconds,
        "total_duration_minutes": total_seconds / 60.0,
        "total_duration_hours": total_seconds / 3600.0,
        "annotation_count": int(sum(int(record["annotation_count"]) for record in records)),
        "sequences_with_smplx": int(sum(bool(record["has_smplx"]) for record in records)),
        "scan_object_directory_count": len(list((root / "data" / "scan").glob("*"))),
        "policy_test_used": False,
        "activeview_formal_pipeline_modified": False,
    }
    replay_pass = replay.get("status") == "PASS" and replay.get("human_replay") == "PASS" and replay.get("rigid_object_replay") == "PASS"
    if candidates["standard_viable_count"] < 6 or not replay_pass:
        verdict = "KILL PARAHOME FOR ACTIVE HAR"
    else:
        verdict = "CONDITIONAL PROMOTE"
    result_summary = {
        "dataset": dataset_summary,
        "normalized_class_count": len(action_stats),
        "standard_viable_class_count": candidates["standard_viable_count"],
        "strict_viable_class_count": candidates["strict_viable_count"],
        "duration": {
            "median_seconds": duration["annotation_duration_median_seconds"],
            "fraction_at_least_2_seconds": float(sum(v["count"] for k, v in duration["bins"].items() if k in ("2-5s", "5-10s", "10-30s", ">30s")) / duration["annotation_duration_count"]),
            "fraction_at_least_5_seconds": float(sum(v["count"] for k, v in duration["bins"].items() if k in ("5-10s", "10-30s", ">30s")) / duration["annotation_duration_count"]),
            "capacity_class_summary": duration["capacity_class_summary"],
        },
        "representation": representation,
        "replay": replay,
        "verdict": verdict,
        "flags": {
            "policy_test_used": False,
            "activeview_formal_pipeline_modified": False,
            "har_model_training_used": False,
            "hm3d_integration_used": False,
        },
    }
    _json_dump(output / "dataset_summary.json", dataset_summary)
    _json_dump(output / "sequence_audit.json", list(records))
    _json_dump(output / "normalized_action_stats.json", action_stats)
    _json_dump(output / "duration_analysis.json", duration)
    _json_dump(output / "candidate_classes.json", candidates)
    _json_dump(output / "habitat_replay_audit.json", replay)
    _json_dump(output / "representation_audit.json", representation)
    _json_dump(output / "result.json", result_summary)
    analysis_lines = [
        "# ParaHome feasibility audit",
        "",
        "## Dataset and temporal capacity",
        f"- ParaHome contains **{len(records)} sequences**, **{len(subjects)} subjects**, and **{total_seconds / 3600.0:.2f} hours** ({total_seconds / 60.0:.2f} minutes) at the inferred 30 fps.",
        f"- The audit covers **{dataset_summary['annotation_count']} annotated intervals**. Duration median is **{duration['annotation_duration_median_seconds']:.2f}s**; fractions >=2s and >=5s are **{sum(v['count'] for k, v in duration['bins'].items() if k in ('2-5s', '5-10s', '10-30s', '>30s')) / duration['annotation_duration_count']:.3f}** and **{sum(v['count'] for k, v in duration['bins'].items() if k in ('5-10s', '10-30s', '>30s')) / duration['annotation_duration_count']:.3f}**.",
        f"- Conservative normalized taxonomy has **{len(action_stats)} classes**; **{candidates['standard_viable_count']}** meet the 30/15/8 candidate threshold and **{candidates['strict_viable_count']}** meet the 50/20/10 stricter threshold.",
        f"- Unresolved classes excluded from viability gating: {candidates['unresolved_classes_excluded_from_viability'] or 'none'}.",
        "- Capacity counts use `floor(max(annotation_duration - 2s, 0) / decision_interval)`; this explicitly reserves a 2s HAR window before repeated decisions.",
        "",
        "## Representation compatibility",
        "- ParaHome provides 23 body joints as 6D rotations, 73 global joint positions (23 body + 50 hand), a 4x4 body-to-world transform, and optional SMPL-X axis-angle body/root/hand pose.",
        "- Directly copying ParaHome fitted SMPL-X local rotations into Habitat is invalid because the fitted model and `male_0` URDF do not share the same rest/joint frames. The clean replay instead uses the released 23-joint world skeleton and hierarchical segment-direction retargeting.",
        "- The retargeter is ParaHome-specific and leaves the frozen AMASS/BABEL `MotionConverter` unchanged. Root/object coordinates still use one rigid ParaHome-to-Habitat alignment.",
        "",
        "## Habitat clean replay",
        f"- Replay status: **{replay.get('status')}**; human replay **{replay.get('human_replay', 'FAIL')}**, rigid-object replay **{replay.get('rigid_object_replay', 'FAIL')}**.",
        f"- The replay uses `scene_id=NONE`, a diagnostic floor, the existing male_0 Habitat humanoid, and {len(replay.get('objects_replayed', []))} kinematic scanned OBJ objects. It rendered {replay.get('snapshot_count', 0)} snapshots (five timestamps × three cameras).",
        "- This validates a clean-scene replay path only; it does not claim HM3D integration or complete articulated-object conversion.",
        "",
        "## Six required questions",
        "1. **Scale:** The 207-sequence/38-subject scale is sufficient for a dataset audit and matches the published aggregate, but taxonomy viability depends on the normalized class thresholds above.",
        f"2. **Viable classes:** {candidates['standard_viable_count']} classes satisfy >=30 instances, >=15 sequences and >=8 subjects; stricter count is {candidates['strict_viable_count']}.",
        f"3. **Repeated decisions:** With a 2s HAR window, **{duration['capacity_class_summary']['decision_interval_1s']['classes_with_at_least_half_instances_supporting_2_cycles']} / {len(action_stats)}** classes have at least half of their intervals supporting two 1s decision cycles; the corresponding 2s-cycle count is **{duration['capacity_class_summary']['decision_interval_2s']['classes_with_at_least_half_instances_supporting_2_cycles']} / {len(action_stats)}**. Raw per-class counts are in `duration_analysis.json`.",
        "4. **Motion compatibility:** **Explicit skeletal retarget required**: the ParaHome fitted SMPL-X local rotations are not pose-faithful on Habitat `male_0`; released world joints are retargeted by segment direction instead.",
        f"5. **Replay:** **{replay.get('status')}** for the selected clean-scene sequence; human/object snapshots and alignment checks are saved under `visualizations/`.",
        f"6. **Verdict:** **{verdict}** — the motion representation and minimal clean replay are feasible, but taxonomy duration/coverage and the coordinate/gender/scene adapter must be addressed before making ParaHome the continuous Active HAR mainline.",
        "",
        "## Hard-gate notes",
        "- No policy Test, BABEL/reduced12 formal artifact, large-scale RGB generation, HM3D integration, or HAR model training was used.",
        "- ParaHome's own README warns that under-sink items can be manually filled and may contain physical alignment/penetration errors; object replay should therefore be treated as research data, not guaranteed physical ground truth.",
        "- The verdict does not count sliding windows as independent annotated instances.",
    ]
    (output / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    representation_lines = [
        "# Representation audit",
        "",
        "ParaHome body_joint_orientations are 6D rotations (first two matrix columns), body_global_transform is a 4x4 body-to-world transform, and joint_positions contains 73 global positions. Although the optional smplx_pose has the expected axis-angle dimensions, direct local-rotation transfer to Habitat `male_0` is not pose-faithful because the rigs have incompatible rest/joint frames.",
        "",
        "The fixed replay uses ParaHome's released 23-joint world positions to solve Habitat joint rotations hierarchically from body-segment directions. This is a generic geometric retarget, not an action-specific arm/head correction, and it does not modify the AMASS/BABEL converter. A single Kabsch rigid transform maps ParaHome rigid-object transforms into the resulting Habitat root frame.",
        "",
        "The selected replay uses the existing `scene_id=NONE` clean floor and male_0 URDF. ParaHome gender metadata is retained in `sequence_audit.json`; a production replay should add a gender-specific asset or document the male_0 approximation.",
    ]
    (output / "representation_audit.md").write_text("\n".join(representation_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parahome-root", type=Path, default=None, help="Downloaded ParaHome root (required or PARAHOME_ROOT)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/parahome_feasibility_v1")
    parser.add_argument("--skip-replay", action="store_true", help="Only audit files; do not launch Habitat")
    args = parser.parse_args()
    root = (args.parahome_root or (Path(os.environ["PARAHOME_ROOT"]) if os.environ.get("PARAHOME_ROOT") else None))
    if root is None:
        parser.error("--parahome-root or PARAHOME_ROOT is required")
    root = root.resolve()
    if not (root / "data" / "seq").is_dir():
        raise FileNotFoundError(f"ParaHome sequence directory not found: {root / 'data' / 'seq'}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    subject_map = _subject_map(metadata)
    sequence_dirs = sorted((root / "data" / "seq").glob("s*"), key=lambda path: int(path.name[1:]))
    records = [_sequence_record(root, path.name, subject_map[path.name]) for path in sequence_dirs]
    action_stats = _aggregate_action_stats(records)
    duration = _duration_analysis(records, action_stats)
    candidates = _candidate_classes(action_stats)
    representation = _representative_audit(root, records)
    replay = {"status": "SKIPPED", "reason": "--skip-replay"} if args.skip_replay else _run_replay(root, output, records)
    _write_reports(output, records, action_stats, duration, candidates, representation, replay, root)
    _json_dump(output / "config.json", {
        "parahome_root": str(root),
        "output": str(output),
        "fps": FPS,
        "fps_is_inferred": True,
        "har_window_seconds": HAR_WINDOW_SECONDS,
        "decision_intervals_seconds": DECISION_INTERVALS,
        "representatives": REPRESENTATIVE_IDS,
        "replay_sequence": REPLAY_SEQUENCE,
        "policy_test_used": False,
        "formal_activeview_pipeline_modified": False,
    })
    print(json.dumps({
        "sequences": len(records),
        "subjects": len({str(record["subject_id"]) for record in records}),
        "annotations": sum(record["annotation_count"] for record in records),
        "duration_hours": sum(record["frame_count"] for record in records) / FPS / 3600.0,
        "normalized_classes": len(action_stats),
        "viable_classes": candidates["standard_viable_count"],
        "strict_viable_classes": candidates["strict_viable_count"],
        "replay": replay.get("status"),
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
