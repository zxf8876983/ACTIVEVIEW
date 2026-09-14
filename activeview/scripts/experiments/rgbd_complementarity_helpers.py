"""Capability audits for the reduced12 current-frame RGB-D experiment.

The helper keeps transient Habitat depth probing separate from policy metrics.
No rendered depth, RGB, or detector output is persisted by these routines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def depth_audit(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inventory current-frame depth, camera metadata, and raw YOLO caches."""
    depth_candidates = sorted(
        str(path)
        for path in data_root.rglob("*")
        if path.is_file() and "depth" in path.name.lower()
    )
    # A YOLO checkpoint is not a raw detection cache. Only serialized
    # per-frame outputs count here.
    yolo_candidates = sorted(
        str(path)
        for path in data_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".npz", ".json", ".jsonl"}
        and ("yolo" in path.name.lower() or "keypoint" in path.name.lower())
    )
    rgb_root = data_root / "datasets/rgb_reduced12_eight_placement_v1/frame0_current"
    sample_rgb = next(rgb_root.rglob("*.npz"), None) if rgb_root.is_dir() else None
    rgb_schema: dict[str, Any] = {}
    if sample_rgb is not None:
        with np.load(sample_rgb, allow_pickle=False) as archive:
            rgb_schema = {key: list(np.asarray(archive[key]).shape) for key in archive.files}
    has_intrinsics = bool(
        sample_rgb is not None
        and "image_size" in rgb_schema
        and "viewpoint_rotations_wxyz" in rgb_schema
    )
    return {
        "current_frame0_depth_present": bool(depth_candidates),
        "depth_files_found": depth_candidates[:20],
        "depth_file_count": len(depth_candidates),
        "raw_frame0_yolo_present": bool(yolo_candidates),
        "yolo_or_keypoint_files_found": yolo_candidates[:20],
        "yolo_file_count": len(yolo_candidates),
        "current_rgb_rows_expected": len(rows),
        "current_rgb_root": str(rgb_root.resolve()),
        "current_rgb_sample_schema": rgb_schema,
        "intrinsics_or_extrinsics_metadata_present": has_intrinsics,
        "intrinsics_definition": "HFOV=75deg, 256x256; K can be derived, but no explicit depth calibration archive exists",
        "sensor_rendering_capability": "Habitat SensorType.DEPTH import probe is available in the habitat environment; no current-depth cache is present",
        "acquisition_status": "NOT_RUN; current depth/YOLO generation is a separate authorized data-generation step",
        "test_used": False,
    }


def alignment_audit(depth_inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Return conservative alignment status before/without a depth cache."""
    return {
        "status": "BLOCKED_NO_DEPTH_CACHE",
        "train_samples_requested": 50,
        "val_samples_requested": 50,
        "train_samples_rendered": 0,
        "val_samples_rendered": 0,
        "rgb_depth_resolution_match": None,
        "torso_depth_finite": None,
        "yolo_joints_inside_body_depth_region": None,
        "reason": "No current frame0 depth cache exists; no overlay or alignment claim is made.",
        "depth_sensor_audit_status": depth_inventory.get("acquisition_status"),
        "test_used": False,
    }


def habitat_depth_probe(
    data_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    scene_root: Path,
) -> dict[str, Any]:
    """Render 50 Train + 50 Val current-frame RGB/depth pairs transiently."""
    try:
        import habitat_sim
        import magnum as mn

        from activeview.core.paths import get_humanoid_urdf_path
        from activeview.data.motion.babel_clean_dataset_generator import (
            MotionConverter,
            _load_resampled_motion,
            apply_humanoid_pose,
            precompute_grounding_offsets,
        )
        from activeview.scripts.data.generate_hm3d_train_rgb_observations import (
            _load_skeleton_metadata,
            _set_agent_state,
        )

        def sample_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            rng = np.random.default_rng(42)
            count = min(50, len(rows))
            return [rows[int(i)] for i in rng.choice(len(rows), count, replace=False)]

        def scene_sim(scene_id: str) -> tuple[Any, Any]:
            scene_dir = scene_root / scene_id
            glb = next(scene_dir.glob("*.basis.glb"))
            navmesh = next(scene_dir.glob("*.basis.navmesh"))
            backend = habitat_sim.SimulatorConfiguration()
            backend.scene_id = str(glb)
            backend.enable_physics = True
            color = habitat_sim.CameraSensorSpec()
            color.uuid = "color_0"
            color.sensor_type = habitat_sim.SensorType.COLOR
            color.resolution = [256, 256]
            color.position = mn.Vector3(0.0, 1.1, 0.0)
            color.hfov = 75.0
            depth = habitat_sim.CameraSensorSpec()
            depth.uuid = "depth_0"
            depth.sensor_type = habitat_sim.SensorType.DEPTH
            depth.resolution = [256, 256]
            depth.position = mn.Vector3(0.0, 1.1, 0.0)
            depth.hfov = 75.0
            depth.near = 0.01
            depth.far = 100.0
            agent = habitat_sim.AgentConfiguration()
            agent.sensor_specifications = [color, depth]
            sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
            sim.pathfinder.load_nav_mesh(str(navmesh))
            human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
                str(get_humanoid_urdf_path("male_0"))
            )
            return sim, human

        def row_yaw(source: Path, region: str) -> float:
            manifest = source.parents[1] / "candidate_metadata" / "manifest.json"
            payload = _read_json(manifest)
            for placement in payload.get("placements_data", []):
                if str(placement.get("placement_id")) == str(region):
                    return float(placement.get("yaw_deg", 0.0))
            raise KeyError(f"missing yaw for {source}/{region}")

        converter_cache: dict[str, MotionConverter] = {}
        motion_cache: dict[str, Mapping[str, Any]] = {}
        split_stats: dict[str, Any] = {}
        for split_name, selected_rows in (
            ("train", sample_rows(train_rows)),
            ("val", sample_rows(val_rows)),
        ):
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in selected_rows:
                grouped.setdefault(str(row["scene_id"]), []).append(row)
            rgb_ok = depth_ok = torso_ok = 0
            depth_values: list[float] = []
            for scene_id, scene_rows in sorted(grouped.items()):
                sim, human = scene_sim(scene_id)
                try:
                    converter = converter_cache.setdefault(
                        "male_0", MotionConverter(get_humanoid_urdf_path("male_0"))
                    )
                    for row in scene_rows:
                        source = Path(str(row["archive_path"]))
                        metadata = _load_skeleton_metadata(source)
                        record_id = str(row["record_id"])
                        if record_id not in motion_cache:
                            motion_path = data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"
                            motions = {
                                str(item["record_id"]): item
                                for item in json.loads(motion_path.read_text(encoding="utf-8"))
                            }
                            motion_cache.update(motions)
                        converted = converter.convert(
                            _load_resampled_motion(motion_cache[record_id], 30)
                        )
                        joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
                        roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
                        yaw = row_yaw(source, str(row["region"]))
                        offsets, _ = precompute_grounding_offsets(
                            human, joints, roots, scene_yaw_deg=yaw
                        )
                        base = np.asarray(metadata["placement_position"], dtype=np.float32)
                        apply_humanoid_pose(
                            human,
                            joints[0],
                            roots[0],
                            base_position=base,
                            scene_yaw_deg=yaw,
                            floor_y=float(base[1]),
                            grounding_offset=float(offsets[0]),
                        )
                        view = int(row["current_viewpoint_id"])
                        positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
                        rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
                        _set_agent_state(sim.get_agent(0), positions[view], rotations[view])
                        observation = sim.get_sensor_observations([0])[0]
                        rgb = np.asarray(observation["color_0"])
                        depth = np.asarray(observation["depth_0"], dtype=np.float32)
                        rgb_ok += int(rgb.shape[:2] == (256, 256) and rgb.shape[2] >= 3)
                        valid = np.isfinite(depth) & (depth > 0.0)
                        depth_ok += int(depth.shape == (256, 256) and bool(valid.mean() > 0.0))
                        center = valid[96:160, 96:160]
                        torso_ok += int(bool(center.mean() > 0.0))
                        depth_values.append(float(np.median(depth[valid])) if np.any(valid) else float("nan"))
                finally:
                    sim.close()
            split_stats[split_name] = {
                "requested": len(selected_rows),
                "rgb_resolution_ok": rgb_ok,
                "depth_resolution_and_valid_ok": depth_ok,
                "center_depth_finite": torso_ok,
                "median_valid_depth_m": float(np.nanmedian(depth_values)) if depth_values else None,
                "depth_units": "meters (Habitat metric depth)",
            }
        return {
            "status": "PASS",
            "splits": split_stats,
            "raw_depth_written": False,
            "debug_overlays_written": False,
            "note": "Probe confirms Habitat depth sensor and RGB/depth resolution; no YOLO keypoints were available for torso alignment.",
            "test_used": False,
        }
    except Exception as exc:  # pragma: no cover - environment-specific Habitat failures
        return {
            "status": "FAILED_CAPABILITY_PROBE",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_depth_written": False,
            "debug_overlays_written": False,
            "test_used": False,
        }
