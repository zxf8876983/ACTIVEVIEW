#!/usr/bin/env python3
"""Render one canonical RGB observation per viewpoint for HM3D-train.

The existing skeleton NPZ files are the source of truth for camera metadata,
placement and record identity.  This script restores only frame 15 of the
canonical 30-frame motion and writes record-level RGB NPZ files under a
separate output root.  It intentionally does not import or run perception.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import (
    URDF_PATH,
    MotionConverter,
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)


VERSION = "activeview-rgb-observation-v1"
FRAME_INDEX = 15
TARGET_FRAMES = 30
IMAGE_SIZE = 256
VIEWS_PER_RECORD = 32
# The user-authorized full generation uses sixteen independent Habitat workers.
WORKERS = 16
SENSOR_HEIGHT_M = 1.1
HFOV_DEG = 75.0


@dataclass(frozen=True)
class SourceRecord:
    """One canonical skeleton archive and its motion-manifest metadata."""

    scene_id: str
    region: str
    record_id: str
    source_path: Path
    motion: Mapping[str, Any]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Expected scalar metadata, got shape {array.shape}")
    return str(array.item())


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Expected scalar integer metadata, got shape {array.shape}")
    return int(array.item())


def _load_skeleton_metadata(path: Path) -> dict[str, Any]:
    """Read and validate immutable source metadata without changing the NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "viewpoint_ids",
            "scene_id",
            "region",
            "placement_id",
            "placement_position",
            "viewpoint_agent_positions",
            "viewpoint_rotations_wxyz",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Skeleton archive missing {sorted(missing)}: {path}")
        ids = np.asarray(archive["viewpoint_ids"])
        positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        rotations = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
        placement = np.asarray(archive["placement_position"], dtype=np.float32)
        if ids.shape != (VIEWS_PER_RECORD,) or not np.array_equal(
            ids.astype(np.int32), np.arange(VIEWS_PER_RECORD, dtype=np.int32)
        ):
            raise ValueError(f"Skeleton viewpoint_ids are not 0..31: {path}")
        if positions.shape != (VIEWS_PER_RECORD, 3):
            raise ValueError(f"Invalid viewpoint positions shape {positions.shape}: {path}")
        if rotations.shape != (VIEWS_PER_RECORD, 4):
            raise ValueError(f"Invalid viewpoint rotations shape {rotations.shape}: {path}")
        if placement.shape != (3,) or not np.isfinite(placement).all():
            raise ValueError(f"Invalid placement position: {path}")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError(f"Non-finite camera metadata: {path}")
        return {
            "scene_id": _scalar_text(archive["scene_id"]),
            "region": _scalar_text(archive["region"]),
            "record_id": path.stem,
            "placement_id": _scalar_text(archive["placement_id"]),
            "placement_position": placement,
            "viewpoint_ids": ids.astype(np.int32),
            "viewpoint_agent_positions": positions,
            "viewpoint_rotations_wxyz": rotations,
        }


def _load_source_records(
    source_root: Path,
    motion_manifest: Path,
    scene_ids: Sequence[str] | None = None,
) -> tuple[list[SourceRecord], list[str]]:
    """Load all canonical skeleton records and match them to motion metadata."""
    motions = json.loads(motion_manifest.read_text(encoding="utf-8"))
    if not isinstance(motions, list):
        raise ValueError(f"Motion manifest must be a JSON list: {motion_manifest}")
    motion_by_id = {str(item["record_id"]): dict(item) for item in motions}
    if len(motion_by_id) != len(motions):
        raise ValueError("Motion manifest contains duplicate record_id values")
    selected = list(scene_ids) if scene_ids is not None else sorted(
        path.parent.name for path in source_root.glob("*/manifest.json")
    )
    records: list[SourceRecord] = []
    for scene_id in selected:
        scene_manifest = source_root / scene_id / "manifest.json"
        if not scene_manifest.is_file():
            raise FileNotFoundError(f"Missing canonical scene manifest: {scene_manifest}")
        payload = json.loads(scene_manifest.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            rel = Path(str(item["path"]))
            source_path = source_root / scene_id / rel
            if source_path.suffix != ".npz" or not source_path.is_file():
                raise FileNotFoundError(f"Missing canonical skeleton archive: {source_path}")
            record_id = str(item["record_id"])
            motion = motion_by_id.get(record_id)
            if motion is None:
                raise KeyError(f"No motion manifest entry for {record_id}")
            metadata = _load_skeleton_metadata(source_path)
            for key in ("scene_id", "region", "record_id", "placement_id"):
                expected = scene_id if key == "scene_id" else (str(item[key]) if key in item else record_id)
                if metadata[key] != expected:
                    raise ValueError(
                        f"Skeleton metadata mismatch for {source_path}: {key}={metadata[key]!r}, expected={expected!r}"
                    )
            if not np.allclose(
                metadata["placement_position"],
                np.asarray(item["placement_position"], dtype=np.float32),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(f"Placement metadata mismatch: {source_path}")
            records.append(
                SourceRecord(
                    scene_id=scene_id,
                    region=str(item["region"]),
                    record_id=record_id,
                    source_path=source_path,
                    motion=motion,
                )
            )
    records.sort(key=lambda row: (row.scene_id, row.region, row.record_id))
    return records, selected


def _simulator(scene_root: Path, scene_id: str) -> tuple[Any, Any]:
    """Create one physics-enabled Habitat simulator and one male_0 human."""
    import habitat_sim
    import magnum as mn

    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb"))
    navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.enable_physics = True
    agents = []
    for index in range(4):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = f"color_{index}"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [IMAGE_SIZE, IMAGE_SIZE]
        sensor.position = mn.Vector3(0.0, SENSOR_HEIGHT_M, 0.0)
        sensor.hfov = HFOV_DEG
        agent = habitat_sim.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        agents.append(agent)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, agents))
    sim.pathfinder.load_nav_mesh(str(navmesh))
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
        str(get_humanoid_urdf_path("male_0"))
    )
    return sim, human


def _set_agent_state(agent: Any, position: np.ndarray, rotation_wxyz: np.ndarray) -> None:
    import habitat_sim
    import quaternion

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quaternion.from_float_array(np.asarray(rotation_wxyz, dtype=np.float32))
    agent.set_state(state)


def _render_record(sim: Any, human: Any, record: SourceRecord, source_meta: Mapping[str, Any]) -> np.ndarray:
    """Restore frame 15 and render the 32 saved camera states in eight batches."""
    motion = _load_resampled_motion(record.motion, TARGET_FRAMES)
    converter = MotionConverter(URDF_PATH)
    converted = converter.convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0)
    base = np.asarray(source_meta["placement_position"], dtype=np.float32)
    apply_humanoid_pose(
        human,
        joints[FRAME_INDEX],
        roots[FRAME_INDEX],
        base_position=base,
        scene_yaw_deg=0.0,
        floor_y=float(base[1]),
        grounding_offset=float(offsets[FRAME_INDEX]),
    )
    positions = np.asarray(source_meta["viewpoint_agent_positions"], dtype=np.float32)
    rotations = np.asarray(source_meta["viewpoint_rotations_wxyz"], dtype=np.float32)
    images: list[np.ndarray] = []
    for start in range(0, VIEWS_PER_RECORD, 4):
        for index in range(4):
            view_index = start + index
            _set_agent_state(sim.get_agent(index), positions[view_index], rotations[view_index])
        observations = sim.get_sensor_observations(list(range(4)))
        for index in range(4):
            image = np.asarray(observations[index][f"color_{index}"])
            if image.ndim != 3 or image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE) or image.shape[2] < 3:
                raise ValueError(f"Unexpected Habitat RGB shape for {record.record_id}: {image.shape}")
            rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
            if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError(f"Unexpected RGB shape after conversion: {rgb.shape}")
            images.append(rgb.copy())
    output = np.stack(images, axis=0)
    if output.shape != (VIEWS_PER_RECORD, IMAGE_SIZE, IMAGE_SIZE, 3) or output.dtype != np.uint8:
        raise ValueError(f"Invalid rendered RGB array: {output.shape} {output.dtype}")
    if not np.isfinite(output).all() or int(output.max()) > 255:
        raise ValueError(f"Invalid RGB values for {record.record_id}")
    return output


def _output_path(output_root: Path, source_root: Path, source_path: Path) -> Path:
    relative = source_path.relative_to(source_root)
    return output_root / relative


def _validate_rgb_file(
    path: Path,
    source_path: Path,
    source_meta: Mapping[str, Any],
    source_sha256: str | None = None,
) -> bool:
    """Return whether an existing output is safe to resume."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "rgb_observation_version",
                "rgb",
                "viewpoint_ids",
                "scene_id",
                "region",
                "record_id",
                "placement_id",
                "frame_index",
                "image_size",
                "source_skeleton_relative_path",
                "source_skeleton_sha256",
                "viewpoint_agent_positions",
                "viewpoint_rotations_wxyz",
            }
            if not required.issubset(archive.files):
                return False
            rgb = np.asarray(archive["rgb"])
            if rgb.dtype != np.uint8 or rgb.shape != (32, 256, 256, 3):
                return False
            if not np.array_equal(np.asarray(archive["viewpoint_ids"]).astype(np.int32), source_meta["viewpoint_ids"]):
                return False
            if _scalar_text(archive["rgb_observation_version"]) != VERSION:
                return False
            if _scalar_text(archive["scene_id"]) != str(source_meta["scene_id"]):
                return False
            if _scalar_text(archive["region"]) != str(source_meta["region"]):
                return False
            if _scalar_text(archive["record_id"]) != str(source_meta["record_id"]):
                return False
            if _scalar_text(archive["placement_id"]) != str(source_meta["placement_id"]):
                return False
            if _scalar_int(archive["frame_index"]) != FRAME_INDEX or _scalar_int(archive["image_size"]) != IMAGE_SIZE:
                return False
            if _scalar_text(archive["source_skeleton_relative_path"]) != str(source_path):
                return False
            expected_hash = source_sha256 or _sha256(source_path)
            if _scalar_text(archive["source_skeleton_sha256"]) != expected_hash:
                return False
            if not np.array_equal(np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32), source_meta["viewpoint_agent_positions"]):
                return False
            if not np.array_equal(np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32), source_meta["viewpoint_rotations_wxyz"]):
                return False
            return bool(rgb.max() > 0 and np.isfinite(rgb).all())
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False


def _write_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_entry(
    worker_id: int,
    tasks: Sequence[SourceRecord],
    source_root: Path,
    output_root: Path,
    scene_root: Path,
    progress_path: Path,
) -> None:
    """Render a disjoint task shard in one short-lived Habitat process.

    Habitat's OpenGL teardown can hang when many independent contexts are
    destroyed in the normal Python interpreter shutdown path.  All output
    files and progress metadata are closed before the successful worker exits;
    the operating system then reclaims the simulator process and its GPU
    resources without running the problematic global destructors.
    """
    sim, human = _simulator(scene_root, tasks[0].scene_id)
    completed: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(tasks, 1):
            source_meta = _load_skeleton_metadata(record.source_path)
            source_hash = _sha256(record.source_path)
            output_path = _output_path(output_root, source_root, record.source_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_file() and _validate_rgb_file(output_path, record.source_path.relative_to(source_root), source_meta, source_hash):
                status = "skipped_valid"
                byte_count = output_path.stat().st_size
            else:
                rgb = _render_record(sim, human, record, source_meta)
                arrays = {
                    "rgb_observation_version": np.asarray(VERSION),
                    "rgb": rgb,
                    "viewpoint_ids": source_meta["viewpoint_ids"].astype(np.int32),
                    "scene_id": np.asarray(record.scene_id),
                    "region": np.asarray(record.region),
                    "record_id": np.asarray(record.record_id),
                    "placement_id": np.asarray(source_meta["placement_id"]),
                    "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
                    "image_size": np.asarray(IMAGE_SIZE, dtype=np.int32),
                    "source_skeleton_relative_path": np.asarray(str(record.source_path.relative_to(source_root))),
                    "source_skeleton_sha256": np.asarray(source_hash),
                    "viewpoint_agent_positions": source_meta["viewpoint_agent_positions"].astype(np.float32),
                    "viewpoint_rotations_wxyz": source_meta["viewpoint_rotations_wxyz"].astype(np.float32),
                }
                _write_atomic(output_path, arrays)
                if not _validate_rgb_file(output_path, record.source_path.relative_to(source_root), source_meta, source_hash):
                    raise ValueError(f"Output failed post-write validation: {output_path}")
                status = "generated"
                byte_count = output_path.stat().st_size
            completed.append(
                {
                    "scene_id": record.scene_id,
                    "region": record.region,
                    "record_id": record.record_id,
                    "relative_path": str(record.source_path.relative_to(source_root)),
                    "status": status,
                    "bytes": byte_count,
                    "source_skeleton_sha256": source_hash,
                }
            )
            if index % 20 == 0:
                progress_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    except BaseException:
        progress_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        raise
    progress_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    os._exit(0)


def _scene_manifest(output_root: Path, scene_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = output_root / scene_id / "rgb_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "scene_id": scene_id,
        "frame_index": FRAME_INDEX,
        "image_size": IMAGE_SIZE,
        "views_per_record": VIEWS_PER_RECORD,
        "dtype": "uint8",
        "records": len(rows),
        "items": list(rows),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def audit_dataset(source_root: Path, output_root: Path, records: Sequence[SourceRecord]) -> dict[str, Any]:
    """Audit one-to-one correspondence and measure actual output size."""
    expected = {str(row.source_path.relative_to(source_root)): row for row in records}
    output_files = {
        str(path.relative_to(output_root)): path
        for path in output_root.rglob("*.npz")
        if path.is_file()
    }
    missing = sorted(set(expected) - set(output_files))
    extra = sorted(set(output_files) - set(expected))
    invalid: list[str] = []
    valid_bytes = 0
    for relative, record in expected.items():
        path = output_files.get(relative)
        if path is None:
            continue
        source_meta = _load_skeleton_metadata(record.source_path)
        source_hash = _sha256(record.source_path)
        if not _validate_rgb_file(path, Path(relative), source_meta, source_hash):
            invalid.append(relative)
        else:
            valid_bytes += path.stat().st_size
    total_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    result = {
        "skeleton_record_count": len(expected),
        "rgb_record_count": len(output_files),
        "missing_rgb_count": len(missing),
        "extra_rgb_count": len(extra),
        "invalid_rgb_count": len(invalid),
        "total_viewpoint_count": len(output_files) * VIEWS_PER_RECORD,
        "valid_record_npz_bytes": valid_bytes,
        "output_total_bytes": total_bytes,
        "output_total_gib": total_bytes / (1024**3),
        "average_npz_bytes_per_record": valid_bytes / len(expected) if expected else 0.0,
        "average_compressed_bytes_per_rgb_frame": valid_bytes / (len(expected) * VIEWS_PER_RECORD) if expected else 0.0,
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "invalid_examples": invalid[:20],
    }
    if missing or extra or invalid:
        raise RuntimeError(f"RGB completeness audit failed: {json.dumps(result)}")
    return result


def _write_dataset_summary(
    output_root: Path,
    source_root: Path,
    scenes: Sequence[str],
    records: Sequence[SourceRecord],
    audit: Mapping[str, Any],
    source_manifest: Path,
    workers: int,
) -> None:
    payload = {
        "version": VERSION,
        "source_skeleton_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "scenes": list(scenes),
        "scene_count": len(scenes),
        "records": len(records),
        "views_per_record": VIEWS_PER_RECORD,
        "frame_index": FRAME_INDEX,
        "image_size": IMAGE_SIZE,
        "dtype": "uint8",
        "rgb_saved": True,
        "depth_saved": False,
        "semantic_saved": False,
        "workers": workers,
        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "skeleton_modified": False,
        "skeleton_regenerated": False,
        "yolo_used": False,
        "videopose3d_used": False,
        "stgcn_used": False,
        "audit": dict(audit),
    }
    target = output_root / "dataset_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate(
    *,
    source_root: Path,
    output_root: Path,
    motion_manifest: Path,
    scene_root: Path,
    workers: int = WORKERS,
    smoke: bool = False,
    scene_limit: int | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Source skeleton root and RGB output root must differ")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Source and RGB roots must not contain one another")
    if workers != WORKERS:
        raise ValueError(f"RGB V1 requires exactly {WORKERS} workers")
    summary = json.loads((source_root / "dataset_summary.json").read_text(encoding="utf-8"))
    all_scenes = [str(scene) for scene in summary.get("scenes_requested", [])]
    if not all_scenes:
        raise ValueError("Canonical dataset summary has no scenes_requested")
    selected_scenes = all_scenes[:scene_limit] if scene_limit is not None else all_scenes
    records, selected = _load_source_records(source_root, motion_manifest, selected_scenes)
    if smoke:
        selected_ids: set[tuple[str, str]] = set()
        for row in records:
            key = (row.scene_id, row.region)
            if sum(1 for scene, region in selected_ids if scene == row.scene_id and region == row.region) < 2:
                selected_ids.add((row.scene_id, row.region))
        filtered: list[SourceRecord] = []
        counts: dict[tuple[str, str], int] = {}
        for row in records:
            key = (row.scene_id, row.region)
            if key not in selected_ids or counts.get(key, 0) >= 2:
                continue
            filtered.append(row)
            counts[key] = counts.get(key, 0) + 1
        records = filtered
        selected_scenes = selected_scenes[:1]
        records = [row for row in records if row.scene_id == selected_scenes[0]]
    output_root.mkdir(parents=True, exist_ok=True)
    by_scene: dict[str, list[SourceRecord]] = {}
    for row in records:
        by_scene.setdefault(row.scene_id, []).append(row)
    for scene_id in selected_scenes:
        scene_records = by_scene.get(scene_id, [])
        if not scene_records:
            raise ValueError(f"No canonical records selected for scene {scene_id}")
        worker_dir = output_root / scene_id / ".workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        processes: list[mp.Process] = []
        for worker_id in range(workers):
            tasks = scene_records[worker_id::workers]
            if not tasks:
                continue
            process = mp.get_context("spawn").Process(
                target=_worker_entry,
                args=(worker_id, tasks, source_root, output_root, scene_root, worker_dir / f"worker_{worker_id}.json"),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"RGB workers failed for scene {scene_id}: {[p.exitcode for p in processes]}")
        rows: list[dict[str, Any]] = []
        for worker_id in range(workers):
            progress = worker_dir / f"worker_{worker_id}.json"
            if progress.exists():
                rows.extend(json.loads(progress.read_text(encoding="utf-8")))
        rows.sort(key=lambda item: str(item["relative_path"]))
        _scene_manifest(output_root, scene_id, rows)
        for progress in worker_dir.glob("worker_*.json"):
            progress.unlink(missing_ok=True)
        worker_dir.rmdir()
    audit = audit_dataset(source_root, output_root, records)
    _write_dataset_summary(output_root, source_root, selected_scenes, records, audit, motion_manifest, workers)
    return {"scenes": selected_scenes, "records": len(records), "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=data_root / "datasets/offline/hm3d-train")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"),
    )
    parser.add_argument(
        "--motion-manifest",
        type=Path,
        default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json",
    )
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--smoke", action="store_true", help="Render two records per region in one scene")
    parser.add_argument("--scene-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate(
        source_root=args.source_root,
        output_root=args.output_root,
        motion_manifest=args.motion_manifest,
        scene_root=args.scene_root,
        workers=args.workers,
        smoke=args.smoke,
        scene_limit=args.scene_limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
