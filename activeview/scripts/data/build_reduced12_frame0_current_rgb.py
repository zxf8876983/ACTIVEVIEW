#!/usr/bin/env python3
"""Render only the current viewpoint at frame 0 for reduced12 policy data.

This cache is intentionally different from the historical visited ``s0/s1``
cache (which stores frame 15 and may contain two viewpoints).  Every output
archive contains exactly one rendered image: the policy row's current
viewpoint.  The zero-filled slots are never used as observations and are
marked unavailable in the mask.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.data.motion.babel_clean_dataset_generator import (
    URDF_PATH,
    MotionConverter,
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.scripts.data.generate_hm3d_train_rgb_observations import (
    IMAGE_SIZE,
    _load_skeleton_metadata,
    _set_agent_state,
    _simulator,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import load_rows


FRAME_INDEX = 0
TARGET_FRAMES = 30
RGB_VERSION = "activeview-rgb-frame0-current-v1"
OUTPUT_REL = "datasets/rgb_reduced12_eight_placement_v1/frame0_current"
MOTION_REL = "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"
ARCHIVE_REL = "datasets/offline/habitat-train/00006-00087"


@dataclass(frozen=True)
class RenderTask:
    scene_id: str
    region: str
    record_id: str
    current_viewpoint_id: int
    source_path: Path
    motion: Mapping[str, Any]


def _load_motions(data_root: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads((data_root / MOTION_REL).read_text(encoding="utf-8"))
    return {str(item["record_id"]): dict(item) for item in payload}


def _tasks(data_root: Path, splits: Sequence[str]) -> list[RenderTask]:
    train_rows, val_rows = load_rows(data_root)
    selected: list[Mapping[str, Any]] = []
    if "train" in splits:
        selected.extend(train_rows)
    if "val" in splits:
        selected.extend(val_rows)
    motions = _load_motions(data_root)
    tasks: list[RenderTask] = []
    seen: set[tuple[str, str, str]] = set()
    for row in selected:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        if key in seen:
            raise ValueError(f"duplicate frame-0 RGB key: {key}")
        seen.add(key)
        source = Path(str(row["archive_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        motion = motions.get(str(row["record_id"]))
        if motion is None:
            raise KeyError(f"missing motion metadata for {row['record_id']}")
        tasks.append(RenderTask(*key, int(row["current_viewpoint_id"]), source, motion))
    return sorted(tasks, key=lambda item: (item.scene_id, item.region, item.record_id))


def _output_path(root: Path, task: RenderTask) -> Path:
    return root / task.scene_id / task.region / f"{task.record_id}.npz"


def _valid(path: Path, task: RenderTask) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            rgb = np.asarray(archive["rgb"])
            mask = np.asarray(archive["available_view_mask"], dtype=bool)
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            return bool(
                str(archive["rgb_observation_version"].item()) == RGB_VERSION
                and int(archive["frame_index"].item()) == FRAME_INDEX
                and rgb.shape == (32, IMAGE_SIZE, IMAGE_SIZE, 3)
                and rgb.dtype == np.uint8
                and mask.shape == (32,)
                and np.array_equal(ids, np.arange(32))
                and mask.sum() == 1
                and bool(mask[task.current_viewpoint_id])
                and str(archive["scene_id"].item()) == task.scene_id
                and str(archive["region"].item()) == task.region
                and str(archive["record_id"].item()) == task.record_id
                and int(rgb[task.current_viewpoint_id].max()) > 0
            )
    except (KeyError, OSError, ValueError, TypeError):
        return False


def _write(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render(sim: Any, human: Any, converter: MotionConverter, task: RenderTask) -> dict[str, Any]:
    metadata = _load_skeleton_metadata(task.source_path)
    converted = converter.convert(_load_resampled_motion(task.motion, TARGET_FRAMES))
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    # Placement yaw and position are immutable fields of the canonical archive.
    yaw_deg = 0.0
    placement_path = task.source_path.parents[1] / "candidate_metadata" / "manifest.json"
    if placement_path.is_file():
        placement_payload = json.loads(placement_path.read_text(encoding="utf-8"))
        for item in placement_payload.get("placements_data", []):
            if str(item.get("placement_id")) == task.region:
                yaw_deg = float(item.get("yaw_deg", 0.0))
                break
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw_deg)
    base = np.asarray(metadata["placement_position"], dtype=np.float32)
    apply_humanoid_pose(
        human,
        joints[FRAME_INDEX],
        roots[FRAME_INDEX],
        base_position=base,
        scene_yaw_deg=yaw_deg,
        floor_y=float(base[1]),
        grounding_offset=float(offsets[FRAME_INDEX]),
    )
    view = int(task.current_viewpoint_id)
    positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
    rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
    _set_agent_state(sim.get_agent(0), positions[view], rotations[view])
    observations = sim.get_sensor_observations([0])
    image = np.asarray(observations[0]["color_0"])
    if image.ndim != 3 or image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE) or image.shape[2] < 3:
        raise ValueError(f"unexpected frame-0 RGB shape: {image.shape}")
    rgb = np.zeros((32, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    rgb[view] = np.asarray(image[:, :, :3], dtype=np.uint8)
    mask = np.zeros(32, dtype=bool)
    mask[view] = True
    return {
        "rgb_observation_version": np.asarray(RGB_VERSION),
        "rgb": rgb,
        "available_view_mask": mask,
        "viewpoint_ids": np.arange(32, dtype=np.int32),
        "scene_id": np.asarray(task.scene_id),
        "region": np.asarray(task.region),
        "record_id": np.asarray(task.record_id),
        "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
        "image_size": np.asarray(IMAGE_SIZE, dtype=np.int32),
        "current_viewpoint_id": np.asarray(view, dtype=np.int32),
        "viewpoint_agent_positions": positions.astype(np.float32),
        "viewpoint_rotations_wxyz": rotations.astype(np.float32),
        "yaw_deg": np.asarray(yaw_deg, dtype=np.float32),
    }


def _worker(worker_id: int, tasks: Sequence[RenderTask], output_root: Path, scene_root: Path, progress: Path) -> None:
    sim, human = _simulator(scene_root, tasks[0].scene_id)
    converter = MotionConverter(URDF_PATH)
    completed = 0
    try:
        for task in tasks:
            target = _output_path(output_root, task)
            if not _valid(target, task):
                _write(target, _render(sim, human, converter, task))
                if not _valid(target, task):
                    raise ValueError(f"frame-0 RGB validation failed: {target}")
            completed += 1
            if completed % 50 == 0:
                progress.parent.mkdir(parents=True, exist_ok=True)
                progress.write_text(json.dumps({"worker": worker_id, "completed": completed}), encoding="utf-8")
    finally:
        sim.close()
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(json.dumps({"worker": worker_id, "completed": completed}), encoding="utf-8")
    os._exit(0)


def build(data_root: Path, output_root: Path, scene_root: Path, splits: Sequence[str], workers: int) -> dict[str, Any]:
    tasks = _tasks(data_root, splits)
    output_root.mkdir(parents=True, exist_ok=True)
    started = __import__("time").monotonic()
    for scene_id in sorted({task.scene_id for task in tasks}):
        scene_tasks = [task for task in tasks if task.scene_id == scene_id]
        processes: list[mp.Process] = []
        for worker_id in range(min(workers, len(scene_tasks))):
            shard = scene_tasks[worker_id::workers]
            process = mp.get_context("spawn").Process(
                target=_worker,
                args=(worker_id, shard, output_root, scene_root, output_root / scene_id / f"worker_{worker_id}.json"),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"frame-0 RGB workers failed for {scene_id}: {[p.exitcode for p in processes]}")
        print(json.dumps({"scene_id": scene_id, "records": len(scene_tasks)}), flush=True)
    manifest = [
        {
            "scene_id": task.scene_id,
            "region": task.region,
            "record_id": task.record_id,
            "current_viewpoint_id": task.current_viewpoint_id,
            "relative_path": str(_output_path(output_root, task).relative_to(output_root)),
            "frame_index": FRAME_INDEX,
            "rgb_observation_version": RGB_VERSION,
        }
        for task in tasks
    ]
    (output_root / "manifest.jsonl").write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in manifest), encoding="utf-8")
    summary = {
        "version": RGB_VERSION,
        "splits": list(splits),
        "records": len(tasks),
        "unique_current_observations": len(tasks),
        "frame_index": FRAME_INDEX,
        "image_size": IMAGE_SIZE,
        "workers": workers,
        "future_candidate_rgb_rendered": False,
        "candidate_rgb_rendered": False,
        "test_used": False,
        "elapsed_seconds": __import__("time").monotonic() - started,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=get_data_root() / OUTPUT_REL)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    print(json.dumps(build(args.data_root.resolve(), args.output_root.resolve(), args.scene_root.resolve(), args.splits, args.workers), indent=2))


if __name__ == "__main__":
    main()
