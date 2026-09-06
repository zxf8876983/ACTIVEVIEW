#!/usr/bin/env python3
"""Render frame-15 RGB only for Stage-D visited s0/s1 observations."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
import sys
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
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.data.generate_hm3d_train_rgb_observations import (
    FRAME_INDEX,
    HFOV_DEG,
    IMAGE_SIZE,
    SENSOR_HEIGHT_M,
    TARGET_FRAMES,
    VERSION,
    _load_skeleton_metadata,
    _set_agent_state,
    _simulator,
)


@dataclass(frozen=True)
class RenderTask:
    scene_id: str
    placement_id: str
    record_id: str
    source_path: Path
    motion: Mapping[str, Any]
    viewpoint_ids: tuple[int, ...]
    yaw_deg: float


def _load_motion_lookup(manifest_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        rows = json.loads((manifest_dir / f"{split}.json").read_text(encoding="utf-8"))
        for row in rows:
            record_id = str(row["record_id"])
            if record_id in output:
                raise ValueError(f"duplicate motion record: {record_id}")
            output[record_id] = dict(row)
    return output


def _placement_yaws(source_root: Path, scene_id: str) -> dict[str, float]:
    path = source_root / scene_id / "candidate_metadata" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "furniture-placement-v2":
        raise ValueError(f"expected furniture-placement-v2: {path}")
    result = {
        str(row["placement_id"]): float(row["yaw_deg"])
        for row in payload["placements_data"]
    }
    if sorted(result) != [f"p{index:02d}" for index in range(8)]:
        raise ValueError(f"invalid placement IDs: {path}")
    return result


def _tasks(
    stage_d_root: Path,
    stage_a_root: Path,
    source_root: Path,
    manifest_dir: Path,
    splits: Sequence[str],
) -> list[RenderTask]:
    motions = _load_motion_lookup(manifest_dir)
    requested: dict[tuple[str, str, str], set[int]] = {}
    sources: dict[tuple[str, str, str], Path] = {}
    for split in splits:
        episodes = {
            str(row["episode_id"]): row
            for row in load_jsonl(stage_a_root / "episodes" / f"{split}_episodes.jsonl")
        }
        for row in load_jsonl(stage_d_root / "features" / f"{split}.jsonl"):
            episode = episodes[str(row["episode_id"])]
            key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
            requested.setdefault(key, set()).update(
                (int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]))
            )
            source = Path(episode["current_view"]["skeleton_source_path"])
            previous = sources.setdefault(key, source)
            if previous.resolve() != source.resolve():
                raise ValueError(f"source mismatch for {key}")
    yaws = {scene: _placement_yaws(source_root, scene) for scene, _, _ in requested}
    output: list[RenderTask] = []
    for (scene, placement, record), view_ids in sorted(requested.items()):
        if record not in motions:
            raise KeyError(f"missing motion metadata for {record}")
        source = sources[(scene, placement, record)]
        expected = source_root / scene / placement / f"{record}.npz"
        if source.resolve() != expected.resolve():
            raise ValueError(f"unexpected source path for {(scene, placement, record)}: {source}")
        output.append(RenderTask(scene, placement, record, source, motions[record], tuple(sorted(view_ids)), yaws[scene][placement]))
    return output


def _valid_output(path: Path, task: RenderTask) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            rgb = np.asarray(archive["rgb"])
            mask = np.asarray(archive["available_view_mask"], dtype=bool)
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            return bool(
                rgb.shape == (32, IMAGE_SIZE, IMAGE_SIZE, 3)
                and rgb.dtype == np.uint8
                and mask.shape == (32,)
                and np.array_equal(ids, np.arange(32))
                and set(np.flatnonzero(mask).tolist()) == set(task.viewpoint_ids)
                and str(archive["scene_id"].item()) == task.scene_id
                and str(archive["region"].item()) == task.placement_id
                and str(archive["record_id"].item()) == task.record_id
                and int(archive["frame_index"].item()) == FRAME_INDEX
                and str(archive["rgb_observation_version"].item()) == VERSION
                and all(int(rgb[view].max()) > 0 for view in task.viewpoint_ids)
            )
    except (KeyError, OSError, ValueError):
        return False


def _write_atomic(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render(sim: Any, human: Any, converter: MotionConverter, task: RenderTask) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = _load_skeleton_metadata(task.source_path)
    motion = _load_resampled_motion(task.motion, TARGET_FRAMES)
    converted = converter.convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=task.yaw_deg)
    base = np.asarray(metadata["placement_position"], dtype=np.float32)
    apply_humanoid_pose(
        human,
        joints[FRAME_INDEX],
        roots[FRAME_INDEX],
        base_position=base,
        scene_yaw_deg=task.yaw_deg,
        floor_y=float(base[1]),
        grounding_offset=float(offsets[FRAME_INDEX]),
    )
    rgb = np.zeros((32, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
    rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
    for start in range(0, len(task.viewpoint_ids), 4):
        group = task.viewpoint_ids[start : start + 4]
        for agent_index, viewpoint_id in enumerate(group):
            _set_agent_state(sim.get_agent(agent_index), positions[viewpoint_id], rotations[viewpoint_id])
        observations = sim.get_sensor_observations(list(range(len(group))))
        for agent_index, viewpoint_id in enumerate(group):
            image = np.asarray(observations[agent_index][f"color_{agent_index}"])
            rgb[viewpoint_id] = np.asarray(image[:, :, :3], dtype=np.uint8)
    return rgb, metadata


def _worker(worker_id: int, tasks: Sequence[RenderTask], output_root: Path, scene_root: Path, progress: Path) -> None:
    sim, human = _simulator(scene_root, tasks[0].scene_id)
    converter = MotionConverter(URDF_PATH)
    completed = 0
    try:
        for task in tasks:
            target = output_root / task.scene_id / task.placement_id / f"{task.record_id}.npz"
            if not _valid_output(target, task):
                rgb, metadata = _render(sim, human, converter, task)
                mask = np.zeros(32, dtype=bool)
                mask[np.asarray(task.viewpoint_ids, dtype=np.int64)] = True
                _write_atomic(target, {
                    "rgb_observation_version": np.asarray(VERSION),
                    "rgb": rgb,
                    "available_view_mask": mask,
                    "viewpoint_ids": np.arange(32, dtype=np.int32),
                    "scene_id": np.asarray(task.scene_id),
                    "region": np.asarray(task.placement_id),
                    "placement_id": np.asarray(task.placement_id),
                    "record_id": np.asarray(task.record_id),
                    "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
                    "image_size": np.asarray(IMAGE_SIZE, dtype=np.int32),
                    "yaw_deg": np.asarray(task.yaw_deg, dtype=np.float32),
                    "viewpoint_agent_positions": metadata["viewpoint_agent_positions"],
                    "viewpoint_rotations_wxyz": metadata["viewpoint_rotations_wxyz"],
                })
                if not _valid_output(target, task):
                    raise ValueError(f"invalid rendered cache: {target}")
            completed += 1
            if completed % 20 == 0:
                progress.write_text(json.dumps({"worker": worker_id, "completed": completed}), encoding="utf-8")
    finally:
        sim.close()
    progress.write_text(json.dumps({"worker": worker_id, "completed": completed}), encoding="utf-8")


def build(
    *, stage_d_root: Path, stage_a_root: Path, source_root: Path, manifest_dir: Path,
    output_root: Path, scene_root: Path, splits: Sequence[str], workers: int,
) -> dict[str, Any]:
    tasks = _tasks(stage_d_root, stage_a_root, source_root, manifest_dir, splits)
    output_root.mkdir(parents=True, exist_ok=True)
    for scene_id in sorted({task.scene_id for task in tasks}):
        scene_tasks = [task for task in tasks if task.scene_id == scene_id]
        processes: list[mp.Process] = []
        for worker_id in range(workers):
            shard = scene_tasks[worker_id::workers]
            if not shard:
                continue
            process = mp.get_context("spawn").Process(
                target=_worker,
                args=(worker_id, shard, output_root, scene_root, output_root / scene_id / f"worker_{worker_id}.json"),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"RGB workers failed for {scene_id}: {[process.exitcode for process in processes]}")
        print(json.dumps({"scene_id": scene_id, "records": len(scene_tasks)}), flush=True)
    view_count = sum(len(task.viewpoint_ids) for task in tasks)
    summary = {
        "version": "activeview-reduced14-visited-rgb-v1",
        "splits": list(splits),
        "records": len(tasks),
        "rendered_viewpoints": view_count,
        "frame_index": FRAME_INDEX,
        "image_size": IMAGE_SIZE,
        "sensor_height_m": SENSOR_HEIGHT_M,
        "hfov_deg": HFOV_DEG,
        "workers": workers,
        "future_candidate_rgb_rendered": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-d-root", type=Path, default=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/stage_d")
    parser.add_argument("--stage-a-root", type=Path, default=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1")
    parser.add_argument("--source-root", type=Path, default=data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1")
    parser.add_argument("--manifest-dir", type=Path, default=data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-val")
    parser.add_argument("--output-root", type=Path, default=data_root / "datasets/rgb_reduced14_kneel_eight_placement_v1/visited_s0_s1")
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    print(json.dumps(build(
        stage_d_root=args.stage_d_root, stage_a_root=args.stage_a_root,
        source_root=args.source_root, manifest_dir=args.manifest_dir,
        output_root=args.output_root, scene_root=args.scene_root,
        splits=args.splits, workers=args.workers,
    ), indent=2))


if __name__ == "__main__":
    main()
