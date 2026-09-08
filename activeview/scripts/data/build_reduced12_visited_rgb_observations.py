#!/usr/bin/env python3
"""Render only reduced12 Train/Val Stage-D visited s0/s1 RGB observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.scripts.data.build_visited_rgb_observations import (
    _tasks,
    _worker,
)


def build(
    *,
    data_root: Path,
    workers: int,
    splits: Sequence[str] = ("train", "val"),
    scene_id: str | None = None,
) -> dict[str, object]:
    if any(split not in {"train", "val"} for split in splits):
        raise ValueError("reduced12 RGB generation is Train/Val-only")
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    source_root = data_root / "datasets/offline/habitat-train/00006-00087"
    manifest_dir = data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val"
    output_root = data_root / "datasets/rgb_reduced12_eight_placement_v1/visited_s0_s1"
    scene_root = get_habitat_data_root() / "hm3d-train"
    tasks = _tasks(
        policy_root / "stage_d",
        policy_root / "stage_a",
        source_root,
        manifest_dir,
        tuple(splits),
    )
    if scene_id is not None:
        tasks = [task for task in tasks if task.scene_id == scene_id]
        if not tasks:
            raise ValueError(f"unknown or empty reduced12 scene: {scene_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = sorted({task.scene_id for task in tasks})
    for scene_id in scene_ids:
        scene_tasks = [task for task in tasks if task.scene_id == scene_id]
        processes = []
        for worker_id in range(workers):
            shard = scene_tasks[worker_id::workers]
            if not shard:
                continue
            progress = output_root / scene_id / f"worker_{worker_id}.json"
            import multiprocessing as mp

            process = mp.get_context("spawn").Process(
                target=_worker,
                args=(worker_id, shard, output_root, scene_root, progress),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(
                f"RGB workers failed for {scene_id}: "
                f"{[process.exitcode for process in processes]}"
            )
        print(json.dumps({"scene_id": scene_id, "records": len(scene_tasks)}), flush=True)
    summary: dict[str, object] = {
        "version": "activeview-reduced12-visited-rgb-v1",
        "protocol": "reduced12-eight-placement-v1",
        "splits": list(splits),
        "records": len(tasks),
        "rendered_viewpoints": sum(len(task.viewpoint_ids) for task in tasks),
        "frame_index": 15,
        "image_size": 256,
        "workers": workers,
        "future_candidate_rgb_rendered": False,
        "test_used": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scene-id", default=None)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    print(json.dumps(build(
        data_root=args.data_root.resolve(),
        workers=args.workers,
        scene_id=args.scene_id,
    ), indent=2))


if __name__ == "__main__":
    main()
