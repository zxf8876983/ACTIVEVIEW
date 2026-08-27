#!/usr/bin/env python3
"""Shard selected16 pure-color Habitat generation across independent workers.

Each worker owns its own Habitat simulator and CUDA estimator and writes to an
isolated directory.  The parent process merges worker metadata and tensors
only after every worker exits successfully.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.core.paths import get_data_root

LOGGER = logging.getLogger(__name__)


def _write_shards(data_root: Path, split: str, worker_root: Path, workers: int, max_records: Optional[int]) -> List[Path]:
    records = json.loads((data_root / f"{split}.json").read_text(encoding="utf-8"))
    if max_records is not None:
        records = records[: max(0, int(max_records))]
    shards: List[Path] = []
    for worker_id in range(workers):
        shard_root = worker_root / f"worker_{worker_id:02d}"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_records = records[worker_id::workers]
        shard_path = shard_root / f"{split}.json"
        shard_path.write_text(json.dumps(shard_records, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(data_root / "label_mapping.json", shard_root / "label_mapping.json")
        shards.append(shard_root)
    return shards


def _merge_split(data_root: Path, split: str, shard_roots: List[Path], target_frames: int, pose_backend: str) -> Dict[str, Any]:
    metadata: List[Dict[str, Any]] = []
    final_skeleton_root = data_root / "estimated_skeletons" / split
    final_skeleton_root.mkdir(parents=True, exist_ok=True)
    for shard_root in shard_roots:
        metadata_path = shard_root / f"{split}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Worker did not produce metadata: {metadata_path}")
        worker_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in worker_metadata:
            source = shard_root / str(item["skeleton_path"])
            target = data_root / str(item["skeleton_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            metadata.append(dict(item))
    metadata.sort(key=lambda item: str(item["record_id"]))
    if len({str(item["record_id"]) for item in metadata}) != len(metadata):
        raise ValueError(f"Duplicate record IDs while merging {split}")
    (data_root / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    tensors = [np.load(data_root / str(item["skeleton_path"]))["skeleton"] for item in metadata]
    data = np.stack(tensors, axis=0).astype(np.float32) if tensors else np.empty((0, 3, target_frames, 17, 1), dtype=np.float32)
    labels = np.asarray([int(item["label_id"]) for item in metadata], dtype=np.int64)
    np.save(data_root / f"{split}_data.npy", data)
    np.save(data_root / f"{split}_labels.npy", labels)
    summary = {
        "split": split,
        "samples": len(metadata),
        "data_shape": list(data.shape),
        "skeleton_preprocessing": metadata[0].get("skeleton_preprocessing", "unrecorded") if metadata else "unrecorded",
        "rendering_protocol": metadata[0].get("rendering_protocol", "unrecorded") if metadata else "unrecorded",
        "coordinate_transform": metadata[0].get("coordinate_transform", "unrecorded") if metadata else "unrecorded",
        "perception_chain": f"RGB -> {pose_backend} -> VideoPose3D",
        "parallel_workers": len(shard_roots),
    }
    (data_root / f"{split}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def generate(*, data_root: Path, split: str, workers: int, image_size: int, target_frames: int, device: str, pose_backend: str, yolo_weights: Path, max_records: Optional[int] = None) -> Dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    worker_root = data_root / "_parallel_workers" / split
    worker_root.mkdir(parents=True, exist_ok=True)
    shard_roots = _write_shards(data_root, split, worker_root, workers, max_records)
    script = Path(__file__).with_name("generate_selected16_habitat_dataset.py")
    processes: List[subprocess.Popen[str]] = []
    for shard_root in shard_roots:
        command = [
            sys.executable, str(script), "--data-root", str(shard_root), "--split", split,
            "--image-size", str(image_size), "--target-frames", str(target_frames), "--device", device,
            "--pose-backend", pose_backend, "--yolo-weights", str(yolo_weights),
        ]
        LOGGER.info("Starting worker: %s", " ".join(command))
        processes.append(subprocess.Popen(command))
    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"Parallel workers failed with return codes {return_codes}")
    return _merge_split(data_root, split, shard_roots, target_frames, pose_backend)


def main() -> None:
    runtime_root = get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=runtime_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pose-backend", choices=("ultralytics_yolo26n",), default="ultralytics_yolo26n")
    parser.add_argument("--yolo-weights", type=Path, default=runtime_root / "checkpoints/ultralytics/yolo26n-pose.pt")
    parser.add_argument("--max-records", type=int, default=None, help="Bounded smoke limit before full generation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = generate(data_root=args.data_root, split=args.split, workers=args.workers, image_size=args.image_size, target_frames=args.target_frames, device=args.device, pose_backend=args.pose_backend, yolo_weights=args.yolo_weights, max_records=args.max_records)
    LOGGER.info("Merged result: %s", summary)


if __name__ == "__main__":
    main()
