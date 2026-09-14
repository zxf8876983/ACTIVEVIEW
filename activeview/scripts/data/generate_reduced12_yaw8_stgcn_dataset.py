#!/usr/bin/env python3
"""Generate a yaw-expanded reduced12 clean ST-GCN dataset.

The source record split is copied verbatim from the frozen reduced12 clean
protocol.  Each record is rendered eight times by rotating the Habitat
humanoid at render time; the RGB -> pose -> VideoPose3D -> normalization chain
is otherwise unchanged.  Runtime arrays and images are written outside Git.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.babel_clean_dataset_generator import (
    BabelCleanDatasetGenerator,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)

LOGGER = logging.getLogger(__name__)
YAW_DEGREES: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)


def expand_records(records: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Create exactly eight deterministic scene-yaw variants per source record."""
    expanded: List[Dict[str, Any]] = []
    for source in records:
        source_id = str(source["record_id"])
        for yaw in YAW_DEGREES:
            item = dict(source)
            item["source_record_id"] = source_id
            item["record_id"] = f"{source_id}__yaw{yaw:03d}"
            item["split"] = split
            item["scene_yaw_deg"] = int(yaw)
            item["yaw_variant"] = f"yaw{yaw:03d}"
            item["yaw_augmentation_protocol"] = "humanoid_scene_yaw_at_habitat_render"
            expanded.append(item)
    return expanded


class Yaw8Generator(BabelCleanDatasetGenerator):
    """Existing clean generator with only the humanoid scene yaw parameter changed."""

    def _render_rgb(  # type: ignore[override]
        self,
        sim: Any,
        human: Any,
        joints: np.ndarray,
        root_transforms: np.ndarray,
        record: Mapping[str, Any],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        import habitat_sim
        import quaternion

        # Preserve the exact clean-protocol camera sampling rule.  The scene
        # yaw is the only changed factor across the eight variants.
        rng = random.Random(self.seed + int(record.get("candidate_index", record.get("babel_sid", 0))))
        angle = np.radians(rng.uniform(-25.0, 25.0))
        distance = rng.uniform(2.4, 2.8)
        scene_yaw = float(record.get("scene_yaw_deg", 0.0))
        frames: List[np.ndarray] = []
        camera_to_world: List[np.ndarray] = []
        grounding_offsets, center_y_values = precompute_grounding_offsets(
            human, joints, root_transforms, scene_yaw_deg=scene_yaw
        )
        for frame_index, (q_joints, root_transform) in enumerate(zip(joints, root_transforms)):
            apply_humanoid_pose(
                human,
                q_joints,
                root_transform,
                base_position=(0.0, 0.0, 0.0),
                scene_yaw_deg=scene_yaw,
                grounding_offset=float(grounding_offsets[frame_index]),
            )
            center_y = float(center_y_values[frame_index] + grounding_offsets[frame_index])
            camera_position = np.array(
                [distance * np.sin(angle), self.camera_height, distance * np.cos(angle)],
                dtype=np.float32,
            )
            target = np.array([0.0, center_y, 0.0], dtype=np.float32)
            direction = target - camera_position
            direction /= np.linalg.norm(direction)
            yaw = np.arctan2(-direction[0], -direction[2])
            pitch = np.arcsin(direction[1])
            rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector(
                [pitch, 0.0, 0.0]
            )
            state = habitat_sim.AgentState()
            state.position = camera_position
            state.rotation = rotation
            sim.get_agent(0).set_state(state)
            observation = sim.get_sensor_observations()["color_sensor"]
            frames.append(np.asarray(observation[:, :, :3], dtype=np.uint8))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = quaternion.as_rotation_matrix(rotation).astype(np.float32)
            c2w[:3, 3] = camera_position
            camera_to_world.append(c2w)
        return frames, np.stack(camera_to_world, axis=0)

    def _process_record(self, sim: Any, human: Any, record: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        item = super()._process_record(sim, human, record)
        item.update(
            {
                "source_record_id": str(record.get("source_record_id", record["record_id"])),
                "scene_yaw_deg": int(record.get("scene_yaw_deg", 0)),
                "yaw_variant": str(record.get("yaw_variant", "yaw000")),
                "yaw_augmentation_protocol": "humanoid_scene_yaw_at_habitat_render",
            }
        )
        return item


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_manifests(source_root: Path, output_root: Path, splits: Iterable[str], workers: int) -> Dict[str, Any]:
    mapping = json.loads((source_root / "label_mapping.json").read_text(encoding="utf-8"))
    counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        source_path = source_root / f"{split}.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        expanded = expand_records(records, split)
        _write_json(output_root / f"{split}.json", expanded)
        counts[split] = {"source_records": len(records), "expanded_records": len(expanded)}
        if len(expanded) != 8 * len(records):
            raise RuntimeError(f"Unexpected expansion count for {split}")
    _write_json(output_root / "label_mapping.json", mapping)
    source_ids: Dict[str, set[str]] = {}
    all_counts = dict(counts)
    for source_split in ("train", "val"):
        source_path = source_root / f"{source_split}.json"
        if source_path.exists():
            source_ids[source_split] = {
                str(item["record_id"]) for item in json.loads(source_path.read_text(encoding="utf-8"))
            }
            all_counts.setdefault(source_split, {
                "source_records": len(source_ids[source_split]),
                "expanded_records": 8 * len(source_ids[source_split]),
            })
    overlap = sorted(source_ids.get("train", set()) & source_ids.get("val", set()))
    audit = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "yaw_degrees": list(YAW_DEGREES),
        "records_per_source": 8,
        "record_split_leakage": len(overlap),
        "overlap_source_record_ids": overlap,
        "policy_train_val_used": False,
        "policy_test_read": False,
        "counts": all_counts,
    }
    _write_json(output_root / "split_audit.json", audit)
    _write_json(output_root / "generation_config.json", {
        "yaw_degrees": list(YAW_DEGREES),
        "target_frames": 30,
        "image_size": 256,
        "camera_distance_range_m": [2.4, 2.8],
        "camera_angle_jitter_deg": [-25.0, 25.0],
        "camera_height_m": 1.2,
        "camera_hfov_deg": 50.0,
        "seed": 42,
        "workers": int(workers),
        "pose_backend": "ultralytics_yolo26n",
        "augmentation_location": "Habitat humanoid rendering before RGB perception",
        "perception_chain": "RGB -> Ultralytics YOLO26n-Pose -> VideoPose3D",
        "normalization": "camera_to_gravity + root_center + torso_scale + yaw_only; align_canonical=True",
        "source_record_split": "reduced12_no_kneel_clean_babel_diversity_v1/raw-train train/val",
        "policy_test_used": False,
    })
    return audit


def generate_dataset(
    *, source_root: Path, output_root: Path, splits: Sequence[str], image_size: int,
    target_frames: int, seed: int, device: str, max_source_records: Optional[int], workers: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_manifests(source_root, output_root, splits, workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    summaries: Dict[str, Any] = {}
    for split in splits:
        manifest = json.loads((output_root / f"{split}.json").read_text(encoding="utf-8"))
        if max_source_records is not None:
            manifest = manifest[: max(0, int(max_source_records)) * len(YAW_DEGREES)]
        # Each worker owns a Habitat simulator and a pose-estimation model.
        # Sharding by expanded records preserves equal work while avoiding
        # unsafe thread-level access to Habitat/OpenGL state.
        worker_root = output_root / "_parallel_workers" / split
        worker_root.mkdir(parents=True, exist_ok=True)
        worker_records = [manifest[i::workers] for i in range(workers)]
        worker_processes: List[subprocess.Popen[Any]] = []
        script_path = Path(__file__).resolve()
        for worker_id, records in enumerate(worker_records):
            shard_root = worker_root / f"worker_{worker_id:02d}"
            shard_root.mkdir(parents=True, exist_ok=True)
            _write_json(shard_root / f"{split}.json", records)
            shutil.copy2(output_root / "label_mapping.json", shard_root / "label_mapping.json")
            command = [
                sys.executable, str(script_path), "--worker-root", str(shard_root),
                "--worker-split", split, "--image-size", str(image_size),
                "--target-frames", str(target_frames), "--seed", str(seed), "--device", device,
            ]
            LOGGER.info("Starting Yaw8 %s worker %d/%d", split, worker_id + 1, workers)
            worker_processes.append(subprocess.Popen(command))
        return_codes = [process.wait() for process in worker_processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"Yaw8 {split} workers failed: {return_codes}")
        metadata: List[Dict[str, Any]] = []
        for worker_id in range(workers):
            shard_root = worker_root / f"worker_{worker_id:02d}"
            shard_metadata = json.loads((shard_root / f"{split}_metadata.json").read_text(encoding="utf-8"))
            for item in shard_metadata:
                source = shard_root / str(item["skeleton_path"])
                target = output_root / str(item["skeleton_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                metadata.append(dict(item))
        metadata.sort(key=lambda item: str(item["record_id"]))
        if len(metadata) != len(manifest) or len({str(item["record_id"]) for item in metadata}) != len(metadata):
            raise RuntimeError(f"Yaw8 {split} merge count/uniqueness mismatch")
        _write_json(output_root / f"{split}_metadata.json", metadata)
        arrays = [np.load(output_root / str(item["skeleton_path"]))["skeleton"] for item in metadata]
        data = np.stack(arrays, axis=0).astype(np.float32)
        np.save(output_root / f"{split}_data.npy", data)
        np.save(output_root / f"{split}_labels.npy", np.asarray([int(item["label_id"]) for item in metadata], dtype=np.int64))
        summaries[split] = {
            "split": split, "samples": len(metadata), "data_shape": list(data.shape),
            "skeleton_preprocessing": metadata[0].get("skeleton_preprocessing", "") if metadata else "",
            "rendering_protocol": metadata[0].get("rendering_protocol", "") if metadata else "",
            "coordinate_transform": metadata[0].get("coordinate_transform", "") if metadata else "",
            "perception_chain": "RGB -> Ultralytics YOLO26n-Pose -> VideoPose3D",
            "parallel_workers": workers,
            "per_yaw_counts": {
                str(yaw): sum(int(item.get("scene_yaw_deg", -1)) == yaw for item in metadata)
                for yaw in YAW_DEGREES
            },
        }
    previous_summary: Dict[str, Any] = {}
    summary_path = output_root / "generation_summary.json"
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated = dict(previous_summary.get("generated", {}))
    generated.update(summaries)
    source_counts = dict(previous_summary.get("source_counts", {}))
    source_counts.update(audit["counts"])
    expected_records = {
        split: int(value["source_records"]) * 8 for split, value in source_counts.items()
    }
    successful_records = {
        split: int(generated[split]["samples"]) for split in generated
        if "samples" in generated[split]
    }
    success_rate = {
        split: float(successful_records[split] / max(1, expected_records[split]))
        for split in successful_records if split in expected_records
    }
    per_yaw_failures = {}
    for yaw in YAW_DEGREES:
        expected_yaw = sum(int(value["source_records"]) for value in source_counts.values())
        actual_yaw = sum(
            int(generated[split].get("per_yaw_counts", {}).get(str(yaw), 0))
            for split in generated
        )
        per_yaw_failures[str(yaw)] = max(0, expected_yaw - actual_yaw)
    generation_summary = {
        "source_counts": source_counts,
        "generated": generated,
        "expected_records": expected_records,
        "successful_records": successful_records,
        "success_rate": success_rate,
        "per_yaw_failures": per_yaw_failures,
        "workers": workers,
        "test_used": False,
    }
    _write_json(output_root / "generation_summary.json", generation_summary)
    return generation_summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-train",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=data_root / "datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1",
    )
    parser.add_argument("--split", choices=("train", "val"), action="append", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-source-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-split", choices=("train", "val"), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.worker_root is not None and args.worker_split is not None:
        mapping = json.loads((args.worker_root / "label_mapping.json").read_text(encoding="utf-8"))
        worker_generator = Yaw8Generator(
            output_root=args.worker_root, image_size=args.image_size, target_frames=args.target_frames,
            seed=args.seed, device=args.device, label_to_id=mapping, pose_backend="ultralytics_yolo26n",
            yolo_weights=get_data_root() / "checkpoints/ultralytics/yolo26n-pose.pt",
            videopose_weights=get_data_root() / "checkpoints/videopose3d/pretrained_h36m_detectron_coco.bin",
        )
        worker_generator.generate_split(args.worker_split, args.worker_root / f"{args.worker_split}.json")
        return
    splits = tuple(args.split) if args.split else ("train", "val")
    summary = generate_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        splits=splits,
        image_size=args.image_size,
        target_frames=args.target_frames,
        seed=args.seed,
        device=args.device,
        max_source_records=args.max_source_records,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
