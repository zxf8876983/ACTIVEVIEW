#!/usr/bin/env python3
"""Render only selected and GT-best viewpoints for the 12 qualitative cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.data.motion.babel_clean_dataset_generator import MotionConverter, URDF_PATH, _load_resampled_motion, apply_humanoid_pose, precompute_grounding_offsets
from activeview.scripts.data.generate_hm3d_train_rgb_observations import IMAGE_SIZE, TARGET_FRAMES, _load_skeleton_metadata, _set_agent_state, _simulator

FRAME_IDS = (0, 15, 29)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _placement_yaw(archive_root: Path, scene_id: str, placement_id: str) -> float:
    path = archive_root / scene_id / "candidate_metadata" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("placements_data", []):
        if str(item.get("placement_id")) == placement_id:
            return float(item["yaw_deg"])
    raise KeyError(f"missing placement yaw for {scene_id}/{placement_id}")


def _effective_viewpoint(case: Mapping[str, Any], key: str) -> int:
    value = case.get(f"{key}_id")
    if value is not None:
        return int(value)
    record = case.get("figure_record", {})
    field = "selected_viewpoint_id" if key == "selected" else "oracle_viewpoint_id"
    if field not in record:
        raise KeyError(f"missing effective {key} viewpoint in case manifest")
    return int(record[field])


def _render_case(sim: Any, human: Any, converter: MotionConverter, case: Mapping[str, Any], motion: Mapping[str, Any], archive_root: Path, output_root: Path) -> int:
    scene_id, placement_id, record_id = str(case["scene_id"]), str(case["placement_id"]), str(case["record_id"])
    source_path = archive_root / scene_id / placement_id / f"{record_id}.npz"
    metadata = _load_skeleton_metadata(source_path)
    motion_data = _load_resampled_motion(motion, TARGET_FRAMES)
    converted = converter.convert(motion_data)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    yaw_deg = _placement_yaw(archive_root, scene_id, placement_id)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw_deg)
    base = np.asarray(metadata["placement_position"], dtype=np.float32)
    view_ids = np.asarray([_effective_viewpoint(case, "selected"), _effective_viewpoint(case, "oracle")], dtype=np.int32)
    positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
    rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
    rgb = np.empty((2, len(FRAME_IDS), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for frame_slot, frame_id in enumerate(FRAME_IDS):
        apply_humanoid_pose(human, joints[frame_id], roots[frame_id], base_position=base, scene_yaw_deg=yaw_deg, floor_y=float(base[1]), grounding_offset=float(offsets[frame_id]))
        for view_slot, viewpoint_id in enumerate(view_ids.tolist()):
            _set_agent_state(sim.get_agent(0), positions[viewpoint_id], rotations[viewpoint_id])
            observations = sim.get_sensor_observations([0])
            image = np.asarray(observations[0]["color_0"])[:, :, :3]
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8 or int(image.max()) <= 0 or float(image.mean()) <= 1.0:
                raise RuntimeError(f"invalid RGB case={case['case_id']} view={viewpoint_id} frame={frame_id}: shape={image.shape} dtype={image.dtype} max={int(image.max())} mean={float(image.mean()):.3f}")
            rgb[view_slot, frame_slot] = image
    target = output_root / f"{case['case_id']}.npz"
    temporary = target.with_suffix(f".npz.tmp.{os.getpid()}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, rgb=rgb, viewpoint_ids=view_ids, frame_ids=np.asarray(FRAME_IDS, dtype=np.int32), scene_id=np.asarray(scene_id), placement_id=np.asarray(placement_id), record_id=np.asarray(record_id), image_size=np.asarray(IMAGE_SIZE, dtype=np.int32), yaw_deg=np.asarray(yaw_deg, dtype=np.float32))
    os.replace(temporary, target)
    return int(rgb.shape[0] * rgb.shape[1])


def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    motions = {str(row["record_id"]): row for row in json.loads(args.motion_manifest.read_text(encoding="utf-8"))}
    if len(cases) != 12:
        raise ValueError(f"expected exactly 12 cases, got {len(cases)}")
    status: dict[str, Any] = {"status": "RENDERING", "launcher_python": None, "habitat_python": sys.executable, "habitat_import_ok": True, "magnum_import_ok": True, "cuda_available_in_habitat_env": False, "gpu_name": None, "renderer_initialized": False, "rendered_cases": 0, "rendered_images": 0, "targeted_cases": 12, "targeted_viewpoints_per_case": 2, "frame_ids": list(FRAME_IDS), "full_dataset_rgb_regenerated": False, "test_used": False, "training_used": False}
    import torch

    status["cuda_available_in_habitat_env"] = bool(torch.cuda.is_available())
    if status["cuda_available_in_habitat_env"]:
        status["gpu_name"] = torch.cuda.get_device_name(0)
    if not status["cuda_available_in_habitat_env"]:
        status.update({"status": "BLOCKED_EXTERNAL_RUNTIME", "reason": "external Habitat Python cannot access CUDA"})
        _write_json(args.runtime_status, status)
        raise RuntimeError(status["reason"])
    scenes = sorted({str(case["scene_id"]) for case in cases})
    converter = MotionConverter(URDF_PATH)
    try:
        for scene_id in scenes:
            sim, human = _simulator(args.scene_root, scene_id)
            status["renderer_initialized"] = True
            _write_json(args.runtime_status, status)
            try:
                for case in [item for item in cases if str(item["scene_id"]) == scene_id]:
                    record_id = str(case["record_id"])
                    if record_id not in motions:
                        raise KeyError(f"missing motion metadata for {record_id}")
                    images = _render_case(sim, human, converter, case, motions[record_id], args.archive_root, args.output)
                    status["rendered_cases"] = int(status["rendered_cases"]) + 1
                    status["rendered_images"] = int(status["rendered_images"]) + images
                    _write_json(args.runtime_status, status)
            finally:
                sim.close()
    except Exception as exc:
        status.update({"status": "RENDER_FAILED", "reason": repr(exc)})
        _write_json(args.runtime_status, status)
        raise
    status["status"] = "COMPLETED"
    _write_json(args.runtime_status, status)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--motion-manifest", type=Path, required=True)
    parser.add_argument("--runtime-status", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
