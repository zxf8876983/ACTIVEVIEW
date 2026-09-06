#!/usr/bin/env python3
"""文件用途：
    执行离线数据生成、划分或缓存构建入口。

主要输入：
    - 命令行参数与已有运行时数据。
主要输出：
    - 数据集、缓存或清单文件。
项目角色：
    - 属于 data 脚本入口，仅调用正式数据模块。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import habitat_sim
import magnum as mn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root

LOGGER = logging.getLogger(__name__)

SCENE_IDS: Sequence[str] = (
    "00006-HkseAnWCgqk",
    "00062-ACZZiU6BXLz",
    "00087-YY8rqV6L6rf",
    "00096-6HRFAUDqpTb",
    "00164-XfUxBGTFQQb",
    "00172-bB6nKqfsb1z",
    "00250-U3oQjwTuMX8",
    "00251-wsAYBFtQaL7",
    "00299-bdp1XNEdvmW",
    "00326-u9rPN5cHWBg",
    "00327-xgLmjqzoAzF",
    "00417-nGhNxKrgBPb",
    "00422-8wJuSPJ9FXG",
    "00444-sX9xad6ULKc",
    "00475-g7hUFVNac26",
    "00476-NtnvZSMK3en",
    "00487-erXNfWVjqZ8",
    "00534-DBBESbk4Y3k",
    "00592-CthA7sQNTPK",
    "00643-ggNAcMh8JPT",
    "00750-E1NrAhMoqvB",
)

REGION_LABELS: Mapping[str, Sequence[str]] = {
    "bedroom": ("bed",),
    "living_room": ("couch",),
    "kitchen": ("kitchen cabinet",),
    "dining_area": ("dining table", "dining chair"),
}

REGION_ACTIONS: Mapping[str, str] = {
    "bedroom": "lie",
    "living_room": "sit",
    "kitchen": "stand up",
    "dining_area": "sit",
}

FURNITURE_LABEL_TERMS: Sequence[str] = (
    "bed", "couch", "chair", "table", "cabinet", "wardrobe", "sofa",
    "desk", "dresser", "shelf", "refrigerator", "oven", "stove", "sink",
    "toilet", "bathtub", "bath", "fireplace", "nightstand", "countertop",
)


def _load_scene_list(path: Path | None) -> List[str]:
    if path is None:
        return list(SCENE_IDS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("scene_ids", payload.get("scenes", []))
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise ValueError("scene-list must be a JSON list of scene IDs")
    return list(payload)


def _load_furniture(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"Invalid furniture manifest: {path}")
    return [dict(item) for item in objects if isinstance(item, dict)]


def collect_eligible_furniture(semantic_objects: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep common furniture anchors with finite semantic centers."""
    eligible: List[Dict[str, Any]] = []
    for item in semantic_objects:
        label = str(item.get("label", "")).strip()
        center = np.asarray(item.get("center_xyz", []), dtype=np.float32)
        if not label or center.shape != (3,) or not np.isfinite(center).all():
            continue
        if not any(term in label.lower() for term in FURNITURE_LABEL_TERMS):
            continue
        eligible.append(dict(item))
    return eligible


def _semantic_object_center(obj: Mapping[str, Any]) -> np.ndarray:
    """Convert semantic-GLB x/y/z to Habitat x/y/z coordinates."""
    center = np.asarray(obj["center_xyz"], dtype=np.float32)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Furniture center_xyz must have shape (3,) and finite values")
    return np.asarray([center[0], center[2], -center[1]], dtype=np.float32)


def make_diverse_anchor_order(
    furniture: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> List[Mapping[str, Any]]:
    """Interleave shuffled furniture categories before sampling placements."""
    by_category: Dict[str, List[Mapping[str, Any]]] = {}
    for item in furniture:
        category = str(item.get("label", "")).strip().lower()
        by_category.setdefault(category, []).append(item)
    categories = list(by_category)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(by_category[category])
    ordered: List[Mapping[str, Any]] = []
    while categories:
        remaining: List[str] = []
        for category in categories:
            bucket = by_category[category]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                remaining.append(category)
        categories = remaining
    return ordered


def sample_scene_placements(
    sim: habitat_sim.Simulator,
    semantic_objects: Sequence[Mapping[str, Any]],
    num_placements: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample category-diverse, navigable furniture-near human placements."""
    if num_placements < 1:
        raise ValueError("num_placements must be positive")
    rng = np.random.default_rng(seed)
    furniture = collect_eligible_furniture(semantic_objects)
    anchors = make_diverse_anchor_order(furniture, rng)
    placements: List[Dict[str, Any]] = []
    for obj in anchors:
        center = _semantic_object_center(obj)
        for _ in range(30):
            radius = float(rng.uniform(0.5, 1.2))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            raw = center + np.asarray(
                [radius * math.cos(angle), 0.0, radius * math.sin(angle)],
                dtype=np.float32,
            )
            snapped = np.asarray(sim.pathfinder.snap_point(raw), dtype=np.float32)
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                continue
            if float(np.linalg.norm(snapped[[0, 2]] - raw[[0, 2]])) > 0.5:
                continue
            if not sim.pathfinder.is_navigable(snapped):
                continue
            try:
                clearance = float(sim.pathfinder.distance_to_closest_obstacle(snapped))
            except RuntimeError:
                continue
            if clearance < 0.28:
                continue
            if any(float(np.linalg.norm(snapped - other["position"])) < 1.0 for other in placements):
                continue
            placements.append(
                {
                    "placement_id": f"p{len(placements):02d}",
                    "position": snapped.astype(np.float32),
                    "anchor_label": str(obj["label"]),
                    "anchor_object_id": int(obj.get("instance_index", len(placements))),
                    "anchor_center_habitat_xyz": center.astype(np.float32),
                    "clearance_m": clearance,
                    "radius_m": radius,
                    "angle_rad": angle,
                    "yaw_deg": float(rng.choice(np.arange(0, 360, 45))),
                }
            )
            break
        if len(placements) == num_placements:
            break
    if len(placements) != num_placements:
        raise RuntimeError(f"Only found {len(placements)}/{num_placements} valid placements")
    return placements


def _find_object(objects: Iterable[Mapping[str, Any]], labels: Sequence[str]) -> Mapping[str, Any]:
    for label in labels:
        candidates = [item for item in objects if str(item.get("label", "")).lower() == label]
        if candidates:
            candidates.sort(key=lambda item: (int(item.get("instance_index", 0)), str(item.get("label", ""))))
            return candidates[0]
    raise ValueError(f"No furniture object with labels {labels}")


def _make_sim(scene_root: Path, scene_id: str) -> habitat_sim.Simulator:
    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb"))
    nav = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.enable_physics = False
    agent = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    sim.pathfinder.load_nav_mesh(str(nav))
    return sim


def _choose_placement(sim: habitat_sim.Simulator, obj: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(obj["center_xyz"], dtype=np.float32)
    if center.shape != (3,):
        raise ValueError("Furniture center_xyz must have shape (3,)")
    # Semantic GLB uses x/y/z; the Habitat scene uses x/z/-y, matching the
    # existing v11.5 semantic-region visualizer.
    target = np.asarray([center[0], center[2], -center[1]], dtype=np.float32)
    candidates: List[tuple[float, float, np.ndarray]] = []
    for radius in np.linspace(0.35, 1.4, 8):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            raw = target + np.asarray(
                [radius * math.cos(angle), 0.0, radius * math.sin(angle)], dtype=np.float32
            )
            snapped = np.asarray(sim.pathfinder.snap_point(raw), dtype=np.float32)
            if not np.isfinite(snapped).all():
                continue
            try:
                clearance = float(sim.pathfinder.distance_to_closest_obstacle(snapped))
            except RuntimeError:
                clearance = 0.0
            candidates.append((clearance, float(np.linalg.norm(snapped - target)), snapped))
    if not candidates:
        raise RuntimeError("No navigable placement candidate around furniture")
    valid = [item for item in candidates if item[0] >= 0.28]
    _, _, placement = min(valid or candidates, key=lambda item: item[1])
    # The candidate-metadata simulator intentionally has no Bullet physics;
    # its navmesh height is the floor reference used by the RGB generator.
    # Floor ray-casting is therefore deferred to the rendering simulator.
    return placement.astype(np.float32)


def _write_region_manifest(
    *, scene_root: Path, scene_id: str, furniture_path: Path, output_path: Path
) -> Path:
    objects = _load_furniture(furniture_path)
    sim = _make_sim(scene_root, scene_id)
    try:
        regions: Dict[str, Dict[str, Any]] = {}
        for region, labels in REGION_LABELS.items():
            obj = _find_object(objects, labels)
            placement = _choose_placement(sim, obj)
            regions[region] = {
                "furniture": str(obj["label"]),
                "instance_index": int(obj.get("instance_index", 0)),
                "action": REGION_ACTIONS[region],
                "placement_habitat_xyz": placement.tolist(),
                "source_center_xyz": list(obj["center_xyz"]),
            }
    finally:
        sim.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"version": "hm3d-train-four-region-v1", "scene_id": scene_id, "regions": regions},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def _run_topdown(
    *, scene_id: str, semantic_root: Path, furniture_dir: Path, topdown_script: Path
) -> Path:
    furniture_path = furniture_dir / "furniture_positions.json"
    if furniture_path.exists():
        return furniture_path
    semantic_dir = semantic_root / scene_id
    semantic_glb = next(semantic_dir.glob("*.semantic.glb"))
    semantic_txt = next(semantic_dir.glob("*.semantic.txt"))
    furniture_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(topdown_script),
        "--semantic-glb",
        str(semantic_glb),
        "--semantic-txt",
        str(semantic_txt),
        "--output-dir",
        str(furniture_dir),
    ]
    LOGGER.info("[%s] building furniture positions", scene_id)
    subprocess.run(command, check=True)
    if not furniture_path.exists():
        raise FileNotFoundError(f"Top-down script did not produce {furniture_path}")
    return furniture_path


def _run_candidate_metadata(
    *,
    scene_root: Path,
    scene_id: str,
    region_manifest: Path | None,
    placements_file: Path | None,
    candidate_dir: Path,
    candidate_script: Path,
) -> None:
    manifest_path = candidate_dir / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            common_valid = (
                existing.get("rotation_reference") == "exact_offline_render_state"
                and float(existing.get("sensor_height_m", -1.0)) == 1.1
                and float(existing.get("target_height_m", -1.0)) == 0.85
            )
            if placements_file is not None:
                placement_cache_valid = (
                    existing.get("version") == "furniture-placement-v2"
                    and int(existing.get("placements", -1)) == 8
                    and existing.get("source_placements_file") == str(placements_file.resolve())
                )
            else:
                placement_cache_valid = existing.get("version") == "semantic-region-v2"
            if common_valid and placement_cache_valid:
                return
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    command = [
        sys.executable,
        str(candidate_script),
        "--scene-root",
        str(scene_root),
        "--scene-id",
        scene_id,
        "--output-dir",
        str(candidate_dir),
        "--num-views",
        "32",
    ]
    if placements_file is not None:
        command.extend(["--placements-file", str(placements_file)])
    elif region_manifest is not None:
        command.extend(["--region-manifest", str(region_manifest)])
    else:
        raise ValueError("one of region_manifest or placements_file is required")
    LOGGER.info("[%s] building candidate metadata", scene_id)
    subprocess.run(command, check=True)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate script did not produce {manifest_path}")


def _run_view_generation(
    *,
    scene_root: Path,
    scene_id: str,
    candidate_dir: Path,
    manifest: Path,
    output_dir: Path,
    generator_script: Path,
    workers: int,
    image_size: int,
    target_frames: int,
    device: str,
    yolo_weights: Path,
    max_records: int | None,
) -> None:
    expected_records = len(json.loads(manifest.read_text(encoding="utf-8")))
    if max_records is not None:
        expected_records = min(expected_records, max_records)
    candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    placement_count = int(candidate_manifest.get("placements", len(candidate_manifest.get("placements_data", []))))
    if placement_count < 1:
        raise ValueError(f"candidate manifest has no placements: {candidate_dir}")
    expected_items = expected_records * placement_count
    final_manifest = output_dir / "manifest.json"
    if final_manifest.exists():
        try:
            existing = json.loads(final_manifest.read_text(encoding="utf-8"))
            if int(existing.get("records", -1)) == expected_records and int(existing.get("samples", -1)) == expected_items * 32:
                LOGGER.info("[%s] complete output exists; skipping", scene_id)
                return
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    command = [
        sys.executable,
        str(generator_script),
        "--candidate-dir",
        str(candidate_dir),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--scene-root",
        str(scene_root),
        "--scene-id",
        scene_id,
        "--workers",
        str(workers),
        "--image-size",
        str(image_size),
        "--target-frames",
        str(target_frames),
        "--device",
        device,
        "--yolo-weights",
        str(yolo_weights),
    ]
    if max_records is not None:
        command.extend(["--max-records", str(max_records)])
    LOGGER.info("[%s] generating %d records x %d placements x 32 views with %d workers", scene_id, expected_records, placement_count, workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, default=None, help="Optional JSON list of scene IDs")
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--semantic-root", type=Path, default=get_habitat_data_root() / "hm3d-train-semantic-annots")
    parser.add_argument("--manifest", type=Path, default=data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-val/official_val.json")
    parser.add_argument("--output-root", type=Path, default=data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1")
    parser.add_argument("--topdown-root", type=Path, default=data_root / "visualizations/hm3d_train_semantic_topdown")
    parser.add_argument("--placement-root", type=Path, default=data_root / "datasets/offline/hm3d-train_reduced14_kneel/placement_sampling_v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-weights", type=Path, default=data_root / "checkpoints/ultralytics/yolo26n-pose.pt")
    parser.add_argument("--max-records", type=int, default=None, help="Bounded smoke limit per scene")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    scenes = _load_scene_list(args.scene_list)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("motion manifest must be a JSON list")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reduced14_mode = "reduced14_kneel" in args.manifest.as_posix()
    missing_placements = [
        scene_id for scene_id in scenes
        if not (args.placement_root / scene_id / "placements.json").exists()
    ]
    if reduced14_mode and missing_placements:
        raise FileNotFoundError(
            "reduced14 eight-placement mode requires placements.json for every scene; "
            f"missing: {', '.join(missing_placements)}"
        )
    placement_mode = reduced14_mode
    scene_list_path = args.output_root / "scene_selection.json"
    scene_selection = {"scene_ids": scenes}
    if placement_mode:
        scene_selection.update({"protocol": "furniture-placement-v2", "placements_per_scene": 8})
    else:
        scene_selection["regions"] = list(REGION_LABELS)
    scene_list_path.write_text(json.dumps(scene_selection, indent=2), encoding="utf-8")
    topdown_script = Path(__file__).with_name("visualize_hm3d_semantic_topdown.py")
    candidate_script = Path(__file__).with_name("generate_semantic_region_candidate_metadata.py")
    generator_script = Path(__file__).with_name("generate_semantic_region_offline_views.py")
    summary_path = args.output_root / "dataset_summary.json"
    summary: Dict[str, Any] = {
        "version": "hm3d-train-eight-placement-reduced14-v1" if placement_mode else "hm3d-train-four-region-offline-v1",
        "scene_set": "hm3d-train",
        "scenes_requested": scenes,
        "placements_per_scene": 8 if placement_mode else None,
        "regions": None if placement_mode else list(REGION_LABELS),
        "records_per_scene": min(len(records), args.max_records) if args.max_records is not None else len(records),
        "views_per_record": 32,
        "workers_per_scene": args.workers,
        "target_frames": args.target_frames,
        "image_size": args.image_size,
        "rgb_saved": False,
        "depth_saved": False,
        "scene_status": [],
    }
    for scene_id in scenes:
        scene_out = args.output_root / scene_id
        status: Dict[str, Any] = {"scene_id": scene_id, "status": "started", "output": str(scene_out)}
        try:
            placements_file = args.placement_root / scene_id / "placements.json"
            if placements_file.exists():
                region_manifest = None
            else:
                furniture = _run_topdown(
                    scene_id=scene_id,
                    semantic_root=args.semantic_root,
                    furniture_dir=args.topdown_root / scene_id,
                    topdown_script=topdown_script,
                )
                region_manifest = _write_region_manifest(
                    scene_root=args.scene_root,
                    scene_id=scene_id,
                    furniture_path=furniture,
                    output_path=scene_out / "region_placement_manifest.json",
                )
                placements_file = None
            candidate_dir = scene_out / "candidate_metadata"
            _run_candidate_metadata(
                scene_root=args.scene_root,
                scene_id=scene_id,
                region_manifest=region_manifest,
                placements_file=placements_file,
                candidate_dir=candidate_dir,
                candidate_script=candidate_script,
            )
            _run_view_generation(
                scene_root=args.scene_root,
                scene_id=scene_id,
                candidate_dir=candidate_dir,
                manifest=args.manifest,
                output_dir=scene_out,
                generator_script=generator_script,
                workers=args.workers,
                image_size=args.image_size,
                target_frames=args.target_frames,
                device=args.device,
                yolo_weights=args.yolo_weights,
                max_records=args.max_records,
            )
            final = json.loads((scene_out / "manifest.json").read_text(encoding="utf-8"))
            status.update({"status": "complete", "records": final.get("records"), "samples": final.get("samples")})
        except Exception as exc:  # noqa: BLE001 - preserve per-scene progress in long jobs
            LOGGER.exception("[%s] failed", scene_id)
            status.update({"status": "failed", "error": repr(exc)})
        summary["scene_status"] = [item for item in summary["scene_status"] if item.get("scene_id") != scene_id]
        summary["scene_status"].append(status)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if status["status"] == "failed":
            LOGGER.error("[%s] failed; continuing with remaining scenes", scene_id)
    completed = sum(item.get("status") == "complete" for item in summary["scene_status"])
    LOGGER.info("Completed %d/%d scenes; summary=%s", completed, len(scenes), summary_path)
    if completed != len(scenes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
