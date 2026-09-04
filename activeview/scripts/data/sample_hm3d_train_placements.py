#!/usr/bin/env python3
"""Sample furniture-anchored human placements for frozen HM3D scenes.

This command writes placement coordinates only.  It intentionally does not
generate skeleton, RGB, depth, perception, or policy artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.scripts.data.generate_hm3d_train_four_region_offline import (
    SCENE_IDS,
    _load_furniture,
    _load_scene_list,
    _make_sim,
    sample_scene_placements,
)

LOGGER = logging.getLogger("activeview.placement_sampling")
VERSION = "hm3d-placement-sampling-v2"


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar/array values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _raw_val_count(path: Path) -> int:
    """Return the number of records in the selected Official-Val manifest."""
    if not path.exists():
        raise FileNotFoundError(f"raw-val manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        for key in ("records", "episodes", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    raise ValueError(f"Unsupported raw-val manifest schema: {path}")


def _validate_placements(placements: Sequence[Mapping[str, Any]], expected: int) -> None:
    """Check the placement invariants before an artifact is published."""
    if len(placements) != expected:
        raise ValueError(f"expected {expected} placements, found {len(placements)}")
    positions: List[np.ndarray] = []
    for index, placement in enumerate(placements):
        position = np.asarray(placement.get("position", []), dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"placement {index} has invalid position")
        if float(placement.get("clearance_m", -1.0)) < 0.28:
            raise ValueError(f"placement {index} violates obstacle clearance")
        if not str(placement.get("anchor_label", "")).strip():
            raise ValueError(f"placement {index} has no furniture anchor label")
        positions.append(position)
    for index, position in enumerate(positions):
        for other in positions[index + 1 :]:
            if float(np.linalg.norm(position - other)) < 1.0:
                raise ValueError("placement separation is below 1 metre")


def _serializable_placements(placements: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(_json_value(placement)) for placement in placements]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = get_data_root()
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=get_habitat_data_root() / "hm3d-train",
    )
    parser.add_argument(
        "--semantic-furniture-root",
        type=Path,
        default=data_root / "visualizations" / "hm3d_train_semantic_topdown",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root
        / "datasets"
        / "offline"
        / "hm3d-train_reduced14_kneel"
        / "placement_sampling_v2",
    )
    parser.add_argument(
        "--raw-val-manifest",
        type=Path,
        default=data_root
        / "datasets"
        / "reduced14_kneel_babel_diversity_v1"
        / "raw-val"
        / "official_val.json",
    )
    parser.add_argument("--num-placements", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(message)s")
    scene_root = args.scene_root.resolve()
    semantic_root = args.semantic_furniture_root.resolve()
    output_root = args.output_root.resolve()
    raw_val_manifest = args.raw_val_manifest.resolve()
    if output_root == scene_root or output_root == semantic_root:
        raise ValueError("output root must be distinct from all source roots")
    if args.num_placements < 1:
        raise ValueError("--num-placements must be positive")
    scenes = _load_scene_list(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    raw_val_count = _raw_val_count(raw_val_manifest)
    generated: List[str] = []
    for scene_index, scene_id in enumerate(scenes):
        furniture_path = semantic_root / scene_id / "furniture_positions.json"
        if not furniture_path.exists():
            raise FileNotFoundError(f"missing furniture manifest for {scene_id}: {furniture_path}")
        objects = _load_furniture(furniture_path)
        simulator = _make_sim(scene_root, scene_id)
        try:
            scene_seed = int(args.seed + scene_index)
            placements = sample_scene_placements(
                simulator,
                objects,
                num_placements=args.num_placements,
                seed=scene_seed,
            )
        finally:
            simulator.close()
        _validate_placements(placements, args.num_placements)
        scene_payload = {
            "version": VERSION,
            "scene_id": scene_id,
            "seed": scene_seed,
            "num_placements": args.num_placements,
            "source_furniture_manifest": str(furniture_path),
            "raw_val_manifest": str(raw_val_manifest),
            "raw_val_record_count": raw_val_count,
            "placement_protocol": {
                "radius_m": [0.5, 1.2],
                "max_snap_error_m": 0.5,
                "min_clearance_m": 0.28,
                "min_interplacement_distance_m": 1.0,
                "attempts_per_anchor": 30,
                "coordinate_conversion": "semantic [x,y,z] -> Habitat [x,z,-y]",
            },
            "placements": _serializable_placements(placements),
        }
        _write_json_atomic(output_root / scene_id / "placements.json", scene_payload)
        generated.append(scene_id)
        LOGGER.info("[%s] wrote %d placements", scene_id, len(placements))
    summary = {
        "version": VERSION,
        "scene_ids": generated,
        "scene_count": len(generated),
        "num_placements_per_scene": args.num_placements,
        "total_placements": len(generated) * args.num_placements,
        "scene_root": str(scene_root),
        "semantic_furniture_root": str(semantic_root),
        "output_root": str(output_root),
        "raw_val_manifest": str(raw_val_manifest),
        "raw_val_record_count": raw_val_count,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "skeleton_generated": False,
        "rgb_generated": False,
        "depth_generated": False,
        "test_used": False,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    LOGGER.info(
        "PLACEMENT_SAMPLING_COMPLETE scenes=%d total=%d output=%s",
        len(generated),
        summary["total_placements"],
        output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
