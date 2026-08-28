#!/usr/bin/env python3
"""Refresh cached camera rotations without rerunning visual perception."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.camera_pose import camera_rotation_wxyz
from activeview.core.paths import get_data_root


LOGGER = logging.getLogger("activeview.refresh_rotations")
OFFLINE_VERSION = "semantic-region-offline-v2"
CANDIDATE_VERSION = "semantic-region-v2"


def _scene_dirs(
    offline_root: Path,
    scene_sets: Sequence[str],
    scene_ids: Sequence[str] | None = None,
) -> Iterable[Tuple[str, Path]]:
    selected = None if scene_ids is None else {str(scene_id) for scene_id in scene_ids}
    for scene_set in scene_sets:
        root = offline_root / scene_set
        if not root.exists():
            continue
        for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if selected is not None and scene_dir.name not in selected:
                continue
            yield scene_set, scene_dir


def _region_rotations(candidate_manifest: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    rotations: Dict[str, np.ndarray] = {}
    for placement in candidate_manifest.get("placements_data", []):
        region = str(placement["region"])
        base = np.asarray(placement["position"], dtype=np.float32)
        region_rotations = []
        for view in placement["viewpoints"]:
            agent_position = np.asarray(
                view.get("snapped_position", view["position"]), dtype=np.float32
            )
            rotation = camera_rotation_wxyz(agent_position, base)
            view["camera_rotation_wxyz"] = rotation.tolist()
            region_rotations.append(rotation)
        rotations[region] = np.asarray(region_rotations, dtype=np.float32)
    return rotations


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _refresh_npz(path: Path, rotations: np.ndarray) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        if "viewpoint_rotations_wxyz" not in archive:
            raise ValueError(f"Missing viewpoint rotations in {path}")
        old = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
        if old.shape != rotations.shape:
            raise ValueError(f"Unexpected rotation shape in {path}: {old.shape}")
        if np.allclose(old, rotations, rtol=0.0, atol=1e-6):
            return False
        arrays = {key: archive[key] for key in archive.files}
    arrays["viewpoint_rotations_wxyz"] = rotations
    temporary = path.with_name(f".{path.name}.rotation_tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _refresh_scene(scene_dir: Path) -> Dict[str, int]:
    manifest_path = scene_dir / "manifest.json"
    candidate_path = scene_dir / "candidate_metadata" / "manifest.json"
    if not manifest_path.exists() or not candidate_path.exists():
        return {"status_skipped": 1}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    if manifest.get("version") != OFFLINE_VERSION or candidate_manifest.get("version") != CANDIDATE_VERSION:
        return {"status_skipped": 1}

    rotations = _region_rotations(candidate_manifest)
    candidate_manifest["rotation_reference"] = "exact_offline_render_state"
    candidate_manifest["sensor_height_m"] = 1.1
    candidate_manifest["target_height_m"] = 0.85
    _atomic_write_json(candidate_path, candidate_manifest)

    changed = 0
    records = 0
    for item in manifest.get("items", []):
        region = str(item["region"])
        if region not in rotations:
            raise ValueError(f"Missing rotation metadata for region {region} in {scene_dir}")
        path = scene_dir / str(item["path"])
        if _refresh_npz(path, rotations[region]):
            changed += 1
        records += 1

    manifest["rotation_reference"] = "exact_offline_render_state"
    manifest["sensor_height_m"] = 1.1
    manifest["target_height_m"] = 0.85
    _atomic_write_json(manifest_path, manifest)
    return {"status_complete": 1, "records": records, "npz_changed": changed}


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-root", type=Path, default=data_root / "datasets/offline")
    parser.add_argument("--scene-sets", nargs="+", default=["hm3d-minival", "hm3d-train"])
    parser.add_argument("--scene-ids", nargs="+", default=None)
    args = parser.parse_args()

    totals: Dict[str, int] = {"scenes_scanned": 0, "scenes_complete": 0, "scenes_skipped": 0, "records": 0, "npz_changed": 0}
    for scene_set, scene_dir in _scene_dirs(args.offline_root, args.scene_sets, args.scene_ids):
        totals["scenes_scanned"] += 1
        result = _refresh_scene(scene_dir)
        if result.get("status_complete"):
            totals["scenes_complete"] += 1
            totals["records"] += result["records"]
            totals["npz_changed"] += result["npz_changed"]
            LOGGER.info("%s/%s: updated %d/%d NPZ files", scene_set, scene_dir.name, result["npz_changed"], result["records"])
        else:
            totals["scenes_skipped"] += 1
    print(json.dumps(totals, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
