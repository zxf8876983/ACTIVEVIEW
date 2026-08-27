#!/usr/bin/env python3
"""Evaluate policies from the 32 candidate-grid initial viewpoints.

This is the grid counterpart of ``evaluate_hm3d_train_random_initializations``.
Each candidate is independently rechecked against the scene navmesh and
obstacle clearance.  Valid grid starts retain their exact cached observation,
so NoMove is an exact baseline here (unlike the continuous-start proxy).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.core.paths import get_data_root, get_habitat_data_root
from ea_avs_mvp_v11.scripts.evaluate_hm3d_train_dynamic_reachability import (
    COMPLETED_SCENES,
    REGIONS,
    _load_candidates,
    _load_mapping,
    _load_model,
    _make_sim,
    _path_cost,
)
from ea_avs_mvp_v11.scripts.evaluate_hm3d_train_random_initializations import (
    _aggregate,
    _build_prediction_cache,
    _evaluate_scene,
)

LOGGER = logging.getLogger("activeview.grid_init_eval")


def _scene_complete(scene_dir: Path) -> bool:
    try:
        manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("version") == "semantic-region-offline-v2" and int(manifest.get("records", 0)) == 980
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _build_grid_geometry_cache(
    scene_dir: Path,
    scene_root: Path,
    cache_dir: Path,
    *,
    clearance_m: float,
    rebuild: bool,
) -> None:
    sim = _make_sim(scene_root, scene_dir.name)
    try:
        candidates = _load_candidates(scene_dir)
        for region in REGIONS:
            target_path = cache_dir / f"geometry_{region}.npz"
            excluded_path = cache_dir / f"excluded_{region}.json"
            if rebuild:
                target_path.unlink(missing_ok=True)
                excluded_path.unlink(missing_ok=True)
            if target_path.exists():
                continue
            placement = candidates[region]
            base = np.asarray(placement["position"], dtype=np.float32)
            views = list(placement["viewpoints"])
            valid = []
            for view in views:
                viewpoint_id = int(view["viewpoint_id"])
                snapped = np.asarray(view.get("snapped_position", view["position"]), dtype=np.float32)
                cost = _path_cost(sim.pathfinder, base, snapped)
                if cost is None or not sim.pathfinder.is_navigable(snapped):
                    continue
                try:
                    if float(sim.pathfinder.distance_to_closest_obstacle(snapped)) < clearance_m:
                        continue
                except RuntimeError:
                    continue
                valid.append((viewpoint_id, np.asarray(view["position"], dtype=np.float32), snapped, float(cost)))
            if not valid:
                excluded_path.write_text(
                    json.dumps(
                        {
                            "scene_id": scene_dir.name,
                            "region": region,
                            "reason": "No candidate-grid viewpoint passed navmesh, clearance, and placement-path checks",
                            "protocol": "32 grid starts; invalid starts excluded",
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                LOGGER.warning("Excluding %s/%s: no valid grid start", scene_dir.name, region)
                continue
            candidate_positions = np.asarray(
                [view.get("snapped_position", view["position"]) for view in views], dtype=np.float32
            )
            count = len(valid)
            costs = np.full((count, 32), np.nan, dtype=np.float32)
            raw_positions = np.asarray([item[1] for item in valid], dtype=np.float32)
            snapped_positions = np.asarray([item[2] for item in valid], dtype=np.float32)
            placement_costs = np.asarray([item[3] for item in valid], dtype=np.float32)
            initial_ids = np.asarray([item[0] for item in valid], dtype=np.int16)
            for row, start in enumerate(snapped_positions):
                for column, candidate in enumerate(candidate_positions):
                    if column == int(initial_ids[row]):
                        continue
                    value = _path_cost(sim.pathfinder, start, candidate)
                    if value is not None:
                        costs[row, column] = value
            proxy_distance = np.linalg.norm(
                snapped_positions[:, None, [0, 2]] - candidate_positions[None, :, [0, 2]], axis=-1
            )
            np.savez_compressed(
                target_path,
                raw_positions=raw_positions,
                snapped_positions=snapped_positions,
                placement_cost_m=placement_costs,
                candidate_positions=candidate_positions,
                candidate_cost_m=costs,
                initial_viewpoint_id=initial_ids,
                no_move_proxy_viewpoint_id=initial_ids,
                no_move_proxy_distance_m=np.zeros(count, dtype=np.float32),
            )
    finally:
        sim.close()


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--scene-set", type=Path, default=data_root / "datasets/offline/hm3d-train")
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/strategy_eval_cache/hm3d-train_grid_init_32_v1")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--output", type=Path, default=data_root / "results/hm3d_train_grid_initializations_32.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clearance-m", type=float, default=0.10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=list(COMPLETED_SCENES))
    args = parser.parse_args()
    category_names, _ = _load_mapping(data_root)
    model, device = _load_model(args.checkpoint, category_names, args.device)
    scene_results = []
    started = time.perf_counter()
    for scene_id in args.scenes:
        scene_dir = args.scene_set / scene_id
        if not _scene_complete(scene_dir):
            LOGGER.warning("Skipping incomplete scene %s", scene_id)
            continue
        scene_cache = args.cache_root / scene_id
        scene_cache.mkdir(parents=True, exist_ok=True)
        _build_grid_geometry_cache(
            scene_dir,
            args.scene_root,
            scene_cache,
            clearance_m=args.clearance_m,
            rebuild=args.rebuild_cache,
        )
        _build_prediction_cache(scene_dir, scene_cache, model, device, len(category_names))
        scene_results.append(
            _evaluate_scene(
                scene_dir,
                scene_cache,
                initializations=None,
                seed=42,
                category_count=len(category_names),
            )
        )
    if not scene_results:
        raise RuntimeError("No complete scenes available")
    result = {
        "protocol": "hm3d-train 32-grid-initializations dynamic reachability v1",
        "scene_set": str(args.scene_set.resolve()),
        "scene_root": str(args.scene_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "scenes": [scene["scene_id"] for scene in scene_results],
        "scene_count": len(scene_results),
        "requested_initializations_per_scene_region": 32,
        "views_per_record": 32,
        "categories": category_names,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "grid_start_protocol": "the 32 candidate viewpoints are rechecked for navmesh navigability, clearance, and placement connectivity; invalid grid starts are excluded",
        "dynamic_reachability_protocol": "recompute current grid start to every pending candidate with Habitat ShortestPath; current start is excluded from the next-view pool",
        "policies": {
            "NoMove": "exact cached observation at the initial grid viewpoint",
            "Fixed": "minimum viewpoint ID in the dynamic reachable pool",
            "Random": "seeded random candidate in the dynamic reachable pool",
            "Nearest": "minimum current-to-candidate geodesic cost",
            "Oracle": "hindsight GT-correctness upper bound in the dynamic reachable pool",
        },
        "policy_results": _aggregate(scene_results, len(category_names)),
        "per_scene": scene_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["policy_results"], ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
