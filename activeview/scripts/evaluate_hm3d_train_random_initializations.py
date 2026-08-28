#!/usr/bin/env python3
"""Run many random-start active-view evaluations using reusable caches.

The random starts are continuous samples in the 1.5--3.0 m annulus around
each semantic placement, snapped to the Habitat navmesh and checked for
clearance and a path to the placement.  No RGB is rendered here.  Predictions
for the 32 offline views are cached once; for each random start only the
dynamic navigation pool and vectorized policy selection are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.evaluate_hm3d_train_dynamic_reachability import (
    COMPLETED_SCENES,
    POLICIES,
    REGIONS,
    _entropy,
    _load_mapping,
    _load_model,
    _make_sim,
    _path_cost,
)
from activeview.core.paths import get_data_root, get_habitat_data_root

LOGGER = logging.getLogger("activeview.random_init_eval")
CACHE_VERSION = "random-initialization-cache-v1"


def _seed_for(seed: int, *parts: str) -> int:
    value = "|".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _scene_complete(scene_dir: Path) -> bool:
    try:
        manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("version") == "semantic-region-offline-v2" and int(manifest.get("records", 0)) == 980
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _load_candidates(scene_dir: Path) -> Dict[str, Mapping[str, Any]]:
    payload = json.loads((scene_dir / "candidate_metadata" / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("version") != "semantic-region-v2":
        raise ValueError(f"Unsupported candidate manifest in {scene_dir}")
    return {str(item["region"]): item for item in payload["placements_data"]}


def _sample_initials(
    sim: Any,
    placement: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    clearance_m: float,
    min_radius_m: float,
    max_radius_m: float,
    max_attempts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample continuous, placeable starts and return raw/snapped/path costs."""
    base = np.asarray(placement["position"], dtype=np.float32)
    views = list(placement["viewpoints"])
    rng = np.random.default_rng(seed)
    raw_points: List[np.ndarray] = []
    snapped_points: List[np.ndarray] = []
    placement_costs: List[float] = []
    attempts = 0
    candidate_centers = np.asarray(
        [view.get("snapped_position", view["position"]) for view in views], dtype=np.float32
    )
    while len(raw_points) < count and attempts < max_attempts:
        attempts += 1
        # A narrow room may occupy only a tiny angular portion of the annulus.
        # Jittering the existing candidate positions keeps starts continuous
        # while avoiding a grid-only proposal distribution.
        if attempts % 3 == 0 and candidate_centers.size:
            center = candidate_centers[int(rng.integers(0, len(candidate_centers)))]
            jitter = rng.normal(0.0, 0.12, size=2).astype(np.float32)
            raw = center + np.asarray([jitter[0], 0.0, jitter[1]], dtype=np.float32)
        else:
            radius = float(rng.uniform(min_radius_m, max_radius_m))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            raw = base + np.asarray([radius * np.sin(angle), 0.0, radius * np.cos(angle)], dtype=np.float32)
        snapped = np.asarray(sim.pathfinder.snap_point(raw), dtype=np.float32)
        if not np.isfinite(snapped).all() or not sim.pathfinder.is_navigable(snapped):
            continue
        horizontal_distance = float(np.linalg.norm((snapped - base)[[0, 2]]))
        if horizontal_distance < min_radius_m - 0.08 or horizontal_distance > max_radius_m + 0.08:
            continue
        try:
            if float(sim.pathfinder.distance_to_closest_obstacle(snapped)) < clearance_m:
                continue
        except RuntimeError:
            continue
        cost = _path_cost(sim.pathfinder, snapped, base)
        if cost is None:
            continue
        raw_points.append(raw)
        snapped_points.append(snapped)
        placement_costs.append(float(cost))
    if len(raw_points) != count:
        raise RuntimeError(
            f"Could only sample {len(raw_points)}/{count} placeable starts after {attempts} attempts"
        )
    return (
        np.asarray(raw_points, dtype=np.float32),
        np.asarray(snapped_points, dtype=np.float32),
        np.asarray(placement_costs, dtype=np.float32),
    )


def _build_prediction_cache(
    scene_dir: Path,
    cache_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    category_count: int,
) -> None:
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    for region in REGIONS:
        if not (cache_dir / f"geometry_{region}.npz").exists():
            continue
        target_path = cache_dir / f"predictions_{region}.npz"
        if target_path.exists():
            continue
        region_items = [item for item in manifest["items"] if item["region"] == region]
        if len(region_items) != 980:
            raise ValueError(f"Expected 980 {region} records, found {len(region_items)}")
        predicted = np.empty((len(region_items), 32), dtype=np.int16)
        entropies = np.empty((len(region_items), 32), dtype=np.float32)
        labels = np.empty(len(region_items), dtype=np.int16)
        record_ids: List[str] = []
        for index, item in enumerate(region_items):
            with np.load(scene_dir / str(item["path"])) as archive:
                skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            if skeleton.shape != (32, 3, 30, 17):
                raise ValueError(f"Unexpected skeleton shape {skeleton.shape}")
            tensor = torch.from_numpy(skeleton).to(device=device, dtype=torch.float32).unsqueeze(-1)
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=-1).cpu().numpy()
            if probabilities.shape != (32, category_count):
                raise ValueError(f"Unexpected model output shape {probabilities.shape}")
            predicted[index] = np.argmax(probabilities, axis=-1).astype(np.int16)
            entropies[index] = _entropy(probabilities).astype(np.float32)
            labels[index] = int(item["label_id"])
            record_ids.append(str(item["record_id"]))
        np.savez_compressed(target_path, predicted=predicted, entropy=entropies, labels=labels)
        (cache_dir / f"records_{region}.json").write_text(
            json.dumps(record_ids, ensure_ascii=False), encoding="utf-8"
        )


def _build_geometry_cache(
    scene_dir: Path,
    scene_root: Path,
    cache_dir: Path,
    *,
    count: int,
    seed: int,
    clearance_m: float,
    min_radius_m: float,
    max_radius_m: float,
    max_attempts: int,
) -> None:
    sim = _make_sim(scene_root, scene_dir.name)
    try:
        candidates = _load_candidates(scene_dir)
        for region in REGIONS:
            target_path = cache_dir / f"geometry_{region}.npz"
            if target_path.exists():
                continue
            placement = candidates[region]
            try:
                raw, snapped, placement_cost = _sample_initials(
                    sim,
                    placement,
                    count=count,
                    seed=_seed_for(seed, scene_dir.name, region, "initial"),
                    clearance_m=clearance_m,
                    min_radius_m=min_radius_m,
                    max_radius_m=max_radius_m,
                    max_attempts=max_attempts,
                )
            except RuntimeError as error:
                (cache_dir / f"excluded_{region}.json").write_text(
                    json.dumps(
                        {
                            "scene_id": scene_dir.name,
                            "region": region,
                            "reason": str(error),
                            "protocol": "strict continuous 1.5--3.0 m start; region excluded when no valid sample exists",
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                LOGGER.warning("Excluding %s/%s: %s", scene_dir.name, region, error)
                continue
            views = list(placement["viewpoints"])
            candidate_positions = np.asarray(
                [view.get("snapped_position", view["position"]) for view in views], dtype=np.float32
            )
            candidate_costs = np.full((count, 32), np.nan, dtype=np.float32)
            for initial_index, start in enumerate(snapped):
                for view_index, candidate in enumerate(candidate_positions):
                    value = _path_cost(sim.pathfinder, start, candidate)
                    if value is not None:
                        candidate_costs[initial_index, view_index] = value
            proxy_distance = np.linalg.norm(
                snapped[:, None, [0, 2]] - candidate_positions[None, :, [0, 2]], axis=-1
            )
            proxy_ids = np.argmin(proxy_distance, axis=1).astype(np.int16)
            np.savez_compressed(
                target_path,
                raw_positions=raw,
                snapped_positions=snapped,
                placement_cost_m=placement_cost,
                candidate_positions=candidate_positions,
                candidate_cost_m=candidate_costs,
                no_move_proxy_viewpoint_id=proxy_ids,
                no_move_proxy_distance_m=proxy_distance[np.arange(count), proxy_ids].astype(np.float32),
            )
    finally:
        sim.close()


def _update_metrics(
    metrics: Dict[str, Any], labels: np.ndarray, predictions: np.ndarray, entropies: np.ndarray
) -> None:
    metrics["confusion"] += np.bincount(
        labels.astype(np.int64) * metrics["num_classes"] + predictions.astype(np.int64),
        minlength=metrics["num_classes"] ** 2,
    ).reshape(metrics["num_classes"], metrics["num_classes"])
    metrics["n"] += int(labels.size)
    metrics["entropy_sum"] += float(np.sum(entropies, dtype=np.float64))


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    scores: List[float] = []
    for label in range(confusion.shape[0]):
        true_positive = float(confusion[label, label])
        false_positive = float(confusion[:, label].sum() - true_positive)
        false_negative = float(confusion[label, :].sum() - true_positive)
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def _finalize_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    confusion = np.asarray(metrics["confusion"], dtype=np.int64)
    correct = int(np.trace(confusion))
    n = int(metrics["n"])
    return {
        "n": n,
        "accuracy": float(correct / n) if n else 0.0,
        "macro_f1": _macro_f1_from_confusion(confusion),
        "mean_entropy": float(metrics["entropy_sum"] / n) if n else 0.0,
    }


def _empty_metrics(num_classes: int) -> Dict[str, Any]:
    return {"num_classes": num_classes, "confusion": np.zeros((num_classes, num_classes), dtype=np.int64), "n": 0, "entropy_sum": 0.0}


def _evaluate_scene(
    scene_dir: Path,
    cache_dir: Path,
    *,
    initializations: int | None,
    seed: int,
    category_count: int,
) -> Dict[str, Any]:
    metrics = {policy: _empty_metrics(category_count) for policy in POLICIES}
    region_summary: Dict[str, Any] = {}
    excluded_regions: Dict[str, str] = {}
    for region in REGIONS:
        geometry_path = cache_dir / f"geometry_{region}.npz"
        prediction_path = cache_dir / f"predictions_{region}.npz"
        if not geometry_path.exists() or not prediction_path.exists():
            excluded_path = cache_dir / f"excluded_{region}.json"
            excluded_regions[region] = (
                json.loads(excluded_path.read_text(encoding="utf-8")).get("reason", "missing cache")
                if excluded_path.exists()
                else "missing geometry/prediction cache"
            )
            continue
        with np.load(geometry_path) as geometry, np.load(prediction_path) as prediction:
            available_initializations = int(np.asarray(geometry["candidate_cost_m"]).shape[0])
            region_initializations = available_initializations if initializations is None else min(initializations, available_initializations)
            costs = np.asarray(geometry["candidate_cost_m"], dtype=np.float32)[:region_initializations]
            proxy_ids = np.asarray(geometry["no_move_proxy_viewpoint_id"], dtype=np.int16)[:region_initializations]
            predicted = np.asarray(prediction["predicted"], dtype=np.int16)
            entropies = np.asarray(prediction["entropy"], dtype=np.float32)
            labels = np.asarray(prediction["labels"], dtype=np.int16)
        dynamic_counts: List[int] = []
        for init_index in range(region_initializations):
            current_costs = costs[init_index]
            pool = np.flatnonzero(np.isfinite(current_costs)).astype(np.int16)
            dynamic_counts.append(int(pool.size))
            # NoMove uses an explicit nearest-cached-view proxy because the
            # continuous initial point itself has no offline RGB/pose tensor.
            no_move_view = int(proxy_ids[init_index])
            _update_metrics(
                metrics["NoMove"],
                labels,
                predicted[:, no_move_view],
                entropies[:, no_move_view],
            )
            if pool.size == 0:
                continue
            fixed_view = int(pool[0])
            nearest_view = int(pool[np.nanargmin(current_costs[pool])])
            _update_metrics(metrics["Fixed"], labels, predicted[:, fixed_view], entropies[:, fixed_view])
            _update_metrics(metrics["Nearest"], labels, predicted[:, nearest_view], entropies[:, nearest_view])
            rng = np.random.default_rng(_seed_for(seed, scene_dir.name, region, str(init_index), "random"))
            random_views = pool[rng.integers(0, pool.size, size=labels.size)]
            row_ids = np.arange(labels.size)
            _update_metrics(metrics["Random"], labels, predicted[row_ids, random_views], entropies[row_ids, random_views])
            pool_predictions = predicted[:, pool]
            pool_entropy = entropies[:, pool]
            correct = pool_predictions == labels[:, None]
            oracle_score = np.where(correct, pool_entropy, np.inf)
            oracle_indices = np.argmin(oracle_score, axis=1)
            fallback_indices = np.argmin(pool_entropy, axis=1)
            has_correct = np.any(correct, axis=1)
            oracle_views = pool[np.where(has_correct, oracle_indices, fallback_indices)]
            _update_metrics(metrics["Oracle"], labels, predicted[row_ids, oracle_views], entropies[row_ids, oracle_views])
        region_summary[region] = {
            "initializations": region_initializations,
            "dynamic_reachable_candidates_min": int(min(dynamic_counts)),
            "dynamic_reachable_candidates_mean": float(np.mean(dynamic_counts)),
            "dynamic_reachable_candidates_max": int(max(dynamic_counts)),
            "no_next_view_initializations": int(sum(count == 0 for count in dynamic_counts)),
        }
    return {
        "scene_id": scene_dir.name,
        "records_per_region": 980,
        "initializations": initializations,
        "regions": region_summary,
        "excluded_regions": excluded_regions,
        "policies": {policy: _finalize_metrics(values) for policy, values in metrics.items()},
    }


def _aggregate(scene_results: Sequence[Mapping[str, Any]], category_count: int) -> Dict[str, Any]:
    aggregate = {policy: _empty_metrics(category_count) for policy in POLICIES}
    for scene in scene_results:
        # Reconstructing from scalar metrics is not sufficient for Macro-F1;
        # the scene-level macro-F1 mean is reported consistently with the
        # previous evaluator, while accuracy is weighted by n.
        for policy in POLICIES:
            aggregate[policy]["n"] += int(scene["policies"][policy]["n"])
            aggregate[policy]["entropy_sum"] += float(scene["policies"][policy]["mean_entropy"]) * int(scene["policies"][policy]["n"])
    return {
        policy: {
            "n": int(values["n"]),
            "accuracy": float(sum(float(scene["policies"][policy]["accuracy"]) * int(scene["policies"][policy]["n"]) for scene in scene_results) / values["n"]) if values["n"] else 0.0,
            "macro_f1": float(np.mean([scene["policies"][policy]["macro_f1"] for scene in scene_results])) if scene_results else 0.0,
            "mean_entropy": float(values["entropy_sum"] / values["n"]) if values["n"] else 0.0,
        }
        for policy, values in aggregate.items()
    }


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--scene-set", type=Path, default=data_root / "datasets/offline/hm3d-train")
    parser.add_argument("--cache-root", type=Path, default=data_root / "datasets/strategy_eval_cache/hm3d-train_random_init_500_v1")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--output", type=Path, default=data_root / "results/hm3d_train_random_initializations_500.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--initializations", type=int, default=500)
    parser.add_argument("--clearance-m", type=float, default=0.10)
    parser.add_argument("--min-radius-m", type=float, default=1.5)
    parser.add_argument("--max-radius-m", type=float, default=3.0)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=list(COMPLETED_SCENES))
    args = parser.parse_args()
    if args.initializations <= 0:
        raise ValueError("--initializations must be positive")
    category_names, _ = _load_mapping(data_root)
    model, device = _load_model(args.checkpoint, category_names, args.device)
    scene_results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for scene_id in args.scenes:
        scene_dir = args.scene_set / scene_id
        if not _scene_complete(scene_dir):
            LOGGER.warning("Skipping incomplete scene %s", scene_id)
            continue
        scene_cache = args.cache_root / scene_id
        scene_cache.mkdir(parents=True, exist_ok=True)
        cache_meta = scene_cache / "cache_manifest.json"
        if args.rebuild_cache:
            for path in scene_cache.glob("geometry_*.npz"):
                path.unlink()
            for path in scene_cache.glob("predictions_*.npz"):
                path.unlink()
        _build_geometry_cache(
            scene_dir,
            args.scene_root,
            scene_cache,
            count=args.initializations,
            seed=args.seed,
            clearance_m=args.clearance_m,
            min_radius_m=args.min_radius_m,
            max_radius_m=args.max_radius_m,
            max_attempts=args.max_attempts,
        )
        _build_prediction_cache(scene_dir, scene_cache, model, device, len(category_names))
        cache_meta.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "scene_id": scene_id,
                    "initializations": args.initializations,
                    "seed": args.seed,
                    "continuous_annulus_m": [args.min_radius_m, args.max_radius_m],
                    "clearance_m": args.clearance_m,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "prediction_cache": "predicted_class + entropy for 980 actions × 32 views",
                    "navigation_cache": "raw/snapped starts + 500 × 32 current-to-candidate geodesic costs",
                    "no_move_note": "nearest cached candidate proxy; continuous starts have no offline observation tensor",
                    "excluded_regions": "strict 1.5--3.0 m sampling failures are recorded and omitted, never relaxed silently",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Evaluating cached random starts for %s", scene_id)
        scene_results.append(
            _evaluate_scene(
                scene_dir,
                scene_cache,
                initializations=args.initializations,
                seed=args.seed,
                category_count=len(category_names),
            )
        )
    if not scene_results:
        raise RuntimeError("No complete scenes available")
    result = {
        "protocol": "hm3d-train random continuous initializations dynamic reachability v1",
        "scene_set": str(args.scene_set.resolve()),
        "scene_root": str(args.scene_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "scenes": [scene["scene_id"] for scene in scene_results],
        "scene_count": len(scene_results),
        "initializations_per_scene_region": args.initializations,
        "records_per_scene_region": 980,
        "views_per_record": 32,
        "categories": category_names,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "seed": args.seed,
        "initial_position_protocol": "continuous uniform radius/azimuth in 1.5--3.0 m annulus; navmesh snap; clearance and placement path required",
        "dynamic_reachability_protocol": "recompute current random start to every candidate with Habitat ShortestPath; no placement-only reachability flag is used",
        "policies": {
            "NoMove": "nearest cached candidate proxy for the continuous initial point; exact no-move observation requires RGB/pose extraction at each continuous start",
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
