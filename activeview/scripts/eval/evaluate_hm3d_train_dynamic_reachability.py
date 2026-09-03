#!/usr/bin/env python3
"""文件用途：
    执行只读验证或评估入口。

主要输入：
    - 冻结预测、缓存与评估协议。
主要输出：
    - 指标摘要或验证报告。
项目角色：
    - 属于 evaluation 脚本入口，不修改模型和数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import habitat_sim
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.recognition.stgcn.model import STGCN
from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.perception.skeleton import get_skeleton_definition

LOGGER = logging.getLogger("activeview.dynamic_eval")
REGIONS: Tuple[str, ...] = ("bedroom", "dining_area", "kitchen", "living_room")
POLICIES: Tuple[str, ...] = ("NoMove", "Fixed", "Random", "Nearest", "Oracle")

# These are the first ten completed folders.  The evaluator deliberately does
# not glob the whole data root, so an incomplete eleventh scene cannot enter a
# result by accident while generation is still running.
COMPLETED_SCENES: Tuple[str, ...] = (
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
)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=-1)


def _stable_seed(seed: int, scene_id: str, region: str) -> int:
    digest = hashlib.sha256(f"{seed}|{scene_id}|{region}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _macro_f1(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> float:
    if not targets:
        return 0.0
    scores: List[float] = []
    for label in range(num_classes):
        true_positive = sum(t == label and p == label for t, p in zip(targets, predictions))
        false_positive = sum(t != label and p == label for t, p in zip(targets, predictions))
        false_negative = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def _make_sim(scene_root: Path, scene_id: str) -> habitat_sim.Simulator:
    scene_dir = scene_root / scene_id
    glbs = sorted(scene_dir.glob("*.basis.glb"))
    navmeshes = sorted(scene_dir.glob("*.basis.navmesh"))
    if len(glbs) != 1 or len(navmeshes) != 1:
        raise FileNotFoundError(f"Expected one GLB/navmesh in {scene_dir}")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glbs[0])
    backend.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.AgentConfiguration()]))
    if not sim.pathfinder.load_nav_mesh(str(navmeshes[0])):
        sim.close()
        raise RuntimeError(f"Could not load navmesh {navmeshes[0]}")
    return sim


def _path_cost(pathfinder: Any, start: np.ndarray, end: np.ndarray) -> float | None:
    """Return a geodesic path cost, or ``None`` when the points are unreachable."""
    start_snap = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
    end_snap = np.asarray(pathfinder.snap_point(end), dtype=np.float32)
    if not np.isfinite(start_snap).all() or not np.isfinite(end_snap).all():
        return None
    if not pathfinder.is_navigable(start_snap) or not pathfinder.is_navigable(end_snap):
        return None
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_snap
    shortest_path.requested_end = end_snap
    if not pathfinder.find_path(shortest_path):
        return None
    cost = float(shortest_path.geodesic_distance)
    return cost if np.isfinite(cost) else None


def _load_mapping(data_root: Path) -> Tuple[List[str], Dict[str, int]]:
    candidates = sorted(data_root.glob("datasets/stgcn_selected16*/label_mapping.json"))
    if not candidates:
        candidates = sorted(data_root.glob("datasets/stgcn_babel_selected16*/label_mapping.json"))
    if not candidates:
        raise FileNotFoundError("No selected16 label_mapping.json found")
    mapping = json.loads(candidates[0].read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    return names, {str(name): int(index) for name, index in mapping.items()}


def _load_model(checkpoint: Path, names: Sequence[str], device_name: str) -> Tuple[STGCN, torch.device]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = STGCN(
        in_channels=3,
        num_classes=len(names),
        graph_strategy="spatial",
        edge_importance_weighting=True,
        skel_def=get_skeleton_definition(backend="h36m_17"),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, device


def _scene_complete(scene_dir: Path) -> bool:
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest.get("version") == "semantic-region-offline-v2"
            and int(manifest.get("records", 0)) == 980
            and int(manifest.get("regions", 0)) == 4
            and int(manifest.get("views_per_record", 0)) == 32
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_candidates(scene_dir: Path) -> Dict[str, Mapping[str, Any]]:
    payload = json.loads((scene_dir / "candidate_metadata" / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("version") != "semantic-region-v2":
        raise ValueError(f"Unsupported candidate metadata version in {scene_dir}")
    return {str(item["region"]): item for item in payload["placements_data"]}


def _choose_initial_and_pool(
    *,
    sim: habitat_sim.Simulator,
    placement: Mapping[str, Any],
    seed: int,
    scene_id: str,
    region: str,
    include_current: bool,
) -> Tuple[int, np.ndarray, List[int], Dict[int, float], Dict[str, Any]]:
    views = list(placement["viewpoints"])
    placement_position = np.asarray(placement["position"], dtype=np.float32)
    placement_costs = {int(view["viewpoint_id"]): _path_cost(
        sim.pathfinder, placement_position, np.asarray(view.get("snapped_position", view["position"]), dtype=np.float32)
    ) for view in views}
    placeable = [view_id for view_id, cost in placement_costs.items() if cost is not None]
    if not placeable:
        raise RuntimeError(f"No placeable initial candidate for {scene_id}/{region}")
    initial_id = random.Random(_stable_seed(seed, scene_id, region)).choice(sorted(placeable))
    initial_view = next(view for view in views if int(view["viewpoint_id"]) == initial_id)
    initial_position = np.asarray(initial_view.get("snapped_position", initial_view["position"]), dtype=np.float32)
    dynamic_costs = {
        int(view["viewpoint_id"]): _path_cost(
            sim.pathfinder,
            initial_position,
            np.asarray(view.get("snapped_position", view["position"]), dtype=np.float32),
        )
        for view in views
    }
    pool = sorted(view_id for view_id, cost in dynamic_costs.items() if cost is not None)
    if not include_current:
        pool = [view_id for view_id in pool if view_id != initial_id]
    metadata = {
        "initial_viewpoint_id": initial_id,
        "initial_viewpoint_position": initial_position.tolist(),
        "placement_reachable_candidates": len(placeable),
        "dynamic_reachable_candidates": len(pool),
        "excluded_current_viewpoint": not include_current,
    }
    return initial_id, initial_position, pool, dynamic_costs, metadata


def _choose_view(
    policy: str,
    pool: Sequence[int],
    costs: Mapping[int, float],
    probabilities: np.ndarray,
    entropy: np.ndarray,
    target: int,
    seed: int,
    key: str,
) -> int:
    if not pool:
        raise ValueError("Cannot choose a view from an empty dynamic pool")
    if policy == "Fixed":
        return int(min(pool))
    if policy == "Random":
        return int(random.Random(_stable_seed(seed, key, policy)).choice(list(pool)))
    if policy == "Nearest":
        return int(min(pool, key=lambda view_id: (float(costs[view_id]), int(view_id))))
    correct = [view_id for view_id in pool if int(np.argmax(probabilities[view_id])) == target]
    if correct:
        return int(min(correct, key=lambda view_id: (float(entropy[view_id]), int(view_id))))
    return int(min(pool, key=lambda view_id: (float(entropy[view_id]), int(view_id))))


def _evaluate_scene(
    *,
    scene_dir: Path,
    scene_root: Path,
    model: STGCN,
    device: torch.device,
    category_names: Sequence[str],
    seed: int,
    include_current: bool,
) -> Dict[str, Any]:
    scene_id = scene_dir.name
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = _load_candidates(scene_dir)
    sim = _make_sim(scene_root, scene_id)
    try:
        geometry: Dict[str, Dict[str, Any]] = {}
        for region in REGIONS:
            _, _, pool, costs, metadata = _choose_initial_and_pool(
                sim=sim,
                placement=candidates[region],
                seed=seed,
                scene_id=scene_id,
                region=region,
                include_current=include_current,
            )
            geometry[region] = {"pool": pool, "costs": costs, "metadata": metadata}

        by_policy: Dict[str, Dict[str, Any]] = {
            policy: {"targets": [], "predictions": [], "entropies": [], "viewpoint_ids": []}
            for policy in POLICIES
        }
        skipped_no_next_view = {region: 0 for region in REGIONS}
        items = manifest["items"]
        for item_index, item in enumerate(items, 1):
            region = str(item["region"])
            item_path = scene_dir / str(item["path"])
            with np.load(item_path) as archive:
                skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            if skeleton.shape != (32, 3, 30, 17):
                raise ValueError(f"Unexpected skeleton shape {skeleton.shape} in {item_path}")
            input_tensor = torch.from_numpy(skeleton).to(device=device, dtype=torch.float32).unsqueeze(-1)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor), dim=-1).cpu().numpy()
            entropies = _entropy(probabilities)
            target = int(item["label_id"])
            pool = geometry[region]["pool"]
            costs = geometry[region]["costs"]
            if not pool:
                # A valid initial placement can be an isolated navmesh
                # island. This is a geometry outcome, not permission to use
                # an unreachable candidate for a movement policy.
                skipped_no_next_view[region] += 1
            for policy in POLICIES:
                if policy == "NoMove":
                    view_id = int(geometry[region]["metadata"]["initial_viewpoint_id"])
                elif not pool:
                    # No valid next action exists for this movement policy.
                    continue
                else:
                    view_id = _choose_view(
                        policy,
                        pool,
                        costs,
                        probabilities,
                        entropies,
                        target,
                        seed,
                        f"{scene_id}|{region}|{item['record_id']}",
                    )
                by_policy[policy]["targets"].append(target)
                by_policy[policy]["predictions"].append(int(np.argmax(probabilities[view_id])))
                by_policy[policy]["entropies"].append(float(entropies[view_id]))
                by_policy[policy]["viewpoint_ids"].append(view_id)
            if item_index % 200 == 0:
                LOGGER.info("[%s] processed %d/%d records", scene_id, item_index, len(items))

        policy_results: Dict[str, Any] = {}
        for policy, values in by_policy.items():
            targets = values["targets"]
            predictions = values["predictions"]
            policy_results[policy] = {
                "n": len(targets),
                "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))) if targets else 0.0,
                "macro_f1": _macro_f1(targets, predictions, len(category_names)),
                "mean_entropy": float(np.mean(values["entropies"])) if targets else 0.0,
                "viewpoint_histogram": {
                    str(view_id): int(values["viewpoint_ids"].count(view_id))
                    for view_id in sorted(set(values["viewpoint_ids"]))
                },
            }
        return {
            "scene_id": scene_id,
            "records": len(items),
            "regions": list(REGIONS),
            "geometry": {
                region: {
                    **geometry[region]["metadata"],
                    "dynamic_reachable_viewpoint_ids": geometry[region]["pool"],
                }
                for region in REGIONS
            },
            "skipped_no_next_view": skipped_no_next_view,
            "policies": policy_results,
        }
    finally:
        sim.close()


def _aggregate(scene_results: Sequence[Mapping[str, Any]], category_count: int) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for policy in POLICIES:
        entropies: List[float] = []
        for scene in scene_results:
            # Per-scene aggregates are sufficient for the main report, while
            # preserving macro-F1 as the mean of scene-level action metrics.
            metrics = scene["policies"][policy]
            entropies.extend([float(metrics["mean_entropy"])])
        n = sum(int(scene["policies"][policy]["n"]) for scene in scene_results)
        accuracy = (
            sum(float(scene["policies"][policy]["accuracy"]) * int(scene["policies"][policy]["n"]) for scene in scene_results)
            / n
            if n
            else 0.0
        )
        aggregate[policy] = {
            "n": n,
            "accuracy": float(accuracy),
            "macro_f1": float(np.mean([scene["policies"][policy]["macro_f1"] for scene in scene_results]))
            if scene_results
            else 0.0,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
    return aggregate


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--scene-set", type=Path, default=data_root / "datasets/offline/hm3d-train")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--output", type=Path, default=data_root / "results/hm3d_train_dynamic_reachability_10scenes.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-current", action="store_true", help="Allow the sampled initial viewpoint as a next-view candidate")
    parser.add_argument("--scenes", nargs="*", default=list(COMPLETED_SCENES))
    args = parser.parse_args()

    category_names, _ = _load_mapping(args.data_root)
    model, device = _load_model(args.checkpoint, category_names, args.device)
    scene_results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for scene_id in args.scenes:
        scene_dir = args.scene_set / scene_id
        if not _scene_complete(scene_dir):
            LOGGER.warning("Skipping incomplete scene %s", scene_id)
            continue
        LOGGER.info("Evaluating %s on %s", scene_id, device)
        scene_results.append(
            _evaluate_scene(
                scene_dir=scene_dir,
                scene_root=args.scene_root,
                model=model,
                device=device,
                category_names=category_names,
                seed=args.seed,
                include_current=args.include_current,
            )
        )

    if not scene_results:
        raise RuntimeError("No complete scenes were available for evaluation")
    result = {
        "protocol": "hm3d-train semantic-region dynamic reachability v1",
        "scene_set": str(args.scene_set.resolve()),
        "scene_root": str(args.scene_root.resolve()),
        "scenes": [scene["scene_id"] for scene in scene_results],
        "scene_count": len(scene_results),
        "records_per_scene": 980,
        "regions_per_scene": 4,
        "views_per_region": 32,
        "checkpoint": str(args.checkpoint.resolve()),
        "categories": list(category_names),
        "device": str(device),
        "seed": args.seed,
        "initial_view_protocol": "one deterministic random candidate per scene/region, chosen from candidates reachable from the placement",
        "dynamic_reachability_protocol": "recompute Habitat navmesh paths from the sampled initial robot position to every candidate; no stored placement-only flag is used for the next-view pool",
        "next_view_excludes_current": not args.include_current,
        "oracle_definition": "hindsight GT-correctness upper bound over the dynamically reachable candidate pool; fallback to minimum entropy if no candidate is correct",
        "policy_results": _aggregate(scene_results, len(category_names)),
        "per_scene": scene_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({policy: result["policy_results"][policy] for policy in POLICIES}, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
