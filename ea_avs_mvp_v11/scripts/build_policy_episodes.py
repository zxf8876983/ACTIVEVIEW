#!/usr/bin/env python3
"""Build Stage A active-view episode manifests from cached offline views."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from collections import Counter
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import habitat_sim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.policy_episode_builder import REGIONS, audit_episode_files, load_scene_index, iter_scene_region_episodes
from ea_avs_mvp_v11.core.paths import get_data_root, get_habitat_data_root
from ea_avs_mvp_v11.dataset.policy_split import SPLITS, audit_policy_splits, load_policy_splits
from ea_avs_mvp_v11.scripts.evaluate_hm3d_train_dynamic_reachability import _make_sim, _path_cost


LOGGER = logging.getLogger("activeview.build_policy_episodes")


def _scene_dirs(offline_root: Path, scene_sets: Sequence[str]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    for scene_set in scene_sets:
        root = offline_root / scene_set
        if not root.exists():
            continue
        result.extend((scene_set, path) for path in sorted(root.iterdir()) if path.is_dir())
    return result


def _valid_scene(index: Any) -> bool:
    return index.manifest.get("version") == "semantic-region-offline-v2"


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=data_root / "datasets/policy_v11_5/splits")
    parser.add_argument("--offline-root", type=Path, default=data_root / "datasets/offline")
    parser.add_argument("--habitat-root", type=Path, default=get_habitat_data_root())
    parser.add_argument("--scene-sets", nargs="+", default=["hm3d-minival", "hm3d-train"])
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/episodes")
    parser.add_argument("--summary", type=Path, default=data_root / "datasets/policy_v11_5/stage_a_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clearance-m", type=float, default=0.10)
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional smoke-test scene limit")
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--max-records", type=int, default=None, help="Optional smoke-test action-record limit")
    args = parser.parse_args()

    splits = load_policy_splits(args.split_dir)
    if args.max_records is not None:
        selected_ids = {
            str(item["record_id"])
            for split in SPLITS
            for item in splits[split]
        }
        selected_ids = set(sorted(selected_ids)[:args.max_records])
        splits = {
            split: [item for item in splits[split] if str(item["record_id"]) in selected_ids]
            for split in SPLITS
        }
    split_audit = audit_policy_splits(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {split: (args.output_dir / f"{split}_episodes.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    exclusion_handle = (args.output_dir / "exclusions.jsonl").open("w", encoding="utf-8")
    episode_counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    candidate_counts: List[int] = []
    used_scenes: List[str] = []
    scanned = 0
    try:
        scene_dirs = _scene_dirs(args.offline_root, args.scene_sets)
        if args.max_scenes is not None:
            scene_dirs = scene_dirs[:args.max_scenes]
        for scene_set, scene_dir in scene_dirs:
            scanned += 1
            try:
                index = load_scene_index(scene_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                exclusion = {"scene_set": scene_set, "scene_id": scene_dir.name, "excluded_reason": "incomplete_scene", "detail": str(error)}
                exclusion_handle.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
                exclusions["incomplete_scene"] += 1
                continue
            scene_root = args.habitat_root / scene_set
            try:
                sim = _make_sim(scene_root, index.scene_id)
            except (OSError, RuntimeError, FileNotFoundError) as error:
                exclusion = {"scene_set": scene_set, "scene_id": index.scene_id, "excluded_reason": "incomplete_scene", "detail": str(error)}
                exclusion_handle.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
                exclusions["incomplete_scene"] += 1
                continue
            used_scenes.append(index.scene_id)
            try:
                for region in args.regions:
                    for split in SPLITS:
                        for episode, exclusion in iter_scene_region_episodes(
                            index, policy_records=splits[split], region=region, pathfinder=sim.pathfinder,
                            path_cost_fn=_path_cost, global_seed=args.seed, clearance_m=args.clearance_m,
                        ):
                            if episode is not None:
                                handles[split].write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")) + "\n")
                                episode_counts[split] += 1
                                candidate_counts.append(int(episode["candidate_count"]))
                            elif exclusion is not None:
                                exclusion_handle.write(json.dumps(exclusion, ensure_ascii=False, separators=(",", ":")) + "\n")
                                exclusions[str(exclusion["excluded_reason"])] += 1
            finally:
                sim.close()
    finally:
        for handle in handles.values():
            handle.close()
        exclusion_handle.close()

    episode_files = {split: args.output_dir / f"{split}_episodes.jsonl" for split in SPLITS}
    expected_record_splits = {
        str(item["record_id"]): str(item["policy_split"])
        for split in SPLITS for item in splits[split]
    }
    episode_audit = audit_episode_files(
        episode_files,
        expected_record_splits=expected_record_splits,
        validate_cached_skeletons=True,
    )

    summary = {
        "protocol": "ACTIVEVIEW v11.5 Stage A",
        "seed": args.seed,
        "policy_split": {
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "counts": {split: len(splits[split]) for split in SPLITS},
        },
        "policy_split_audit": split_audit,
        "per_class_split_counts": json.loads((args.split_dir / "summary.json").read_text(encoding="utf-8")).get("per_class_split_counts", {}),
        "scenes_scanned": scanned,
        "complete_scenes_used": len(used_scenes),
        "scene_ids_used": used_scenes,
        "episodes": {**{split: episode_counts[split] for split in SPLITS}, "total": sum(episode_counts.values())},
        "candidate_pool": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "mean": mean(candidate_counts) if candidate_counts else 0.0,
            "median": median(candidate_counts) if candidate_counts else 0.0,
            "max": max(candidate_counts) if candidate_counts else 0,
        },
        "excluded": {name: exclusions[name] for name in ("incomplete_scene", "no_valid_grid_start", "no_reachable_next_candidate", "missing_cached_skeleton")},
        "episode_audit": episode_audit["counts"],
        "integrity_checks": episode_audit["integrity_checks"],
        "offline_root": str(args.offline_root.resolve()),
        "scene_sets": list(args.scene_sets),
        "episode_files": {split: str((args.output_dir / f"{split}_episodes.jsonl").resolve()) for split in SPLITS},
        "exclusions_file": str((args.output_dir / "exclusions.jsonl").resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
