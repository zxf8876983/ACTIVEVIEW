#!/usr/bin/env python3
"""Run the reproducible v11.5 Stage A acceptance audit.

The default mode audits every serialized Episode and every referenced cached
skeleton archive. ``--verify-habitat`` additionally replays Habitat
ShortestPath for each final current/candidate pair (or the first N Episodes).
This command is intentionally read-only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.policy_episode_builder import audit_episode_files
from ea_avs_mvp_v11.core.paths import get_data_root, get_habitat_data_root

SPLITS = ("train", "val", "test")


def _load_context(root: Path) -> tuple[Dict[str, Path], Dict[str, str]]:
    summary = json.loads((root / "stage_a_summary.json").read_text(encoding="utf-8"))
    files = {split: Path(summary["episode_files"][split]) for split in SPLITS}
    expected: Dict[str, str] = {}
    for split in SPLITS:
        records = json.loads((root / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        expected.update({str(item["record_id"]): split for item in records})
    return files, expected


def _iter_episodes(files: Mapping[str, Path]) -> Iterable[Mapping[str, Any]]:
    for path in files.values():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                episode = json.loads(line)
                if isinstance(episode, Mapping):
                    yield episode


def _verify_habitat(
    files: Mapping[str, Path], habitat_root: Path, max_episodes: int | None,
) -> Dict[str, Any]:
    import numpy as np
    from ea_avs_mvp_v11.scripts.build_policy_episodes import _make_sim
    from ea_avs_mvp_v11.scripts.evaluate_hm3d_train_dynamic_reachability import _path_cost

    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in _iter_episodes(files):
        grouped[str(episode["scene_id"])].append(episode)
    checked = 0
    failures = []
    path_cache: Dict[tuple[str, str, int, int], float | None] = {}
    for scene_id, episodes in sorted(grouped.items()):
        if max_episodes is not None and checked >= max_episodes:
            break
        scene_set_root = next(
            (habitat_root / scene_set for scene_set in ("hm3d-minival", "hm3d-train")
             if (habitat_root / scene_set / scene_id).exists()),
            None,
        )
        if scene_set_root is None:
            failures.append({"scene_id": scene_id, "reason": "missing_habitat_scene"})
            continue
        sim = _make_sim(scene_set_root, scene_id)
        try:
            for episode in episodes:
                if max_episodes is not None and checked >= max_episodes:
                    break
                current = episode["current_view"]
                for candidate in episode["candidate_pool"]:
                    cache_key = (
                        scene_id,
                        str(episode.get("region", "")),
                        int(current["viewpoint_id"]),
                        int(candidate["viewpoint_id"]),
                    )
                    if cache_key not in path_cache:
                        path_cache[cache_key] = _path_cost(
                            sim.pathfinder,
                            np.asarray(current["agent_position"], dtype=np.float32),
                            np.asarray(candidate["snapped_position"], dtype=np.float32),
                        )
                    actual = path_cache[cache_key]
                    expected = float(candidate["geodesic_distance_m"])
                    if actual is None or abs(float(actual) - expected) > 1e-5 * max(1.0, abs(expected)):
                        failures.append({
                            "episode_id": episode.get("episode_id"),
                            "viewpoint_id": candidate.get("viewpoint_id"),
                            "expected": expected,
                            "actual": actual,
                        })
                checked += 1
        finally:
            sim.close()
    return {"episodes_checked": checked, "path_failures": failures, "scenes_checked": len(grouped)}


def main() -> int:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--habitat-root", type=Path, default=get_habitat_data_root())
    parser.add_argument("--skip-cached-skeletons", action="store_true")
    parser.add_argument("--verify-habitat", action="store_true")
    parser.add_argument("--max-habitat-episodes", type=int, default=None)
    args = parser.parse_args()

    files, expected = _load_context(args.dataset_root)
    audit = audit_episode_files(
        files,
        expected_record_splits=expected,
        validate_cached_skeletons=not args.skip_cached_skeletons,
    )
    report: Dict[str, Any] = {"episode_audit": audit}
    if args.verify_habitat:
        report["habitat_shortest_path"] = _verify_habitat(
            files, args.habitat_root, args.max_habitat_episodes,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    passed = all(
        bool(value)
        for key, value in audit["integrity_checks"].items()
        if key != "split_overlap"
    ) and not bool(audit["integrity_checks"].get("split_overlap", False))
    if args.verify_habitat:
        passed = passed and not report["habitat_shortest_path"]["path_failures"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
