#!/usr/bin/env python3
"""Create descriptive Stage B diagnostics for Stage C without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.utility_label_builder import file_sha256
from ea_avs_mvp_v11.core.paths import get_data_root


SPLITS = ("train", "val", "test")


def _percentiles(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    p10, p25, p50, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()), "p10": float(p10), "p25": float(p25), "p50": float(p50), "p75": float(p75), "p90": float(p90)}


def _bucket(value: float, edges: List[float]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            return f"[{lower:g},{upper:g})"
    return f"[{edges[-1]:g},inf)"


def _summarize_bucket(values: Mapping[str, List[float]]) -> Dict[str, Any]:
    return {key: {**_percentiles(items), "positive_ratio": float(np.mean(np.asarray(items) > 0.0)) if items else 0.0} for key, items in sorted(values.items())}


def build(stage_b_root: Path, output_path: Path) -> Dict[str, Any]:
    stage_b_summary = json.loads((stage_b_root / "stage_b_summary.json").read_text(encoding="utf-8"))
    stage_a_summary_path = stage_b_root.parent / "stage_a_summary.json"
    stage_a_summary = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
    output: Dict[str, Any] = {
        "protocol": "ACTIVEVIEW v11.5 Stage C Stage B diagnostics",
        "stage": "C0",
        "source_stage_b_summary": str((stage_b_root / "stage_b_summary.json").resolve()),
        "source_stage_b_summary_sha256": file_sha256(stage_b_root / "stage_b_summary.json"),
        "splits": {},
    }
    for split in SPLITS:
        rows = [json.loads(line) for line in (stage_b_root / "utility_labels" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        stage_a_by_id = {
            str(episode["episode_id"]): episode
            for episode in (
                json.loads(line)
                for line in Path(stage_a_summary["episode_files"][split]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        max_utils = [max(float(item["utility"]) for item in row["candidates"]) for row in rows]
        pair_utils = [float(item["utility"]) for row in rows for item in row["candidates"]]
        azimuth_bins: Dict[str, List[float]] = {}
        geodesic_bins: Dict[str, List[float]] = {}
        regions: Dict[str, List[float]] = {}
        classes: Dict[str, List[float]] = {}
        for row in rows:
            stage_a_episode = stage_a_by_id.get(str(row["episode_id"]))
            candidate_azimuth = {
                int(item["viewpoint_id"]): float(item["relative_azimuth_deg"])
                for item in (stage_a_episode or {}).get("candidate_pool", [])
            }
            for candidate in row["candidates"]:
                utility = float(candidate["utility"])
                regions.setdefault(str(row["region"]), []).append(utility)
                classes.setdefault(str(row["label_id"]), []).append(utility)
                azimuth = candidate_azimuth.get(int(candidate["viewpoint_id"]))
                if azimuth is not None:
                    azimuth_bins.setdefault(_bucket(azimuth, [-180.0, -90.0, 0.0, 90.0, 180.0]), []).append(utility)
                geodesic_bins.setdefault(_bucket(float(candidate["geodesic_distance_m"]), [0.0, 2.0, 3.0, 4.0, 5.0]), []).append(utility)
        output["splits"][split] = {
            "episode_count": len(rows),
            "recognition": stage_b_summary.get("metrics", {}).get(split, {}).get("policies", {}),
            "max_candidate_utility": _percentiles(max_utils),
            "candidate_pair_utility": {
                **_percentiles(pair_utils),
                "positive_ratio": float(np.mean(np.asarray(pair_utils) > 0.0)) if pair_utils else 0.0,
                "near_zero_ratio": float(np.mean(np.abs(np.asarray(pair_utils)) <= 1e-6)) if pair_utils else 0.0,
                "negative_ratio": float(np.mean(np.asarray(pair_utils) < 0.0)) if pair_utils else 0.0,
            },
            "headroom": stage_b_summary.get("metrics", {}).get(split, {}).get("headroom", {}),
            "by_region": _summarize_bucket(regions),
            "by_action_class": _summarize_bucket(classes),
            "by_geodesic_distance": _summarize_bucket(geodesic_bins),
            "by_relative_azimuth": _summarize_bucket(azimuth_bins),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--output", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/stage_b_diagnostics.json")
    args = parser.parse_args()
    print(json.dumps(build(args.stage_b_root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
