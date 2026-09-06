#!/usr/bin/env python3
"""Build frozen DINOv2 spatial features for visited reduced14 observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.rgb_cache import build_or_load_spatial_cache
from activeview.perception.rgb_features import load_dinov2, observation_keys_from_feature_rows


def build_cache(data_root: Path, device: torch.device, batch_size: int) -> dict[str, object]:
    policy_root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rgb_root = data_root / "datasets/rgb_reduced14_kneel_eight_placement_v1/visited_s0_s1"
    cache_dir = data_root / "features/dinov2_vitb14_spatial4x4_reduced14_eight_placement/initial_history"
    split_counts: dict[str, int] = {}
    all_keys = set()
    for split in ("train", "val", "test"):
        rows = load_jsonl(policy_root / "stage_d/features" / f"{split}.jsonl")
        keys, _ = observation_keys_from_feature_rows(rows)
        split_counts[split] = len(keys)
        all_keys.update(keys)
    ordered = sorted(all_keys, key=lambda item: item.tuple)
    values, _, info = build_or_load_spatial_cache(
        rgb_root=rgb_root,
        cache_dir=cache_dir,
        keys=ordered,
        model_loader=load_dinov2,
        device=device,
        batch_size=batch_size,
    )
    result: dict[str, object] = {
        "cache_dir": str(cache_dir.resolve()),
        "unique_observations": len(ordered),
        "split_observations": split_counts,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        **info,
        "future_candidate_rgb_used": False,
    }
    (cache_dir / "build_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; spatial RGB extraction requires GPU")
    print(json.dumps(build_cache(args.data_root.resolve(), device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
