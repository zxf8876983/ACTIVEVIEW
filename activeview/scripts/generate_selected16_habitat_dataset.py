#!/usr/bin/env python3
"""Generate selected16 pure-color Habitat RGB-estimated skeleton tensors."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.dataset.babel_clean_dataset_generator import BabelCleanDatasetGenerator


def main() -> None:
    runtime_root = get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=runtime_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--pose-backend", choices=("ultralytics_yolo26n",), default="ultralytics_yolo26n")
    parser.add_argument("--yolo-weights", type=Path, default=runtime_root / "checkpoints/ultralytics/yolo26n-pose.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = BabelCleanDatasetGenerator(
        output_root=args.data_root,
        image_size=args.image_size,
        target_frames=args.target_frames,
        seed=args.seed,
        device=args.device,
        pose_backend=args.pose_backend,
        yolo_weights=args.yolo_weights,
    )
    summary = generator.generate_split(
        args.split,
        args.data_root / f"{args.split}.json",
        max_records=args.max_records,
    )
    logging.getLogger(__name__).info("Generated %s", summary)


if __name__ == "__main__":
    main()
