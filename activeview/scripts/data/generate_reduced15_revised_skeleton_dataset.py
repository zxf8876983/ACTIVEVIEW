#!/usr/bin/env python3
"""Generate revised reduced15 fixed-30-frame tensors with 16 workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.data.generate_selected16_habitat_parallel import generate


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root / "datasets" / "reduced15_revised_babel_diversity_v1" / "stgcn_development")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-weights", type=Path, default=data_root / "checkpoints" / "ultralytics" / "yolo26n-pose.pt")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if args.workers != 16:
        raise ValueError("revised reduced15 protocol requires exactly 16 workers")
    summary = generate(
        data_root=args.data_root,
        split=args.split,
        workers=args.workers,
        image_size=args.image_size,
        target_frames=args.target_frames,
        device=args.device,
        pose_backend="ultralytics_yolo26n",
        yolo_weights=args.yolo_weights,
        max_records=args.max_records,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
