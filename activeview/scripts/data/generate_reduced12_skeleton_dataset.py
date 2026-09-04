#!/usr/bin/env python3
"""Generate estimated skeleton tensors for the reduced-12 manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.babel_clean_dataset_generator import BabelCleanDatasetGenerator


def main() -> None:
    data_root = get_data_root()
    default_root = data_root / "datasets" / "reduced12_babel_diversity_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("stgcn_development", "activeview"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=data_root / "checkpoints" / "ultralytics" / "yolo26n-pose.pt",
    )
    parser.add_argument(
        "--videopose-weights",
        type=Path,
        default=data_root / "checkpoints" / "videopose3d" / "pretrained_h36m_detectron_coco.bin",
    )
    args = parser.parse_args()
    output_root = args.dataset_root / args.subset
    mapping = json.loads((output_root / "label_mapping.json").read_text(encoding="utf-8"))
    generator = BabelCleanDatasetGenerator(
        output_root=output_root,
        image_size=args.image_size,
        target_frames=args.target_frames,
        seed=args.seed,
        device=args.device,
        label_to_id=mapping,
        yolo_weights=args.yolo_weights,
        videopose_weights=args.videopose_weights,
    )
    splits = ("train", "val") if args.subset == "stgcn_development" else ("train", "val", "test")
    summaries = {}
    for split in splits:
        summaries[split] = generator.generate_split(
            split,
            output_root / f"{split}.json",
            max_records=args.max_records,
        )
    logging.getLogger(__name__).info("Generated reduced12 %s: %s", args.subset, summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
