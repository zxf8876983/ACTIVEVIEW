#!/usr/bin/env python3
"""文件用途：
    执行离线数据生成、划分或缓存构建入口。

主要输入：
    - 命令行参数与已有运行时数据。
主要输出：
    - 数据集、缓存或清单文件。
项目角色：
    - 属于 data 脚本入口，仅调用正式数据模块。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.babel_clean_dataset_generator import BabelCleanDatasetGenerator


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
