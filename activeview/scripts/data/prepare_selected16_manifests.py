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
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.data.motion.babel_selected16_manifest import (
    DEFAULT_AUXILIARY_LABELS,
    build_selected16_manifests,
)


def main() -> None:
    runtime_root = __import__("activeview.core.paths", fromlist=["get_data_root"]).get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=runtime_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed")
    parser.add_argument("--babel-dir", type=Path, default=None)
    parser.add_argument("--amass-index", type=Path, default=None)
    parser.add_argument("--official-mapping", type=Path, default=None)
    parser.add_argument("--train-cap", type=int, default=400)
    parser.add_argument("--val-cap", type=int, default=100)
    parser.add_argument("--min-source-frames-exclusive", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auxiliary-labels",
        nargs=2,
        default=list(DEFAULT_AUXILIARY_LABELS),
        metavar=("AUX_1", "AUX_2"),
        help="Canonical auxiliary labels appended to the official 14 classes (default: lie stumble)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = build_selected16_manifests(
        output_dir=args.output_dir,
        babel_dir=args.babel_dir,
        amass_index_path=args.amass_index,
        official_mapping_path=args.official_mapping,
        train_cap=args.train_cap,
        val_cap=args.val_cap,
        min_frames_exclusive=args.min_source_frames_exclusive,
        seed=args.seed,
        auxiliary_labels=args.auxiliary_labels,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
