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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.splits.policy_split import SPLITS, write_policy_splits, write_predefined_policy_splits


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-split-dir", type=Path, default=data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-val")
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.input_manifest is not None:
        records = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Input manifest must be a JSON list")
        summary = write_policy_splits(records, args.output_dir, seed=args.seed)
    else:
        split_records = {
            split: json.loads((args.input_split_dir / f"{split}.json").read_text(encoding="utf-8"))
            for split in SPLITS
        }
        if any(not isinstance(rows, list) for rows in split_records.values()):
            raise ValueError("Every predefined split manifest must be a JSON list")
        summary = write_predefined_policy_splits(split_records, args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
