#!/usr/bin/env python3
"""Prepare the independent 15-class BABEL protocol with kneel replacing wave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.asset_paths import get_babel_dir
from activeview.data.motion.reduced12_protocol import build_reduced15_kneel_protocol


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=data_root / "datasets" / "reduced15_replacement_babel_diversity_v1")
    parser.add_argument("--babel-dir", type=Path, default=get_babel_dir())
    parser.add_argument("--amass-index", type=Path, default=data_root / "cache" / "amass_download" / "amass_file_index.json")
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--active-val-cap", type=int, default=100)
    parser.add_argument("--min-source-frames-exclusive", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_reduced15_kneel_protocol(
        output_root=args.output_root,
        babel_dir=args.babel_dir,
        amass_index_path=args.amass_index,
        min_frames_exclusive=args.min_source_frames_exclusive,
        train_cap=args.train_cap,
        active_val_cap=args.active_val_cap,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
