#!/usr/bin/env python3
"""Prepare the complete cap-100 Official Val manifest for active sensing."""

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
from activeview.data.motion.babel_source_utils import _source_lookup
from activeview.data.motion.reduced12_protocol import (
    REDUCED15_KNEEL_LABELS,
    _with_split,
    collect_reduced12_records,
    select_diverse_records,
)


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=data_root / "datasets" / "reduced15_replacement_babel_diversity_v1" / "activeview_official_val")
    parser.add_argument("--babel-dir", type=Path, default=get_babel_dir())
    parser.add_argument("--amass-index", type=Path, default=data_root / "cache" / "amass_download" / "amass_file_index.json")
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--min-source-frames-exclusive", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    index = json.loads(args.amass_index.read_text(encoding="utf-8"))
    lookup = _source_lookup(index)
    records, _ = collect_reduced12_records(
        args.babel_dir / "val.json",
        "official_val",
        lookup,
        min_frames_exclusive=args.min_source_frames_exclusive,
        labels=REDUCED15_KNEEL_LABELS,
        record_prefix="reduced15k",
    )
    selected = select_diverse_records(
        records,
        cap_per_class=args.cap,
        seed=args.seed + 1,
        labels=REDUCED15_KNEEL_LABELS,
    )
    label_mapping = {label: i for i, label in enumerate(REDUCED15_KNEEL_LABELS)}
    rows = _with_split(selected, "val", label_mapping)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "val.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "protocol": "complete Official Val cap-100 active-sensing manifest",
        "categories": list(REDUCED15_KNEEL_LABELS),
        "records": len(rows),
        "cap_per_class": args.cap,
        "seed": args.seed + 1,
        "source_split": "official_val",
        "test_used": False,
    }
    (args.output_root / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
