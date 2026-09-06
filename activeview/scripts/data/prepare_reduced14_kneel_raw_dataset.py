#!/usr/bin/env python3
"""Prepare cap-300/100 raw-train/raw-val manifests for the 14-class protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.asset_paths import get_babel_dir
from activeview.data.motion.babel_source_utils import _source_lookup
from activeview.data.motion.reduced12_protocol import (
    REDUCED14_KNEEL_LABELS,
    _duration_bin,
    _split_records,
    _with_split,
    collect_reduced12_records,
    select_diverse_records,
)


def _diversity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    durations = [float(item["duration_seconds"]) for item in records]
    return {
        "records": len(records),
        "unique_sources": len({str(item["source_group"]) for item in records}),
        "unique_subjects": len({str(item["subject_id"]) for item in records}),
        "unique_amass_datasets": len({str(item["amass_dataset"]) for item in records}),
        "duration_bins": dict(Counter(_duration_bin(value) for value in durations)),
        "duration_seconds_total": float(sum(durations)),
        "duration_seconds_mean": float(sum(durations) / len(durations)) if durations else 0.0,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=data_root / "datasets" / "reduced14_kneel_babel_diversity_v1",
    )
    parser.add_argument("--babel-dir", type=Path, default=get_babel_dir())
    parser.add_argument(
        "--amass-index", type=Path,
        default=data_root / "cache" / "amass_download" / "amass_file_index.json",
    )
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--val-cap", type=int, default=100)
    parser.add_argument("--min-source-frames-exclusive", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lookup = _source_lookup(json.loads(args.amass_index.read_text(encoding="utf-8")))
    train_raw, train_excluded = collect_reduced12_records(
        args.babel_dir / "train.json", "official_train", lookup,
        min_frames_exclusive=args.min_source_frames_exclusive,
        labels=REDUCED14_KNEEL_LABELS, record_prefix="reduced14k",
    )
    val_raw, val_excluded = collect_reduced12_records(
        args.babel_dir / "val.json", "official_val", lookup,
        min_frames_exclusive=args.min_source_frames_exclusive,
        labels=REDUCED14_KNEEL_LABELS, record_prefix="reduced14k",
    )
    label_mapping = {label: i for i, label in enumerate(REDUCED14_KNEEL_LABELS)}
    train_selected = select_diverse_records(
        train_raw, cap_per_class=args.train_cap, seed=args.seed,
        labels=REDUCED14_KNEEL_LABELS,
    )
    val_selected = select_diverse_records(
        val_raw, cap_per_class=args.val_cap, seed=args.seed + 1,
        labels=REDUCED14_KNEEL_LABELS,
    )
    stgcn_train, stgcn_val = _split_records(
        train_selected, (0.9, 0.1), args.seed, labels=REDUCED14_KNEEL_LABELS,
    )
    active_train, active_val, active_test = _split_records(
        val_selected, (0.6, 0.2, 0.2), args.seed + 1, labels=REDUCED14_KNEEL_LABELS,
    )

    raw_train = args.output_root / "raw-train"
    raw_val = args.output_root / "raw-val"
    _write_json(raw_train / "label_mapping.json", label_mapping)
    _write_json(raw_val / "label_mapping.json", label_mapping)
    _write_json(raw_train / "train.json", _with_split(stgcn_train, "train", label_mapping))
    _write_json(raw_train / "val.json", _with_split(stgcn_val, "val", label_mapping))
    _write_json(raw_val / "official_val.json", _with_split(val_selected, "val", label_mapping))
    _write_json(raw_val / "train.json", _with_split(active_train, "train", label_mapping))
    _write_json(raw_val / "val.json", _with_split(active_val, "val", label_mapping))
    _write_json(raw_val / "test.json", _with_split(active_test, "test", label_mapping))

    summary = {
        "protocol": "reduced14 kneel diversity-aware BABEL protocol",
        "categories": list(REDUCED14_KNEEL_LABELS),
        "label_mapping": label_mapping,
        "seed": args.seed,
        "caps": {"official_train_per_class": args.train_cap, "official_val_per_class": args.val_cap},
        "raw_train": {
            "train": len(stgcn_train), "val": len(stgcn_val),
            "diversity": {"train": _diversity(stgcn_train), "val": _diversity(stgcn_val)},
        },
        "raw_val": {
            "official_val": len(val_selected), "train": len(active_train),
            "val": len(active_val), "test": len(active_test),
            "diversity": {
                "official_val": _diversity(val_selected), "train": _diversity(active_train),
                "val": _diversity(active_val), "test": _diversity(active_test),
            },
        },
        "selection_priority": ["unique source", "subject diversity", "AMASS dataset diversity", "duration-bin diversity"],
        "excluded_records": len(train_excluded) + len(val_excluded),
        "test_used": False,
        "source_files": {
            "official_train": str((args.babel_dir / "train.json").resolve()),
            "official_val": str((args.babel_dir / "val.json").resolve()),
            "amass_index": str(args.amass_index.resolve()),
        },
    }
    _write_json(args.output_root / "protocol_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
