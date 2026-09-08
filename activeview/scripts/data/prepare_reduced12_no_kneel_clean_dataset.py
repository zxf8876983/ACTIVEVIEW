#!/usr/bin/env python3
"""Prepare an isolated 12-class raw-train/raw-val protocol.

The protocol is the current reduced14 taxonomy with ``kneel`` and
``clean something`` removed.  Official BABEL Train/Val records are selected
with the existing diversity-first sampler and caps of 300/50 per class.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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
    REDUCED12_NO_KNEEL_CLEAN_LABELS,
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


def build_protocol(
    *,
    output_root: Path,
    babel_dir: Path,
    amass_index: Path,
    train_cap: int = 300,
    val_cap: int = 50,
    min_source_frames_exclusive: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build flat raw-train/raw-val manifests without touching old datasets."""
    labels = REDUCED12_NO_KNEEL_CLEAN_LABELS
    lookup = _source_lookup(json.loads(amass_index.read_text(encoding="utf-8")))
    train_raw, train_excluded = collect_reduced12_records(
        babel_dir / "train.json", "official_train", lookup,
        min_frames_exclusive=min_source_frames_exclusive,
        labels=labels, record_prefix="reduced12kc",
    )
    val_raw, val_excluded = collect_reduced12_records(
        babel_dir / "val.json", "official_val", lookup,
        min_frames_exclusive=min_source_frames_exclusive,
        labels=labels, record_prefix="reduced12kc",
    )
    label_mapping = {label: index for index, label in enumerate(labels)}
    train_selected = select_diverse_records(
        train_raw, cap_per_class=train_cap, seed=seed, labels=labels,
    )
    val_selected = select_diverse_records(
        val_raw, cap_per_class=val_cap, seed=seed + 1, labels=labels,
    )
    stgcn_train, stgcn_val = _split_records(
        train_selected, (0.9, 0.1), seed, labels=labels,
    )
    active_train, active_val, active_test = _split_records(
        val_selected, (0.6, 0.2, 0.2), seed + 1, labels=labels,
    )

    raw_train = output_root / "raw-train"
    raw_val = output_root / "raw-val"
    _write_json(raw_train / "label_mapping.json", label_mapping)
    _write_json(raw_val / "label_mapping.json", label_mapping)
    _write_json(raw_train / "train.json", _with_split(stgcn_train, "train", label_mapping))
    _write_json(raw_train / "val.json", _with_split(stgcn_val, "val", label_mapping))
    _write_json(raw_val / "official_val.json", _with_split(val_selected, "val", label_mapping))
    _write_json(raw_val / "train.json", _with_split(active_train, "train", label_mapping))
    _write_json(raw_val / "val.json", _with_split(active_val, "val", label_mapping))
    _write_json(raw_val / "test.json", _with_split(active_test, "test", label_mapping))

    summary = {
        "protocol": "reduced12 no-kneel/no-clean BABEL diversity protocol",
        "categories": list(labels),
        "label_mapping": label_mapping,
        "seed": seed,
        "caps": {"official_train_per_class": train_cap, "official_val_per_class": val_cap},
        "split_definition": {
            "raw_train": "official BABEL Train selected per class then 90/10 train/val",
            "raw_val": "official BABEL Val selected per class then 60/20/20 train/val/test records",
        },
        "raw_counts": {
            "official_train": dict(Counter(str(item["action_label"]) for item in train_raw)),
            "official_val": dict(Counter(str(item["action_label"]) for item in val_raw)),
        },
        "selected_counts": {
            "raw_train": {
                "train": dict(Counter(str(item["action_label"]) for item in stgcn_train)),
                "val": dict(Counter(str(item["action_label"]) for item in stgcn_val)),
            },
            "raw_val": {
                "official_val": dict(Counter(str(item["action_label"]) for item in val_selected)),
                "train": dict(Counter(str(item["action_label"]) for item in active_train)),
                "val": dict(Counter(str(item["action_label"]) for item in active_val)),
                "test": dict(Counter(str(item["action_label"]) for item in active_test)),
            },
        },
        "diversity": {
            "raw_train": {"train": _diversity(stgcn_train), "val": _diversity(stgcn_val)},
            "raw_val": {
                "official_val": _diversity(val_selected),
                "train": _diversity(active_train),
                "val": _diversity(active_val),
                "test": _diversity(active_test),
            },
        },
        "selection_priority": ["unique source", "subject diversity", "AMASS dataset diversity", "duration-bin diversity"],
        "excluded_records": len(train_excluded) + len(val_excluded),
        "test_used": False,
        "source_files": {
            "official_train": str((babel_dir / "train.json").resolve()),
            "official_val": str((babel_dir / "val.json").resolve()),
            "amass_index": str(amass_index.resolve()),
        },
    }
    _write_json(output_root / "protocol_summary.json", summary)
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=data_root / "datasets" / "reduced12_no_kneel_clean_babel_diversity_v1",
    )
    parser.add_argument("--babel-dir", type=Path, default=get_babel_dir())
    parser.add_argument(
        "--amass-index", type=Path,
        default=data_root / "cache" / "amass_download" / "amass_file_index.json",
    )
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--val-cap", type=int, default=50)
    parser.add_argument("--min-source-frames-exclusive", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_protocol(
        output_root=args.output_root,
        babel_dir=args.babel_dir,
        amass_index=args.amass_index,
        train_cap=args.train_cap,
        val_cap=args.val_cap,
        min_source_frames_exclusive=args.min_source_frames_exclusive,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
