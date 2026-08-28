#!/usr/bin/env python3
"""Build the v11.5 action-sample Policy Train/Val/Test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.dataset.policy_split import write_policy_splits


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json")
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input manifest must be a JSON list")
    summary = write_policy_splits(records, args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
