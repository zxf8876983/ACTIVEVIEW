#!/usr/bin/env python3
"""Build the diagnostic-only Stage C-v3 future-perception teacher cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_v3_teacher import build_teacher_cache
from activeview.core.paths import get_data_root


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_teacher_cache(
        feature_root=args.feature_root,
        stage_b_root=args.stage_b_root,
        output_dir=args.output_dir,
        splits=("train", "val"),
    )
    print(summary["feature_file_counts"])


if __name__ == "__main__":
    main()
