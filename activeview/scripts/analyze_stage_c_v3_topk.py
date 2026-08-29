#!/usr/bin/env python3
"""Run EXP013's Top-K reachability audit on frozen Stage C-v0 Val predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_v3_topk import run_topk_audit
from activeview.core.paths import get_data_root


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/evaluations/predictions/set_ranker_val.jsonl")
    parser.add_argument("--stage-b-utility", type=Path, default=data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_topk_audit(predictions_path=args.predictions, stage_b_utility_path=args.stage_b_utility, output_path=args.output)
    print({"test_used": result["test_used"], "episode_count": result["episode_count"]})


if __name__ == "__main__":
    main()
