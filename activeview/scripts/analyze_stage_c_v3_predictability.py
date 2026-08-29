#!/usr/bin/env python3
"""Run EXP012's Train-reference/Val-query utility predictability audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_v3_predictability import run_predictability_audit
from activeview.core.paths import get_data_root


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--v0-predictions", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/evaluations/predictions/set_ranker_val.jsonl")
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_predictability_audit(
        feature_root=args.feature_root,
        stage_b_root=args.stage_b_root,
        v0_predictions=args.v0_predictions,
        label_mapping=args.label_mapping,
        output_path=args.output,
    )
    print({"test_used": result["test_used"], "k": result["k"]})


if __name__ == "__main__":
    main()
