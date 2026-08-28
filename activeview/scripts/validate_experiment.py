"""Validate one controlled research experiment without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.research.registry import get_experiment
from activeview.research.validator import validate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allow-incomplete-provenance", action="store_true")
    args = parser.parse_args()
    registry = args.repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv"
    source_dir = Path(get_experiment(registry, args.experiment)["source_dir"])
    report = validate_experiment(source_dir, registry_path=registry, require_complete_provenance=not args.allow_incomplete_provenance)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
