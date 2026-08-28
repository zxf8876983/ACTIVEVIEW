"""Validate that Phase 0 research infrastructure is present and empty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.research.registry import REGISTRY_FIELDS


def validate_infrastructure(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    errors: list[str] = []
    stage_root = repo_root / "experiments" / "stage_c_v1"
    required = [repo_root / ".ai" / name for name in ("RESEARCH_PLAN.md", "RESEARCH_LOG.md", "REJECTED_IDEAS.md")]
    required += [stage_root / "README.md", stage_root / "EXPERIMENT_REGISTRY.csv"]
    required += [stage_root / "templates" / name for name in ("hypothesis.md", "conclusion.md", "config.yaml")]
    required += [
        repo_root / "activeview" / "research" / "test_gate.py",
        repo_root / "tests" / "unit" / "test_research_test_gate.py",
        repo_root / "tests" / "integration" / "test_research_experiment_lifecycle.py",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing:{path.relative_to(repo_root)}")
    registry = stage_root / "EXPERIMENT_REGISTRY.csv"
    rows: list[dict[str, str]] = []
    if registry.is_file():
        with registry.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != REGISTRY_FIELDS:
                errors.append("invalid_registry_header")
            rows = list(reader)
        if rows:
            errors.append("registry_not_empty")
    experiment_dirs = [path for path in stage_root.glob("EXP*_*" ) if path.is_dir()]
    if experiment_dirs:
        errors.append("real_experiment_directories_exist")
    config = stage_root / "templates" / "config.yaml"
    if config.is_file() and "test: false" not in config.read_text(encoding="utf-8"):
        errors.append("template_test_not_locked")
    return {"passed": not errors, "error_count": len(errors), "errors": errors, "registry_empty": not rows, "real_experiments": len(experiment_dirs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_infrastructure(args.repo_root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
