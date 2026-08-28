"""Transition a reviewed PLANNED experiment to RUNNING."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_stage_experiments_root
from activeview.research.experiment import ExperimentStatus
from activeview.research.manifest import experiment_from_manifest, git_dirty, load_manifest, save_manifest, utc_now, write_status
from activeview.research.registry import get_experiment, update_experiment


def _source_dir(experiment_id: str, repo_root: Path) -> Path:
    registry = repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv"
    row = get_experiment(registry, experiment_id)
    return Path(row["source_dir"])


def start_experiment(experiment_id: str, *, repo_root: Path = REPO_ROOT, require_clean: bool = True) -> dict[str, str]:
    source_dir = _source_dir(experiment_id, repo_root)
    manifest = load_manifest(source_dir)
    experiment = experiment_from_manifest(manifest)
    if str(experiment.status) != ExperimentStatus.PLANNED.value:
        raise ValueError(f"Only PLANNED experiments can start: {experiment.status}")
    if require_clean and git_dirty():
        raise RuntimeError("Working tree must be clean before starting an experiment")
    manifest["experiment"]["status"] = ExperimentStatus.RUNNING.value
    manifest["experiment"]["started_at"] = utc_now()
    save_manifest(source_dir, manifest)
    write_status(source_dir, manifest)
    update_experiment(repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv", experiment_id, {"status": "RUNNING"})
    return {"experiment_id": experiment_id, "status": "RUNNING", "source_dir": str(source_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="Testing-only escape hatch; not recommended for real runs")
    args = parser.parse_args()
    print(json.dumps(start_experiment(args.experiment, require_clean=not args.allow_dirty), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
