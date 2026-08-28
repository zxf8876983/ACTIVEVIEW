"""Transition a reviewed PLANNED experiment to RUNNING."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.research.experiment import ExperimentStatus
from activeview.active_view.utility_label_builder import file_sha256
from activeview.research.manifest import experiment_from_manifest, git_dirty, git_value, load_manifest, save_manifest, utc_now, validate_controlled_config, write_status
from activeview.research.provenance import provenance_complete, verify_frozen_provenance
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
    config_path = source_dir / "config.yaml"
    config_errors = validate_controlled_config(config_path, experiment.experiment_id)
    if config_errors:
        raise ValueError("Invalid experiment config: " + ", ".join(config_errors))
    if not experiment.hypothesis.strip() or not experiment.core_change.strip() or "TODO" in (experiment.hypothesis + experiment.core_change):
        raise ValueError("hypothesis and core_change must be completed before start")
    if (source_dir / "final_test_authorization.json").exists():
        raise ValueError("Final authorization must not exist before start")
    provenance = manifest.get("provenance", {})
    provenance_errors = verify_frozen_provenance(provenance)
    if not isinstance(provenance, Mapping) or not provenance_complete(provenance):
        provenance_errors = [*provenance_errors, "frozen_provenance_incomplete"]
    if provenance_errors:
        raise ValueError("Frozen provenance validation failed: " + ", ".join(provenance_errors))
    run_commit = git_value("rev-parse", "HEAD", default="unknown")
    command_path = source_dir / "command.sh"
    manifest["experiment"]["status"] = ExperimentStatus.RUNNING.value
    manifest["experiment"]["started_at"] = utc_now()
    manifest["git"]["start_commit"] = run_commit
    manifest["git"]["run_commit"] = run_commit
    manifest["paths"]["run_config_sha256"] = file_sha256(config_path)
    manifest["paths"]["hypothesis_sha256"] = file_sha256(source_dir / "hypothesis.md")
    manifest["paths"]["command_sha256_at_start"] = file_sha256(command_path)
    save_manifest(source_dir, manifest)
    write_status(source_dir, manifest)
    update_experiment(repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv", experiment_id, {"status": "RUNNING", "git_commit_start": run_commit})
    return {"experiment_id": experiment_id, "status": "RUNNING", "source_dir": str(source_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="Testing-only escape hatch; not recommended for real runs")
    args = parser.parse_args()
    print(json.dumps(start_experiment(args.experiment, require_clean=not args.allow_dirty), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
