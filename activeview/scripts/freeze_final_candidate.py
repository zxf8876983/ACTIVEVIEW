"""Freeze an accepted experiment before committing final-test authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.research.experiment import ExperimentStatus
from activeview.research.manifest import load_manifest, resolve_source_path, save_manifest, utc_now, write_status
from activeview.research.registry import get_experiment, update_experiment
from activeview.research.validator import validate_experiment


def freeze_final_candidate(
    experiment_id: str, *, repo_root: Path = REPO_ROOT,
    data_root: Path | None = None,
) -> dict[str, str]:
    """Transition COMPLETED+ACCEPT to tracked FINAL_FROZEN state.

    This command changes Git-tracked experiment files. The caller must commit
    that change before invoking ``authorize_final_test``.
    """
    registry = repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv"
    source_dir = resolve_source_path(get_experiment(registry, experiment_id)["source_dir"], repo_root)
    manifest = load_manifest(source_dir)
    experiment = manifest.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("Manifest experiment section is invalid")
    if experiment.get("status") != "COMPLETED" or experiment.get("decision") != "ACCEPT":
        raise ValueError("Only COMPLETED + ACCEPT experiments can be final-frozen")
    report = validate_experiment(
        source_dir,
        registry_path=registry,
        data_root=data_root or get_data_root(),
    )
    if not report["passed"]:
        raise ValueError(f"Experiment validation failed: {report['errors']}")
    manifest["experiment"]["status"] = ExperimentStatus.FINAL_FROZEN.value
    manifest["experiment"]["frozen_at"] = utc_now()
    manifest["protocol"]["final_model_frozen"] = True
    manifest["protocol"]["test_authorized"] = False
    manifest["protocol"]["test_used"] = False
    save_manifest(source_dir, manifest)
    write_status(source_dir, manifest)
    update_experiment(registry, experiment_id, {"status": ExperimentStatus.FINAL_FROZEN.value})
    return {"experiment_id": experiment_id, "status": ExperimentStatus.FINAL_FROZEN.value, "source_dir": str(source_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(freeze_final_candidate(args.experiment, data_root=args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
