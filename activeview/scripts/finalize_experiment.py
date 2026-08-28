"""Finalize a RUNNING experiment with a retained decision and Val metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.research.experiment import Decision, ExperimentStatus
from activeview.core.paths import get_data_root
from activeview.research.manifest import git_value, load_manifest, resolve_source_path, save_manifest, utc_now, write_status
from activeview.research.registry import get_experiment, update_experiment
from activeview.research.validator import validate_experiment


def finalize_experiment(
    experiment_id: str, *, decision: str, repo_root: Path = REPO_ROOT,
    data_root: Path | None = None,
) -> dict:
    if decision not in {item.value for item in Decision if item is not Decision.NA}:
        raise ValueError("decision must be ACCEPT, REJECT or INCONCLUSIVE")
    registry = repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv"
    source_dir = resolve_source_path(get_experiment(registry, experiment_id)["source_dir"], repo_root)
    manifest = load_manifest(source_dir)
    if str(manifest.get("experiment", {}).get("status")) != ExperimentStatus.RUNNING.value:
        raise ValueError("Only RUNNING experiments can be finalized")
    required = [source_dir / filename for filename in ("val_metrics.json", "analysis.json", "conclusion.md")]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completion files: {missing}")
    preflight = validate_experiment(source_dir, registry_path=registry, data_root=data_root or get_data_root())
    if not preflight["passed"]:
        raise ValueError(f"Pre-finalization validation failed: {preflight['errors']}")
    manifest["experiment"]["status"] = ExperimentStatus.COMPLETED.value
    manifest["experiment"]["decision"] = decision
    manifest["experiment"]["completed_at"] = utc_now()
    manifest["git"]["end_commit"] = git_value("rev-parse", "HEAD", default="unknown")
    save_manifest(source_dir, manifest)
    write_status(source_dir, manifest)
    metrics: dict = {}
    try:
        metrics = json.loads((source_dir / "val_metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    recognition = metrics.get("recognition", {}) if isinstance(metrics, dict) else {}
    regret = metrics.get("regret", {}) if isinstance(metrics, dict) else {}
    updates = {"status": "COMPLETED", "decision": decision, "completed_at": manifest["experiment"]["completed_at"], "git_commit_end": manifest["git"]["end_commit"], "val_accuracy": recognition.get("accuracy", ""), "val_macro_f1": recognition.get("macro_f1", ""), "val_mean_regret": regret.get("mean", ""), "val_median_regret": regret.get("median", ""), "val_p90_regret": regret.get("p90", ""), "val_headroom_capture": metrics.get("positive_headroom_capture", "") if isinstance(metrics, dict) else ""}
    update_experiment(registry, experiment_id, updates)
    return {"experiment_id": experiment_id, "status": "COMPLETED", "decision": decision, "source_dir": str(source_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--decision", required=True, choices=("ACCEPT", "REJECT", "INCONCLUSIVE"))
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(finalize_experiment(args.experiment, decision=args.decision, data_root=args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
