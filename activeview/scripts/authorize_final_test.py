"""Create explicit authorization for one final Test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root
from activeview.research.manifest import git_dirty, git_value, load_manifest, manifest_sha256, resolve_runtime_path, resolve_source_path, utc_now, write_json_atomic
from activeview.research.registry import get_experiment
from activeview.research.validator import validate_experiment


def authorize_final_test(
    experiment_id: str, *, repo_root: Path = REPO_ROOT, confirm_final_model_frozen: bool = False,
    require_clean: bool = True, data_root: Path | None = None,
) -> dict:
    if not confirm_final_model_frozen:
        raise PermissionError("Explicit --confirm-final-model-frozen is required")
    registry = repo_root / "experiments" / "stage_c_v1" / "EXPERIMENT_REGISTRY.csv"
    runtime_root = (data_root or get_data_root()).expanduser().resolve()
    row = get_experiment(registry, experiment_id)
    source_dir = resolve_source_path(row["source_dir"], repo_root)
    manifest = load_manifest(source_dir)
    if manifest.get("experiment", {}).get("status") != "FINAL_FROZEN" or manifest.get("experiment", {}).get("decision") != "ACCEPT":
        raise ValueError("Only FINAL_FROZEN + ACCEPT experiments can be authorized")
    runtime_dir = resolve_runtime_path(manifest.get("experiment", {}).get("runtime_dir", row["runtime_dir"]), runtime_root)
    report = validate_experiment(source_dir, registry_path=registry, data_root=runtime_root)
    if not report["passed"]:
        raise ValueError(f"Experiment validation failed: {report['errors']}")
    if require_clean and git_dirty():
        raise RuntimeError("Working tree must be clean before final Test authorization")
    config_path = source_dir / "config.yaml"
    authorization = {
        "experiment_id": experiment_id, "authorized": True,
        "frozen_git_commit": git_value("rev-parse", "HEAD", default="unknown"),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256_before_authorization": manifest_sha256(source_dir),
        "authorized_at": utc_now(),
    }
    authorization_path = runtime_dir / "final_test_authorization.json"
    write_json_atomic(authorization_path, authorization)
    return {"experiment_id": experiment_id, "status": "FINAL_FROZEN", "authorization": str(authorization_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--confirm-final-model-frozen", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(authorize_final_test(args.experiment, confirm_final_model_frozen=args.confirm_final_model_frozen, data_root=args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
