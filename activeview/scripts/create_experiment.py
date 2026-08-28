"""Create an immutable, PLANNED research experiment record."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root, get_stage_experiments_root
from activeview.research.experiment import Experiment
from activeview.research.manifest import git_dirty, git_value, runtime_path_value, source_path_value, utc_now, write_json_atomic, write_status
from activeview.research.provenance import collect_stage_c_research_provenance
from activeview.research.registry import next_experiment_id, register_experiment


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    if not value:
        raise ValueError("name must contain at least one filename-safe character")
    return value


def _hypothesis_text(experiment: Experiment) -> str:
    return f"""# Experiment Hypothesis

## Experiment ID

{experiment.experiment_id}

## Scientific Question

{experiment.hypothesis or "TODO: state the scientific question before starting."}

## Hypothesis

{experiment.hypothesis or "TODO: state one falsifiable hypothesis."}

## Motivation

{experiment.motivation or "TODO: explain the motivation."}

## Baseline

{experiment.baseline or "TODO: name the frozen baseline."}

## Single Core Change

{experiment.core_change or "TODO: specify exactly one core change."}

## Frozen Components

{chr(10).join(f"- {item}" for item in experiment.frozen_items) or "- Stage A, Stage B, Stage C features, frozen ST-GCN, record split"}

## Metrics

{chr(10).join(f"- {item}" for item in experiment.metrics) or "- Accuracy, Macro-F1, regret, headroom capture"}

## Acceptance Criteria

{chr(10).join(f"- {item}" for item in experiment.acceptance_criteria) or "- TODO: define before start."}

## Rejection Criteria

{chr(10).join(f"- {item}" for item in experiment.rejection_criteria) or "- TODO: define before start."}

## Forbidden Changes

- Stage A/B or accepted Stage C-v0 artifacts
- Test evaluation before explicit final authorization
"""


def create_experiment(
    *, stage: str, name: str, hypothesis: str, motivation: str = "", baseline: str = "",
    core_change: str = "", data_root: Path | None = None, repo_root: Path | None = None,
) -> Dict[str, Any]:
    repo = (repo_root or REPO_ROOT).resolve()
    stage_root = repo / "experiments" / stage
    registry_path = stage_root / "EXPERIMENT_REGISTRY.csv"
    experiment_id = next_experiment_id(registry_path)
    directory_name = f"{experiment_id}_{_safe_name(name)}"
    source_dir = stage_root / directory_name
    if source_dir.exists():
        raise FileExistsError(f"Experiment directory already exists: {source_dir}")
    runtime_root = (data_root or get_data_root()).expanduser().resolve()
    runtime_dir = runtime_root / "experiments" / stage / directory_name
    source_value = source_path_value(source_dir, repo)
    runtime_value = runtime_path_value(runtime_dir, runtime_root)
    source_created = False
    runtime_created = False
    registered = False
    try:
        source_dir.mkdir(parents=True)
        source_created = True
        runtime_dir.mkdir(parents=True, exist_ok=False)
        runtime_created = True
        for child in ("checkpoints", "logs", "predictions", "plots", "runtime"):
            (runtime_dir / child).mkdir()
        experiment = Experiment(
            experiment_id=experiment_id, name=name, stage=stage, hypothesis=hypothesis,
            motivation=motivation, baseline=baseline, core_change=core_change,
            frozen_items=["Stage A", "Stage B", "Stage C features", "frozen ST-GCN", "record split"],
            metrics=["Accuracy", "Macro-F1", "mean/median/p90 regret", "positive headroom capture"],
            acceptance_criteria=[], rejection_criteria=[], source_dir=source_value,
            runtime_dir=runtime_value, created_at=utc_now(),
        )
        config_path = source_dir / "config.yaml"
        config_path.write_text("\n".join([
            f"experiment:\n  id: {experiment_id}\n  name: {name}\n  stage: {stage}",
            "baseline:\n  model:\n  checkpoint:",
            "change:\n  category:\n  description:",
            "frozen:\n  stage_a: true\n  stage_b: true\n  stage_c_features: true\n  stgcn_checkpoint: true\n  record_split: true\n  candidate_protocol: true",
            "training:\n  seed:\n  epochs:\n  batch_size:\n  learning_rate:",
            "evaluation:\n  train: true\n  val: true\n  test: false",
            "protocol:\n  test_locked: true\n  test_authorized: false", "",
        ]), encoding="utf-8")
        manifest: Dict[str, Any] = {
            "schema_version": "stage-c-research-v1",
            "experiment": experiment.to_dict(),
            "git": {"creation_commit": git_value("rev-parse", "HEAD", default="unknown"), "start_commit": None, "run_commit": None, "end_commit": None, "dirty_at_creation": git_dirty()},
            "provenance": collect_stage_c_research_provenance(data_root),
            "protocol": {"test_locked": True, "test_used": False, "final_model_frozen": False, "test_authorized": False},
            "training": {"seed": None}, "model": {}, "loss": {}, "sampler": {}, "optimizer": {},
            "paths": {"source_dir": source_value, "runtime_dir": runtime_value, "draft_config_sha256": file_sha256(config_path), "run_config_sha256": None},
        }
        write_json_atomic(source_dir / "run_manifest.json", manifest)
        write_status(source_dir, manifest)
        (source_dir / "hypothesis.md").write_text(_hypothesis_text(experiment), encoding="utf-8")
        (source_dir / "conclusion.md").write_text("# Experiment Conclusion\n\n## Observation\n\nTODO\n\n## Interpretation\n\nTODO\n\n## Decision\n\nNA\n\n## Next\n\nTODO\n\n## Protocol\n\nTest used: false\nFrozen Stage A/B changed: false\nFrozen Stage C features changed: false\n", encoding="utf-8")
        command = "#!/usr/bin/env bash\nset -euo pipefail\n\n# Experiment: " + experiment_id + "\n# Approved training command must be written here after hypothesis/config review.\n# Stage C-v1 Test evaluation is forbidden during development.\n\necho \"No training command configured.\"\nexit 1\n"
        command_path = source_dir / "command.sh"
        command_path.write_text(command, encoding="utf-8")
        command_path.chmod(0o755)
        registry_row = {"experiment_id": experiment_id, "name": name, "stage": stage, "status": "PLANNED", "hypothesis": hypothesis, "core_change": core_change, "baseline": baseline, "created_at": experiment.created_at, "completed_at": "", "git_commit_start": "", "git_commit_end": "", "decision": "NA", "test_used": "false", "source_dir": source_value, "runtime_dir": runtime_value, "notes": "Research experiment created; human review required before start."}
        register_experiment(registry_path, registry_row)
        registered = True
        return {"experiment_id": experiment_id, "source_dir": str(source_dir.resolve()), "runtime_dir": str(runtime_dir.resolve()), "dirty_at_creation": manifest["git"]["dirty_at_creation"], "registry": str(registry_path.resolve())}
    except Exception:
        if not registered:
            if source_created and source_dir.exists():
                shutil.rmtree(source_dir)
            if runtime_created and runtime_dir.exists():
                shutil.rmtree(runtime_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--motivation", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--core-change", default="")
    args = parser.parse_args()
    print(json.dumps(create_experiment(stage=args.stage, name=args.name, hypothesis=args.hypothesis, motivation=args.motivation, baseline=args.baseline, core_change=args.core_change), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
