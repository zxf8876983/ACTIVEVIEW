"""Fail-closed prerequisite gate for the EXP051-R1 closed-loop rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_NAMES = (
    "joint_revision_final.pth",
    "joint_revision_best.pth",
    "checkpoint.pth",
    "model.pth",
)


def _candidate_paths(repo_root: Path) -> list[Path]:
    roots = (
        repo_root / "experiments/stage_d/EXP050_joint_rollout_revision",
        repo_root.parent.parent / "data/ActiveView/experiments/stage_d/EXP050_joint_rollout_revision",
        Path("/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP050_joint_rollout_revision"),
    )
    return [root / name for root in roots for name in EXPECTED_CHECKPOINT_NAMES]


def build_result(repo_root: Path) -> dict[str, Any]:
    candidates = _candidate_paths(repo_root)
    existing = [str(path) for path in candidates if path.is_file()]
    status = "READY_FOR_GATE" if existing else "BLOCKED_MISSING_JOINT_REVISION_CHECKPOINT"
    return {
        "experiment_id": "EXP051-R1",
        "status": status,
        "checkpoint_candidates_checked": [str(path) for path in candidates],
        "joint_revision_checkpoint_paths_found": existing,
        "test_used": False,
        "training_performed": False,
        "wm_e_frozen": True,
        "joint_revision_frozen": True,
        "stgcn_frozen": True,
        "habitat_rendering_performed": False,
        "perception_regenerated": False,
        "history_shift_executed": False,
        "h2_rollout_executed": False,
        "reason": (
            "EXP050 Joint Revision checkpoint is required for exact H1 reproduction; "
            "it is absent, and retraining is forbidden."
            if not existing
            else "Checkpoint discovered; proceed to the separate exact reproduction gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_result(args.repo_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"EXP051_R1={result['status']}")
    return 0 if result["status"] == "READY_FOR_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
