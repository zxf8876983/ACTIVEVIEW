#!/usr/bin/env python3
"""Evaluate frozen Test baselines alongside the completed Final Test result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.baselines import _by_id, _selected_row, build_baseline_trajectories, build_single_step_oracles
from activeview.active_view.data import load_jsonl
from activeview.active_view.evaluation import summarize_methods
from activeview.core.paths import get_data_root

N_CLASSES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _random_rows(stage_b_rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for record in stage_b_rows:
        candidates = list(record["candidates"])
        actions: list[int | None] = [None] + [int(item["viewpoint_id"]) for item in candidates]
        selected_id = actions[int(rng.integers(0, len(actions)))]
        if selected_id is None:
            output.append(_selected_row(record, None, moves=0, cost=0.0))
            continue
        candidate = _by_id(record)[selected_id]
        output.append(_selected_row(record, selected_id, moves=1, cost=float(candidate["geodesic_distance_m"])))
    return output


def _metric_pair(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    summary = summarize_methods({"method": rows}, [str(index) for index in range(N_CLASSES)])
    recognition = summary["method"]["recognition"]
    return {"accuracy": float(recognition["accuracy"]), "macro_f1": float(recognition["macro_f1"]), "count": int(recognition["n"])}


def evaluate(data_root: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    stage_b_path = data_root / "datasets/policy_v11_5/stage_b/utility_labels/test.jsonl"
    v0_path = data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/test_predictions.jsonl"
    final_result_path = REPO_ROOT / "experiments/stage_d/FINAL_TEST/result.json"
    stage_b_rows = load_jsonl(stage_b_path)
    v0_rows = load_jsonl(v0_path)
    if any(str(row.get("policy_split", "")).lower() != "test" for row in stage_b_rows + v0_rows):
        raise ValueError("all baseline inputs must have explicit policy_split=test")
    stage_b_ids = [str(row["episode_id"]) for row in stage_b_rows]
    v0_ids = [str(row["episode_id"]) for row in v0_rows]
    if set(stage_b_ids) != set(v0_ids) or len(stage_b_ids) != len(v0_ids):
        raise ValueError("Stage-B and frozen-v0 Test episode IDs are not exactly aligned")

    baselines = build_baseline_trajectories(stage_b_rows, v0_rows)
    oracles = build_single_step_oracles(stage_b_rows)
    method_rows = {
        "NoMove": baselines["NoMove"],
        "Random": _random_rows(stage_b_rows, seed),
        "FrozenStageCv0": baselines["FrozenStageCv0"],
        "SafeOracle": oracles["SafeOracle"],
    }
    moving_ids = {str(row["episode_id"]) for row in v0_rows if not bool(row["predicted_stays"])}
    moving = {name: [row for row in rows if str(row["episode_id"]) in moving_ids] for name, rows in method_rows.items()}
    full_metrics = {name: _metric_pair(rows) for name, rows in method_rows.items()}
    moving_metrics = {name: _metric_pair(rows) for name, rows in moving.items()}
    final = json.loads(final_result_path.read_text(encoding="utf-8"))
    full_metrics["Multi-positive H2"] = final["methods"]["MULTI_POSITIVE_JR_H2"]["full"]
    moving_metrics["Multi-positive H2"] = final["methods"]["MULTI_POSITIVE_JR_H2"]["moving"]

    def delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
        return {metric: float(left[metric]) - float(right[metric]) for metric in ("accuracy", "macro_f1")}

    result: dict[str, Any] = {
        "experiment_id": "FINAL_TEST_BASELINES",
        "status": "COMPLETED",
        "split": "test",
        "test_used": True,
        "training_performed": False,
        "random_seed": seed,
        "population": {"full": len(stage_b_rows), "moving_subset": len(moving_ids), "moving_definition": "frozen Stage-C-v0 predicted_stays=false"},
        "methods": {"full": full_metrics, "moving": moving_metrics},
        "deltas": {
            "MULTI_MINUS_FROZEN_STAGE_CV0": {"full": delta(full_metrics["Multi-positive H2"], full_metrics["FrozenStageCv0"]), "moving": delta(moving_metrics["Multi-positive H2"], moving_metrics["FrozenStageCv0"])},
            "SAFE_ORACLE_MINUS_FROZEN_STAGE_CV0": {"full": delta(full_metrics["SafeOracle"], full_metrics["FrozenStageCv0"]), "moving": delta(moving_metrics["SafeOracle"], moving_metrics["FrozenStageCv0"])},
        },
        "provenance": {"stage_b_test": str(stage_b_path.resolve()), "stage_b_test_sha256": _sha256(stage_b_path), "v0_test_predictions": str(v0_path.resolve()), "v0_test_predictions_sha256": _sha256(v0_path), "final_test_result": str(final_result_path.resolve()), "final_test_result_sha256": _sha256(final_result_path)},
        "leakage_flags": {"habitat_rendering_performed": False, "perception_regenerated": False, "model_training_performed": False, "test_used": True},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    names = ["NoMove", "Random", "FrozenStageCv0", "SafeOracle", "Multi-positive H2"]
    lines = ["# Final Test Baselines", "", "| Method | Full Accuracy | Full Macro-F1 | Moving Accuracy | Moving Macro-F1 |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {name} | {full_metrics[name]['accuracy']:.6f} | {full_metrics[name]['macro_f1']:.6f} | {moving_metrics[name]['accuracy']:.6f} | {moving_metrics[name]['macro_f1']:.6f} |" for name in names)
    lines.extend(["", "Moving subset is exactly the frozen Stage-C-v0 `predicted_stays=false` episode set.", "", f"- Multi-positive H2 − FrozenStageCv0 (full): ΔAccuracy {result['deltas']['MULTI_MINUS_FROZEN_STAGE_CV0']['full']['accuracy']:+.6f}, ΔMacro-F1 {result['deltas']['MULTI_MINUS_FROZEN_STAGE_CV0']['full']['macro_f1']:+.6f}.", f"- Multi-positive H2 − FrozenStageCv0 (moving): ΔAccuracy {result['deltas']['MULTI_MINUS_FROZEN_STAGE_CV0']['moving']['accuracy']:+.6f}, ΔMacro-F1 {result['deltas']['MULTI_MINUS_FROZEN_STAGE_CV0']['moving']['macro_f1']:+.6f}.", f"- SafeOracle − FrozenStageCv0 (full): ΔAccuracy {result['deltas']['SAFE_ORACLE_MINUS_FROZEN_STAGE_CV0']['full']['accuracy']:+.6f}, ΔMacro-F1 {result['deltas']['SAFE_ORACLE_MINUS_FROZEN_STAGE_CV0']['full']['macro_f1']:+.6f}.", f"- SafeOracle − FrozenStageCv0 (moving): ΔAccuracy {result['deltas']['SAFE_ORACLE_MINUS_FROZEN_STAGE_CV0']['moving']['accuracy']:+.6f}, ΔMacro-F1 {result['deltas']['SAFE_ORACLE_MINUS_FROZEN_STAGE_CV0']['moving']['macro_f1']:+.6f}.", "", "No Test data were regenerated; this is an offline evaluation of frozen artifacts."])
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/stage_d/FINAL_TEST_baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_root = (args.data_root or get_data_root()).resolve()
    print(json.dumps(evaluate(data_root, args.output_dir.resolve(), args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
