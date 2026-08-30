#!/usr/bin/env python3
"""Run EXP015's fixed-first sequential budget/oracle analysis on Val only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_evaluation import (
    build_baseline_trajectories,
    build_fixed_first_oracle,
    build_single_step_oracles,
    build_stage_d_trajectories,
    summarize_stage_d_methods,
)
from activeview.active_view.utility_label_builder import file_sha256


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def analyze(
    *, cache_root: Path, stage_b_root: Path, exp014_predictions: Path,
    v0_predictions: Path, label_mapping: Path, output_path: Path,
) -> dict[str, Any]:
    cache_summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text(encoding="utf-8"))
    cache_rows = load_jsonl(Path(cache_summary["feature_files"]["val"]))
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_rows = load_jsonl(v0_predictions)
    exp014_rows = load_jsonl(exp014_predictions)
    trajectories = build_stage_d_trajectories(stage_b_rows, v0_rows, cache_rows, exp014_rows)
    baseline = build_baseline_trajectories(stage_b_rows, v0_rows)
    oracles = build_single_step_oracles(stage_b_rows)
    fixed_oracle = build_fixed_first_oracle(stage_b_rows, v0_rows, cache_rows)
    categories = _categories(label_mapping)
    methods = {**baseline, "EXP014": trajectories, **oracles, "FixedFirstSecondStepOracle": fixed_oracle}
    metrics = summarize_stage_d_methods(methods, categories)

    v0_by_id = {str(row["episode_id"]): row for row in v0_rows}
    stage_b_by_id = {str(row["episode_id"]): row for row in stage_b_rows}
    cache_by_id = {str(row["episode_id"]): row for row in cache_rows}
    exp_by_id = {str(row["episode_id"]): row for row in exp014_rows}
    second_total = 0
    second_exact = 0
    second_stay_true = second_stay_pred = 0
    second_move_pred = 0
    second_stay_tp = second_stay_fp = second_stay_fn = 0
    second_candidate_exact = 0
    for episode_id, second in exp_by_id.items():
        cached = cache_by_id[episode_id]
        record = stage_b_by_id[episode_id]
        values = [0.0] + [float(value) for value in cached["second_step_utility_targets"]]
        oracle_index = int(max(range(len(values)), key=lambda index: values[index]))
        oracle_stays = oracle_index == 0
        pred_stays = bool(second["predicted_stays"])
        second_total += 1
        second_exact += int(pred_stays == oracle_stays and (pred_stays or int(second["predicted_candidate_viewpoint_id"]) == int(cached["remaining_candidate_ids"][oracle_index - 1])))
        second_stay_tp += int(pred_stays and oracle_stays)
        second_stay_fp += int(pred_stays and not oracle_stays)
        second_stay_fn += int((not pred_stays) and oracle_stays)
        second_candidate_exact += int(not oracle_stays and not pred_stays and int(second["predicted_candidate_viewpoint_id"]) == int(cached["remaining_candidate_ids"][oracle_index - 1]))
        second_stay_true += int(oracle_stays)
        second_stay_pred += int(pred_stays)
        second_move_pred += int(not pred_stays)

    v0_stay = []
    for row in stage_b_rows:
        prediction = v0_by_id[str(row["episode_id"])]
        if bool(prediction["predicted_stays"]):
            v0_stay.append(row)
    missed = sum(int(not bool(row["oracle"]["safe_oracle_stays"])) for row in v0_stay)
    decomposition = {"A_v0_stay": 0, "B_v0_move_exp014_stay": 0, "C_v0_move_exp014_move": 0}
    decomposition_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in decomposition}
    for row in trajectories:
        first = v0_by_id[str(row["episode_id"])]
        if bool(first["predicted_stays"]):
            decomposition["A_v0_stay"] += 1
            decomposition_rows["A_v0_stay"].append(row)
        elif int(row["moves"]) == 1:
            decomposition["B_v0_move_exp014_stay"] += 1
            decomposition_rows["B_v0_move_exp014_stay"].append(row)
        else:
            decomposition["C_v0_move_exp014_move"] += 1
            decomposition_rows["C_v0_move_exp014_move"].append(row)
    decomposition_metrics = {
        key: {
            "count": len(rows),
            "accuracy": metrics["EXP014"]["recognition"]["accuracy"] if not rows else summarize_stage_d_methods({"group": rows}, categories)["group"]["recognition"]["accuracy"],
            "mean_regret": 0.0 if not rows else summarize_stage_d_methods({"group": rows}, categories)["group"]["decision_regret"]["mean"],
        }
        for key, rows in decomposition_rows.items()
    }
    result = {
        "protocol": "ACTIVEVIEW Stage D EXP015 fixed-first sequential budget/oracle analysis",
        "split": "val", "test_used": False, "episode_count": len(stage_b_rows),
        "methods": metrics,
        "fixed_first_second_step_oracle_definition": "Frozen v0 first Move/Stay and Top-1 action; then argmax of Stay utility 0 and true U2(p2|s1), U2(p3|s1)",
        "initial_stay_ceiling": {
            "v0_stay_episode_count": len(v0_stay), "v0_stay_but_safe_oracle_moves": missed,
            "initial_missed_move_rate": missed / len(v0_stay) if v0_stay else 0.0,
            "missed_move_safe_utility_mean": (
                float(sum(float(row["oracle"]["safe_oracle_utility"]) for row in v0_stay if not bool(row["oracle"]["safe_oracle_stays"])) / missed)
                if missed else 0.0
            ),
        },
        "second_step_quality": {
            "eligible_episode_count": second_total,
            "action_match_rate": second_exact / second_total if second_total else 0.0,
            "oracle_stay_rate": second_stay_true / second_total if second_total else 0.0,
            "predicted_stay_rate": second_stay_pred / second_total if second_total else 0.0,
            "predicted_move_rate": second_move_pred / second_total if second_total else 0.0,
            "stay_precision": second_stay_tp / (second_stay_tp + second_stay_fp) if second_stay_tp + second_stay_fp else 0.0,
            "stay_recall": second_stay_tp / (second_stay_tp + second_stay_fn) if second_stay_tp + second_stay_fn else 0.0,
            "candidate_exact_hit_rate_move_only": second_candidate_exact / max(second_total - second_stay_true, 1),
        },
        "stage_d_decomposition": {"counts": decomposition, "metrics": decomposition_metrics},
        "provenance": {
            "cache_summary_sha256": file_sha256(cache_root / "stage_d_feature_summary.json"),
            "stage_d_val_feature_sha256": cache_summary["feature_file_sha256"]["val"],
            "stage_b_val_sha256": file_sha256(stage_b_root / "utility_labels" / "val.jsonl"),
            "v0_val_predictions_sha256": file_sha256(v0_predictions),
            "exp014_val_predictions_sha256": file_sha256(exp014_predictions),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--exp014-predictions", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(cache_root=args.cache_root, stage_b_root=args.stage_b_root, exp014_predictions=args.exp014_predictions, v0_predictions=args.v0_predictions, label_mapping=args.label_mapping, output_path=args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
