#!/usr/bin/env python3
"""Evaluate EXP014 on Val with a complete two-step trajectory simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import StageDDataset, collate_stage_d, load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import (
    build_baseline_trajectories,
    build_fixed_first_oracle,
    build_single_step_oracles,
    build_stage_d_trajectories,
    predict_second_step_dataset,
    summarize_stage_d_methods,
)
from activeview.active_view.stage_d_policy import SequentialObservationRanker
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def evaluate(
    *, cache_root: Path, stage_b_root: Path, checkpoint: Path, v0_predictions: Path,
    label_mapping: Path, output_dir: Path, device_name: str, batch_size: int,
) -> dict[str, Any]:
    summary_path = cache_root / "stage_d_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json")
    dataset = StageDDataset(Path(summary["feature_files"]["val"]), stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stage_d, num_workers=0)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = SequentialObservationRanker().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    second_predictions = predict_second_step_dataset(model, loader, device)
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    v0_rows = load_jsonl(v0_predictions)
    trajectories = build_stage_d_trajectories(stage_b_rows, v0_rows, dataset.rows, second_predictions)
    baselines = build_baseline_trajectories(stage_b_rows, v0_rows)
    oracles = build_single_step_oracles(stage_b_rows)
    fixed_oracle = build_fixed_first_oracle(stage_b_rows, v0_rows, dataset.rows)
    method_rows = {**baselines, "EXP014": trajectories, **oracles, "FixedFirstSecondStepOracle": fixed_oracle}
    categories = _categories(label_mapping)
    metrics = summarize_stage_d_methods(method_rows, categories)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "val_second_step_predictions.jsonl"
    _write_jsonl(prediction_path, second_predictions)
    result = {
        "protocol": "ACTIVEVIEW Stage D EXP014 Val-only trajectory evaluation",
        "split": "val", "test_used": False, "episode_count": len(stage_b_rows),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "cache_summary_sha256": file_sha256(summary_path), "cache_file_sha256": summary["feature_file_sha256"], "cache_stats_sha256": summary["feature_stats_sha256"],
        "v0_predictions": str(v0_predictions.resolve()), "v0_predictions_sha256": file_sha256(v0_predictions),
        "metrics": metrics, "prediction_file": str(prediction_path.resolve()), "prediction_file_sha256": file_sha256(prediction_path),
    }
    (output_dir / "val_metrics_full.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v0-predictions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = evaluate(cache_root=args.cache_root, stage_b_root=args.stage_b_root, checkpoint=args.checkpoint, v0_predictions=args.v0_predictions, label_mapping=args.label_mapping, output_dir=args.output_dir, device_name=args.device, batch_size=args.batch_size)
    print(json.dumps({"test_used": False, "metrics": result["metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
