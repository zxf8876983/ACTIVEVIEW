#!/usr/bin/env python3
"""Evaluate the diagnostic-only future-perception teacher on Val only."""

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

from activeview.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup
from activeview.active_view.stage_c_v3_teacher import (
    FutureTeacherDataset,
    collate_future_teacher,
    load_teacher_stats,
    load_jsonl,
    predict_teacher_dataset,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import FuturePerceptionTeacherMLP


def _categories(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def evaluate_val(
    *, cache_root: Path, stage_b_root: Path, checkpoint: Path,
    label_mapping: Path | None, output_dir: Path, device_name: str,
    batch_size: int, baseline_predictions: Path | None = None,
) -> dict[str, Any]:
    summary = json.loads((cache_root / "stage_c_v3_teacher_summary.json").read_text(encoding="utf-8"))
    if label_mapping is None:
        label_mapping = Path(summary["label_mapping"])
    stats = load_teacher_stats(cache_root / "teacher_feature_stats.json")
    dataset = FutureTeacherDataset(Path(summary["feature_files"]["val"]), stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_future_teacher, num_workers=0)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = FuturePerceptionTeacherMLP().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    stage_b_lookup = load_stage_b_lookup(stage_b_root / "utility_labels/val.jsonl")
    predictions = predict_teacher_dataset(model, loader, stage_b_lookup, device)
    metrics = evaluate_predictions(predictions, _categories(label_mapping))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "val_predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    result = {
        "protocol": "ACTIVEVIEW Stage C-v3 future-perception teacher Val-only diagnostic",
        "diagnostic_only": True, "deployable_policy": False,
        "future_candidate_perception_used": True, "split": "val", "test_used": False,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "teacher_summary_sha256": file_sha256(cache_root / "stage_c_v3_teacher_summary.json"),
        "prediction_file": str(prediction_path.resolve()), "metrics": metrics,
    }
    if baseline_predictions is not None:
        baseline_rows = load_jsonl(baseline_predictions)
        result["frozen_v0_val_metrics"] = evaluate_predictions(baseline_rows, _categories(label_mapping))
        result["frozen_v0_val_predictions_sha256"] = file_sha256(baseline_predictions)
    (output_dir / "val_metrics_full.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = evaluate_val(
        cache_root=args.cache_root, stage_b_root=args.stage_b_root,
        checkpoint=args.checkpoint, label_mapping=args.label_mapping,
        output_dir=args.output_dir, device_name=args.device, batch_size=args.batch_size,
        baseline_predictions=args.baseline_predictions,
    )
    print(json.dumps({"test_used": False, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
