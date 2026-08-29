#!/usr/bin/env python3
"""Evaluate one Stage C-v2 checkpoint on Val only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup
from activeview.active_view.stage_c_failure_analysis import analyze_rows, load_jsonl, prepare_aligned_rows
from activeview.active_view.stage_c_v2_dataset import StageCV2Dataset, collate_stage_c_v2, load_v2_statistics
from activeview.active_view.stage_c_v2_evaluation import predict_dataset_v2
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import build_utility_predictor
from activeview.core.paths import get_data_root


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def _categories(mapping_path: Path) -> list[str]:
    mapping = _load(mapping_path)
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _comparison(baseline: Dict[str, Any], metrics: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    current = {
        "accuracy": float(metrics["recognition"]["StageC"]["accuracy"]),
        "macro_f1": float(metrics["recognition"]["StageC"]["macro_f1"]),
        "mean_regret": float(metrics["decision_regret"]["mean"]),
        "median_regret": float(metrics["decision_regret"]["median"]),
        "p90_regret": float(metrics["decision_regret"]["p90"]),
        "headroom": float(metrics["positive_headroom_capture"]["aggregate_positive_clipped_ratio"]),
        "c2_rate": float(analysis["failure_taxonomy"]["C2_wrong_high_utility_loss"]["ratio"]),
    }
    baseline_c2 = float(baseline.get("c2_rate", baseline.get("c2_wrong_high_utility_loss_rate")))
    previous = {
        "accuracy": float(baseline["accuracy"]), "macro_f1": float(baseline["macro_f1"]),
        "mean_regret": float(baseline["regret"]["mean"]), "median_regret": float(baseline["regret"].get("median", 0.0)),
        "p90_regret": float(baseline["regret"]["p90"]), "headroom": float(baseline["headroom_capture"]), "c2_rate": baseline_c2,
    }
    return {"baseline": previous, "experiment": current, **{f"{key}_delta": current[key] - previous[key] for key in ("accuracy", "macro_f1", "mean_regret", "p90_regret", "headroom", "c2_rate")}}


def evaluate_val(*, feature_root: Path, source_feature_root: Path, stage_b_root: Path, dataset_root: Path, checkpoint: Path, baseline_path: Path, output_dir: Path, model_type: str, device_name: str, batch_size: int, experiment_id: str = "") -> Dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    feature_summary_path = feature_root / "stage_c_v2_feature_summary.json"
    feature_summary = _load(feature_summary_path)
    stats = load_v2_statistics(feature_root / "stage_c_v2_feature_stats.json")
    dataset = StageCV2Dataset(feature_root, "val", stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stage_c_v2, num_workers=0)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_utility_predictor(model_type, geometry_dim=int(feature_summary["schema"]["candidate_geometry_dim"])).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    predictions = predict_dataset_v2(model, model_type, loader, load_stage_b_lookup(stage_b_root / "utility_labels/val.jsonl"), device)
    categories = _categories(Path(feature_summary["label_mapping"]))
    metrics = evaluate_predictions(predictions, categories)
    stage_a_summary = _load(dataset_root / "stage_a_summary.json")
    aligned = prepare_aligned_rows(
        load_jsonl(stage_a_summary["episode_files"]["val"]),
        load_jsonl(stage_b_root / "utility_labels/val.jsonl"),
        load_jsonl(_load(source_feature_root / "stage_c_feature_summary.json")["feature_files"]["val"]),
        predictions, expected_split="val",
    )
    analysis = analyze_rows(aligned, categories, split="val", model=model_type)
    comparison = _comparison(_load(baseline_path), metrics, analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "val_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    metrics_payload = {
        "protocol": "ACTIVEVIEW Stage C-v2 Val-only evaluation", "experiment_id": experiment_id, "model_type": model_type,
        "split": "val", "test_used": False, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary_sha256": file_sha256(feature_summary_path), "metrics": metrics, "comparison_to_frozen_v0_val": comparison,
        "prediction_file": str(prediction_path.resolve()), "prediction_file_sha256": file_sha256(prediction_path),
    }
    analysis["comparison_to_frozen_v0_val"] = comparison
    analysis["evaluation_protocol"] = {"split": "val", "test_used": False, "stgcn_frozen": True, "source_stage_c_v0_accepted": True}
    _write_json(output_dir / "val_analysis_full.json", analysis)
    _write_json(output_dir / "val_metrics_full.json", metrics_payload)
    return {"output_dir": str(output_dir.resolve()), "split": "val", "test_used": False, "episode_count": len(predictions), "metrics": metrics, "comparison_to_frozen_v0_val": comparison}


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("joint_aware_set_ranker", "candidate_conditioned_attention", "skeleton_policy_transformer"), required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = evaluate_val(feature_root=args.feature_root, source_feature_root=args.source_feature_root, stage_b_root=args.stage_b_root, dataset_root=args.dataset_root, checkpoint=args.checkpoint, baseline_path=args.baseline, output_dir=args.output_dir, model_type=args.model_type, device_name=args.device, batch_size=args.batch_size, experiment_id=args.experiment_id)
    experiment = result["comparison_to_frozen_v0_val"]["experiment"]
    print(json.dumps({**experiment, "test_used": False}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
