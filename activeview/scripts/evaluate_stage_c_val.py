#!/usr/bin/env python3
"""Run the fixed Train→Val evaluation contract for one Stage C-v1 experiment.

This entry point intentionally evaluates Val only.  Test evaluation remains a
separately authorized final step and is not reachable through this command.
"""

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

from activeview.active_view.stage_c_dataset import (
    EpisodeFeatureDataset,
    collate_episode_batch,
    load_feature_statistics,
)
from activeview.active_view.stage_c_evaluation import (
    evaluate_predictions,
    load_stage_b_lookup,
    predict_dataset,
)
from activeview.active_view.stage_c_failure_analysis import (
    analyze_rows,
    load_jsonl,
    prepare_aligned_rows,
)
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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def _categories(feature_root: Path) -> list[str]:
    mapping_path = (
        feature_root.parent.parent
        / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"
        / "label_mapping.json"
    )
    mapping = _load(mapping_path)
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _comparison(
    baseline: Dict[str, Any],
    metrics: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare generic Val metrics without applying experiment-specific criteria.

    ``analysis`` is used only to obtain the C2 diagnostic, which is not part of
    the generic recognition/regret summary.  Acceptance decisions remain in
    each experiment's README and are intentionally not emitted here.
    """
    current_c2 = float(
        analysis["failure_taxonomy"]["C2_wrong_high_utility_loss"]["ratio"]
    )
    baseline_c2 = _baseline_c2_rate(baseline)
    current = {
        "accuracy": float(metrics["recognition"]["StageC"]["accuracy"]),
        "macro_f1": float(metrics["recognition"]["StageC"]["macro_f1"]),
        "mean_regret": float(metrics["decision_regret"]["mean"]),
        "median_regret": float(metrics["decision_regret"]["median"]),
        "p90_regret": float(metrics["decision_regret"]["p90"]),
        "headroom": float(
            metrics["positive_headroom_capture"][
                "aggregate_positive_clipped_ratio"
            ]
        ),
        "c2_rate": current_c2,
    }
    baseline_values = {
        "accuracy": float(baseline["accuracy"]),
        "macro_f1": float(baseline["macro_f1"]),
        "mean_regret": float(baseline["regret"]["mean"]),
        "median_regret": (
            float(baseline["regret"]["median"])
            if "median" in baseline.get("regret", {})
            else None
        ),
        "p90_regret": float(baseline["regret"]["p90"]),
        "headroom": float(baseline["headroom_capture"]),
        "c2_rate": baseline_c2,
    }
    return {
        "baseline": baseline_values,
        "experiment": current,
        "accuracy_delta": current["accuracy"] - baseline_values["accuracy"],
        "macro_f1_delta": current["macro_f1"] - baseline_values["macro_f1"],
        "mean_regret_delta": current["mean_regret"] - baseline_values["mean_regret"],
        "p90_regret_delta": current["p90_regret"] - baseline_values["p90_regret"],
        "headroom_delta": current["headroom"] - baseline_values["headroom"],
        "c2_rate_delta": current["c2_rate"] - baseline_values["c2_rate"],
    }


def _baseline_c2_rate(baseline: Dict[str, Any]) -> float:
    """Read the canonical C2 field, retaining compatibility with EXP001."""
    if "c2_rate" in baseline:
        return float(baseline["c2_rate"])
    if "c2_wrong_high_utility_loss_rate" in baseline:
        return float(baseline["c2_wrong_high_utility_loss_rate"])
    raise KeyError(
        "Baseline must define c2_rate or the legacy "
        "c2_wrong_high_utility_loss_rate"
    )


def evaluate_val_experiment(
    *,
    feature_root: Path,
    stage_b_root: Path,
    dataset_root: Path,
    checkpoint: Path,
    baseline_path: Path,
    output_dir: Path,
    model_type: str,
    device_name: str,
    batch_size: int,
    experiment_id: str = "",
) -> Dict[str, Any]:
    """Evaluate one frozen checkpoint on Val and compare with the frozen baseline."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    baseline = _load(baseline_path)
    feature_summary = _load(feature_root / "stage_c_feature_summary.json")
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    device = torch.device(
        device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    dataset = EpisodeFeatureDataset(feature_root / "features" / "val.jsonl", **stats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episode_batch,
        num_workers=0,
    )
    model = build_utility_predictor(model_type).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    predictions = predict_dataset(
        model,
        loader,
        load_stage_b_lookup(stage_b_root / "utility_labels" / "val.jsonl"),
        device,
    )
    metrics = evaluate_predictions(predictions, _categories(feature_root))

    stage_a_summary = _load(dataset_root / "stage_a_summary.json")
    stage_a_rows = load_jsonl(stage_a_summary["episode_files"]["val"])
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl")
    feature_rows = load_jsonl(feature_summary["feature_files"]["val"])
    aligned = prepare_aligned_rows(
        stage_a_rows,
        stage_b_rows,
        feature_rows,
        predictions,
        expected_split="val",
    )
    analysis = analyze_rows(aligned, _categories(feature_root), split="val", model=model_type)
    analysis["evaluation_protocol"] = {
        "split": "val",
        "test_used": False,
        "stgcn_frozen": True,
        "source_stage_c_v0_accepted": True,
    }
    comparison = _comparison(baseline, metrics, analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "val_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    metrics_payload = {
        "protocol": "ACTIVEVIEW v11.5 Stage C-v1 Val-only evaluation",
        "experiment_id": experiment_id,
        "model_type": model_type,
        "split": "val",
        "test_used": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary_sha256": file_sha256(feature_root / "stage_c_feature_summary.json"),
        "metrics": metrics,
        "comparison_to_frozen_v0_val": comparison,
        "prediction_file": str(prediction_path.resolve()),
        "prediction_file_sha256": file_sha256(prediction_path),
    }
    analysis["comparison_to_frozen_v0_val"] = comparison
    analysis["artifact_provenance"] = {
        "baseline_sha256": file_sha256(baseline_path),
        "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary_sha256": file_sha256(feature_root / "stage_c_feature_summary.json"),
    }
    _write_json(output_dir / "val_analysis_full.json", analysis)
    # Keep the full machine-readable metrics under the runtime output root.
    # A caller may copy a compact summary into an experiment note if desired.
    _write_json(output_dir / "val_metrics_full.json", metrics_payload)
    return {
        "output_dir": str(output_dir.resolve()),
        "split": "val",
        "test_used": False,
        "episode_count": len(predictions),
        "metrics": metrics,
        "comparison_to_frozen_v0_val": comparison,
    }


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("set_ranker", "pairwise_mlp"), default="set_ranker")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = evaluate_val_experiment(
        feature_root=args.feature_root,
        stage_b_root=args.stage_b_root,
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        baseline_path=args.baseline,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        model_type=args.model_type,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    metrics = result["metrics"]
    comparison = result["comparison_to_frozen_v0_val"]
    experiment_metrics = comparison["experiment"]
    compact = {
        "accuracy": experiment_metrics["accuracy"],
        "macro_f1": experiment_metrics["macro_f1"],
        "mean_regret": experiment_metrics["mean_regret"],
        "median_regret": experiment_metrics["median_regret"],
        "p90_regret": experiment_metrics["p90_regret"],
        "headroom": experiment_metrics["headroom"],
        "c2_rate": experiment_metrics["c2_rate"],
        "test_used": False,
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
