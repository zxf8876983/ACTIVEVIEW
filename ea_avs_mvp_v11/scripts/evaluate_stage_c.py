#!/usr/bin/env python3
"""Evaluate a trained Stage C predictor with frozen Stage B diagnostics."""

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

from ea_avs_mvp_v11.active_view.stage_c_dataset import EpisodeFeatureDataset, collate_episode_batch, load_feature_statistics
from ea_avs_mvp_v11.active_view.stage_c_evaluation import evaluate_predictions, load_stage_b_lookup, predict_dataset
from ea_avs_mvp_v11.active_view.stage_c_features import schema_metadata
from ea_avs_mvp_v11.active_view.utility_label_builder import file_sha256
from ea_avs_mvp_v11.active_view.utility_predictor import build_utility_predictor
from ea_avs_mvp_v11.core.paths import get_data_root


def evaluate_model(*, feature_root: Path, stage_b_root: Path, checkpoint: Path, output_dir: Path, model_type: str, device_name: str, batch_size: int) -> Dict[str, Any]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    feature_summary_path = feature_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    mapping = json.loads((feature_root.parent.parent / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed" / "label_mapping.json").read_text(encoding="utf-8"))
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    model = build_utility_predictor(model_type).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    predictions_dir = output_dir / "predictions"; predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {}
    for split in ("val", "test"):
        dataset = EpisodeFeatureDataset(feature_root / "features" / f"{split}.jsonl", **stats)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_episode_batch, num_workers=0)
        rows = predict_dataset(model, loader, load_stage_b_lookup(stage_b_root / "utility_labels" / f"{split}.jsonl"), device)
        metrics[split] = evaluate_predictions(rows, categories)
        with (predictions_dir / f"{model_type}_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    summary = {
        "protocol": "ACTIVEVIEW v11.5 Stage C offline evaluation", "stage": "C", "status": "evaluated",
        "model_type": model_type, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary": str(feature_summary_path.resolve()), "feature_summary_sha256": file_sha256(feature_summary_path),
        "source_stage_a_summary_sha256": feature_summary["source_stage_a_summary_sha256"],
        "source_stage_a_episode_sha256": feature_summary["source_stage_a_episode_sha256"],
        "source_stage_b_summary_sha256": feature_summary["source_stage_b_summary_sha256"],
        "source_stage_b_utility_sha256": feature_summary["source_stage_b_utility_sha256"],
        "stgcn_checkpoint_sha256": feature_summary["stgcn_checkpoint_sha256"],
        "label_mapping": str((feature_root.parent.parent / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed" / "label_mapping.json").resolve()),
        "label_mapping_sha256": feature_summary["label_mapping_sha256"],
        "canonical_split_counts": feature_summary["canonical_split_counts"],
        "feature_file_counts": feature_summary["feature_file_counts"],
        "feature_schema": schema_metadata(), "body_yaw_used": False, "movement_cost_penalty_used": False,
        "future_candidate_perception_used_as_input": False, "stgcn_frozen": True, "categories": categories,
        "evaluation_only_fields": ["label_id", "selected_true_utility", "selected_predicted_label_id", "selected_entropy", "candidate_oracle_predicted_label_id", "safe_oracle_predicted_label_id"],
        "feature_root": str(feature_root.resolve()), "metrics": metrics,
        "prediction_files": {split: str((predictions_dir / f"{model_type}_{split}.jsonl").resolve()) for split in ("val", "test")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{model_type}_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/evaluations")
    parser.add_argument("--model-type", choices=("pairwise_mlp", "set_ranker"), required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(evaluate_model(feature_root=args.feature_root, stage_b_root=args.stage_b_root, checkpoint=args.checkpoint, output_dir=args.output_dir, model_type=args.model_type, device_name=args.device, batch_size=args.batch_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
