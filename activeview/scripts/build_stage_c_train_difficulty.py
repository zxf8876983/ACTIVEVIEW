#!/usr/bin/env python3
"""Build Train-only record difficulty from the frozen Stage C-v0 predictor."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_dataset import EpisodeFeatureDataset, collate_episode_batch, load_feature_statistics
from activeview.active_view.stage_c_evaluation import load_stage_b_lookup, predict_dataset
from activeview.active_view.utility_label_builder import file_sha256
from activeview.active_view.utility_predictor import build_utility_predictor
from activeview.core.paths import get_data_root


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def build_train_difficulty(
    *,
    feature_root: Path,
    stage_b_root: Path,
    checkpoint: Path,
    difficulty_output: Path,
    prediction_output: Path,
    hard_fraction: float = 0.20,
    device_name: str = "cuda:0",
    batch_size: int = 256,
) -> Dict[str, Any]:
    if not 0.0 < hard_fraction < 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    dataset = EpisodeFeatureDataset(feature_root / "features/train.jsonl", **stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_episode_batch, num_workers=0)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_utility_predictor("set_ranker").to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    predictions = predict_dataset(
        model,
        loader,
        load_stage_b_lookup(stage_b_root / "utility_labels/train.jsonl"),
        device,
    )
    _write_jsonl(prediction_output, predictions)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["record_id"])].append(row)
    records = []
    for record_id, rows in grouped.items():
        regrets = [float(row["regret"]) for row in rows]
        records.append({
            "record_id": record_id,
            "label_ids": sorted({int(row["label_id"]) for row in rows}),
            "episode_count": len(rows),
            "mean_safe_oracle_regret": sum(regrets) / len(regrets),
        })
    records.sort(key=lambda row: (-float(row["mean_safe_oracle_regret"]), str(row["record_id"])))
    hard_count = max(1, int(math.ceil(len(records) * hard_fraction)))
    hard_records = records[:hard_count]
    normal_records = records[hard_count:]
    difficulty = {
        "protocol": "ACTIVEVIEW v11.5 EXP002 Train-only difficulty",
        "source_split": "train",
        "source_model": "frozen Stage C-v0 SetUtilityRanker",
        "metric": "mean_safe_oracle_regret",
        "hard_fraction": hard_fraction,
        "record_count": len(records),
        "hard_record_count": len(hard_records),
        "normal_record_count": len(normal_records),
        "difficulty_threshold": float(hard_records[-1]["mean_safe_oracle_regret"]),
        "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary_sha256": file_sha256(feature_root / "stage_c_feature_summary.json"),
        "prediction_output": "ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v1/EXP002_hard_record_sampling/reference/train_predictions.jsonl",
        "hard_records": hard_records,
        "normal_records": normal_records,
        "records": records,
    }
    _write_json(difficulty_output, difficulty)
    return difficulty


def main() -> None:
    data_root = get_data_root()
    exp_root = REPO_ROOT / "experiments/stage_c_v1/EXP002_hard_record_sampling"
    runtime_root = data_root / "experiments/stage_c_v1/EXP002_hard_record_sampling/reference"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stage_c/set_ranker_best.pth")
    parser.add_argument("--difficulty-output", type=Path, default=exp_root / "train_record_difficulty.json")
    parser.add_argument("--prediction-output", type=Path, default=runtime_root / "train_predictions.jsonl")
    parser.add_argument("--hard-fraction", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = build_train_difficulty(
        feature_root=args.feature_root,
        stage_b_root=args.stage_b_root,
        checkpoint=args.checkpoint,
        difficulty_output=args.difficulty_output,
        prediction_output=args.prediction_output,
        hard_fraction=args.hard_fraction,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: result[key] for key in ("record_count", "hard_record_count", "normal_record_count", "difficulty_threshold", "source_split")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
