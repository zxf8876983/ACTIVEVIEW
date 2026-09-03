#!/usr/bin/env python3
"""文件用途：
    执行离线数据生成、划分或缓存构建入口。

主要输入：
    - 命令行参数与已有运行时数据。
主要输出：
    - 数据集、缓存或清单文件。
项目角色：
    - 属于 data 脚本入口，仅调用正式数据模块。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.data.preprocessing.policy_data import EpisodeFeatureDataset, collate_episode_batch, load_feature_statistics
from activeview.methods.active_view.policy import load_stage_b_lookup, predict_dataset
from activeview.data.generation.utility_labels import file_sha256
from activeview.methods.active_view.utility_predictor import build_utility_predictor
from activeview.core.paths import get_data_root


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def generate(
    *, feature_root: Path, stage_b_root: Path, checkpoint: Path,
    output_dir: Path, device_name: str, batch_size: int,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    summary_path = feature_root / "stage_c_feature_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    geometry_dim = int(summary["candidate_geometry_dim"])
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_utility_predictor("set_ranker", geometry_dim=geometry_dim).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    if not splits or any(split not in {"train", "val", "test"} for split in splits):
        raise ValueError("splits must be drawn from train, val, test")
    for split in splits:
        dataset = EpisodeFeatureDataset(feature_root / "features" / f"{split}.jsonl", **stats)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_episode_batch, num_workers=0)
        rows = predict_dataset(model, loader, load_stage_b_lookup(stage_b_root / "utility_labels" / f"{split}.jsonl"), device, model_type="set_ranker")
        path = output_dir / f"{split}_predictions.jsonl"
        _write_jsonl(path, rows)
        counts[split] = len(rows)
        hashes[split] = file_sha256(path)
    result = {
        "protocol": "ACTIVEVIEW frozen Stage C-v0 Train/Val proposal inference",
        "model": "frozen Stage C-v0 Set Ranker", "test_used": False, "test_generated": False,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "feature_summary": str(summary_path.resolve()), "feature_summary_sha256": file_sha256(summary_path),
        "prediction_files": {split: str((output_dir / f"{split}_predictions.jsonl").resolve()) for split in splits},
        "prediction_file_sha256": hashes, "prediction_counts": counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stage_c/set_ranker_best.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    print(json.dumps(generate(feature_root=args.feature_root, stage_b_root=args.stage_b_root, checkpoint=args.checkpoint, output_dir=args.output_dir, device_name=args.device, batch_size=args.batch_size, splits=tuple(args.splits)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
