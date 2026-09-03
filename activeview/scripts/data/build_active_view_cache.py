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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.recognition.stgcn.model import STGCN
from activeview.data.preprocessing.cache import (
    build_cache_summary,
    build_second_step_rows,
    load_jsonl,
    load_pairwise_geodesic,
)
from activeview.core.paths import get_data_root
from activeview.perception.skeleton import get_skeleton_definition
from activeview.data.generation.utility_labels import file_sha256


def _load_frozen_stgcn(checkpoint: Path, device_name: str) -> tuple[STGCN, torch.device]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = STGCN(
        in_channels=3,
        num_classes=16,
        graph_strategy="spatial",
        edge_importance_weighting=True,
        skel_def=get_skeleton_definition(backend="h36m_17"),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device


def _pairwise_index(root: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[int, dict[int, float]]]:
    keys = {(str(row["scene_id"]), str(row["region"])) for row in rows}
    result: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    for scene_id, region in sorted(keys):
        path = root / scene_id / f"{region}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing navigation-only pairwise cache for {scene_id}/{region}: {path}"
            )
        result[(scene_id, region)] = load_pairwise_geodesic(path)
    return result


def build(
    *,
    dataset_root: Path,
    stage_b_root: Path,
    feature_root: Path,
    train_predictions: Path,
    val_predictions: Path,
    pairwise_root: Path,
    checkpoint: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    stage_a_summary = json.loads((dataset_root / "stage_a_summary.json").read_text(encoding="utf-8"))
    model, device = _load_frozen_stgcn(checkpoint, device_name)
    source_paths = {"train": train_predictions, "val": val_predictions}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    total_episode_counts: dict[str, int] = {}
    v0_move_counts: dict[str, int] = {}
    for split, prediction_path in source_paths.items():
        stage_a_rows = load_jsonl(Path(stage_a_summary["episode_files"][split]))
        stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / f"{split}.jsonl")
        feature_rows = load_jsonl(feature_root / "features" / f"{split}.jsonl")
        prediction_rows = load_jsonl(prediction_path)
        pairwise = _pairwise_index(pairwise_root, stage_a_rows)
        rows, move_count = build_second_step_rows(
            stage_a_rows=stage_a_rows,
            stage_b_rows=stage_b_rows,
            feature_rows=feature_rows,
            v0_prediction_rows=prediction_rows,
            pairwise_by_region=pairwise,
            stgcn_model=model,
            device=device,
        )
        split_rows[split] = rows
        total_episode_counts[split] = len(stage_a_rows)
        v0_move_counts[split] = move_count
        print(json.dumps({"split": split, "total_episodes": len(stage_a_rows), "v0_move_episodes": move_count, "second_step_rows": len(rows)}))
    summary = build_cache_summary(
        output_dir=output_dir,
        train_rows=split_rows["train"],
        val_rows=split_rows["val"],
        source_paths=source_paths,
        checkpoint=checkpoint,
        pairwise_root=pairwise_root,
    )
    summary["source_episode_counts"] = total_episode_counts
    summary["v0_move_eligible_episode_counts"] = v0_move_counts
    summary["eligible_record_counts"] = {
        split: len({str(row["record_id"]) for row in split_rows[split]})
        for split in ("train", "val")
    }
    summary["source_stage_a_summary"] = str((dataset_root / "stage_a_summary.json").resolve())
    summary["source_stage_a_summary_sha256"] = file_sha256(dataset_root / "stage_a_summary.json")
    summary["source_stage_b_root"] = str(stage_b_root.resolve())
    summary["source_stage_b_utility_sha256"] = {
        split: file_sha256(stage_b_root / "utility_labels" / f"{split}.jsonl")
        for split in ("train", "val")
    }
    summary["source_stage_c_v0_feature_root"] = str(feature_root.resolve())
    (output_dir / "stage_d_feature_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--pairwise-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    build(
        dataset_root=args.dataset_root,
        stage_b_root=args.stage_b_root,
        feature_root=args.feature_root,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        pairwise_root=args.pairwise_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
