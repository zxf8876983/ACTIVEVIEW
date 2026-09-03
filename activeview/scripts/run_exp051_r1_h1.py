#!/usr/bin/env python3
"""Reproduce EXP050 H1 with the newly frozen checkpoint (Val only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.core.paths import get_data_root
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache
from activeview.scripts.run_stage_d_exp049_051 import (
    M_VALUES,
    _JointRevision,
    _candidate_cache_rows,
    _decision_rows,
    _expanded_stage_b_rows,
    _joint_select,
    _legal_order,
    _load_pairwise_and_azimuths,
)


def run(data_root: Path, checkpoint: Path, device: torch.device) -> dict[str, object]:
    rows = _rows(data_root, "val")
    if len(rows) != 9742:
        raise RuntimeError(f"unexpected Val population: {len(rows)}")
    sources = _episode_sources(data_root, "val")
    cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz")
    v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    v0 = {str(row["episode_id"]): row for row in v0_rows}
    pairwise, azimuths = _load_pairwise_and_azimuths(data_root, rows, sources)
    orders = {str(row["episode_id"]): _legal_order(row, pairwise, azimuths, v0) for row in rows}
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = _JointRevision().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    results: dict[str, object] = {}
    for budget in M_VALUES:
        cache_rows = _candidate_cache_rows(rows, orders, budget, pairwise, v0)
        selected = _joint_select(model, cache, rows, orders, budget, device)
        decisions = _decision_rows(cache_rows, v0, {"selected": selected})
        trajectory_records = _expanded_stage_b_rows(stage_b, cache, cache_rows, orders)
        trajectory_rows = build_stage_d_trajectories(trajectory_records, v0_rows, cache_rows, decisions)
        # The evaluator consumes the canonical label mapping from Stage-B summary.
        summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
        mapping = json.loads(Path(summary["label_mapping"]).read_text())
        categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
        results[str(budget)] = summarize_trajectory_rows(trajectory_rows, categories)
    return {"experiment_id": "EXP050-R1", "status": "COMPLETED", "checkpoint": str(checkpoint.resolve()), "results": results, "test_used": False, "training_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    result = run(args.data_root.resolve(), args.checkpoint.resolve(), device)
    output = Path(__file__).resolve().parents[2] / "experiments/stage_d/EXP050_R1_joint_rollout_revision"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
