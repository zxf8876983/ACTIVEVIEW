#!/usr/bin/env python3
"""Re-train the frozen EXP050 Joint Revision once and persist its weights.

This script intentionally mirrors the original EXP050 training path and does
not evaluate Test or alter any experiment hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from activeview.core.paths import get_data_root
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.scripts.run_stage_d_exp046_048 import _load_cache
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp049_051 import (
    _fit_joint,
    _legal_order,
    _load_pairwise_and_azimuths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    train_rows = _rows(data_root, "train")
    if len(train_rows) != 29133:
        raise RuntimeError(f"unexpected Train population: {len(train_rows)}")
    sources = _episode_sources(data_root, "train")
    cache_path = data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"
    train_cache = _load_cache(cache_path)
    pairwise, azimuths = _load_pairwise_and_azimuths(data_root, train_rows, sources)
    v0_rows = train_rows
    v0_path = data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl"
    v0 = {str(row["episode_id"]): row for row in load_jsonl(v0_path)}
    orders = {str(row["episode_id"]): _legal_order(row, pairwise, azimuths, v0) for row in v0_rows}
    model, training = _fit_joint(train_rows, train_cache, orders, device)
    output_dir = data_root / "experiments/stage_d/EXP050_joint_rollout_revision"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "joint_revision_final.pth"
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "model_state_dict": model.state_dict(),
        "seed": 42,
        "epochs": 20,
        "batch_size": 512,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "candidate_budgets": [2, 4, 8, 16, "ALL_LEGAL"],
        "architecture": "JointRevision",
        "source_commit": source_commit,
        "training": training,
    }
    torch.save(payload, checkpoint)
    manifest = {
        "experiment_id": "EXP050-R1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_commit": source_commit,
        "training": training,
        "test_used": False,
        "training_performed": True,
    }
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
