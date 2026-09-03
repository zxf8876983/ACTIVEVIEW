#!/usr/bin/env python3
"""EXP056: paired seed stability for the original and multi-positive JR losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.core.paths import get_data_root
from activeview.scripts.run_exp051_r1_closed_loop import _candidate_order, run as run_closed_loop
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache
from activeview.scripts.run_stage_d_exp049_051 import _JointRevision, _load_pairwise_and_azimuths
from activeview.scripts.train_exp055_multi_positive import _Dataset, _audit, _examples, _full_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments/stage_d/EXP056_objective_stability"
SEEDS = (42, 43, 44)
N_CLASSES = 16
EPOCHS = 20


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _orders(data_root: Path, split: str, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str]) -> dict[str, list[int]]:
    v0_path = data_root / f"experiments/stage_d/EXP014_two_step_sequential/v0_predictions/{split}_predictions.jsonl"
    v0 = {str(row["episode_id"]): row for row in load_jsonl(v0_path)}
    pairwise, azimuths = _load_pairwise_and_azimuths(data_root, rows, sources)
    output: dict[str, list[int]] = {}
    for row in rows:
        episode = str(row["episode_id"])
        if bool(v0[episode]["predicted_stays"]):
            output[episode] = []
            continue
        key = (str(row["scene_id"]), str(row["region"]))
        output[episode] = _candidate_order(row, int(row["s1_viewpoint_id"]), {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])}, pairwise[key], azimuths[key])
    return output


def _fit_variant(arrays: tuple[np.ndarray, ...], stats: dict[str, Any], objective: str, seed: int, device: torch.device, checkpoint: Path) -> dict[str, Any]:
    _seed(seed)
    loader = DataLoader(_Dataset(arrays), batch_size=512, shuffle=True)
    model = _JointRevision().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            fallback = batch["fallback"].long().to(device)
            positive = batch["positive"].float().to(device)
            labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask)
            valid_scores = scores.masked_fill(~mask, -1e9)
            fallback_loss = nn.functional.cross_entropy(valid_scores, fallback, reduction="none")
            if objective == "MULTI_POSITIVE_JR":
                positive_mask = positive.bool() & mask
                all_lse = torch.logsumexp(valid_scores, dim=1)
                positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1)
                main = torch.where(positive_mask.any(dim=1), all_lse - positive_lse, fallback_loss)
            else:
                main = fallback_loss
            bce_all = nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none")
            bce = (bce_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            posterior_target = labels.unsqueeze(1).expand(-1, posterior.size(1))
            posterior_all = nn.functional.cross_entropy(posterior.reshape(-1, N_CLASSES), posterior_target.reshape(-1), reduction="none").reshape_as(scores)
            posterior_loss = (posterior_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (main + 0.25 * bce + 0.05 * posterior_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f"EXP056 {objective} seed={seed} epoch {epoch}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_state_dict": model.state_dict(), "seed": seed, "objective": objective, "epochs": EPOCHS}, checkpoint)
    return {**stats, "objective": objective, "seed": seed, "final_loss": history[-1], "loss_history": history, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint)}


def _evaluate(data_root: Path, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    value = run_closed_loop(data_root, checkpoint, device)
    val_rows = _rows(data_root, "val")
    v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    stage_b_rows = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    full = _full_metrics(value["h2_terminal_predictions_moving"], value["moving_episode_ids"], v0_rows, stage_b_rows)
    if len(value["h2_terminal_predictions_moving"]) != len(val_rows):
        raise RuntimeError("EXP056 terminal prediction count mismatch")
    return {"moving": value["h2"]["terminal"], "full": full, "raw": value}


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def run(data_root: Path, device: torch.device) -> dict[str, Any]:
    train_rows = _rows(data_root, "train")
    val_rows = _rows(data_root, "val")
    if (len(train_rows), len(val_rows)) != (29133, 9742):
        raise RuntimeError("canonical Train/Val population mismatch")
    train_sources = _episode_sources(data_root, "train")
    train_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz")
    train_orders = _orders(data_root, "train", train_rows, train_sources)
    arrays, train_stats = _examples(train_rows, train_cache, train_orders)
    results: dict[str, dict[str, dict[str, Any]]] = {"ORIGINAL_JR": {}, "MULTI_POSITIVE_JR": {}}
    checkpoint_entries: list[dict[str, Any]] = []
    runtime = data_root / "experiments/stage_d/EXP056_objective_stability/runtime"
    for seed in SEEDS:
        for objective in ("ORIGINAL_JR", "MULTI_POSITIVE_JR"):
            checkpoint = runtime / f"{objective.lower()}_seed{seed}.pth"
            stats = _fit_variant(arrays, train_stats, objective, seed, device, checkpoint)
            evaluated = _evaluate(data_root, checkpoint, device)
            results[objective][str(seed)] = {"train": stats, "moving": evaluated["moving"], "full": evaluated["full"], "raw": evaluated["raw"]}
            checkpoint_entries.append({"objective": objective, "seed": seed, "path": str(checkpoint.resolve()), "sha256": stats["checkpoint_sha256"]})
    per_seed: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        original = results["ORIGINAL_JR"][str(seed)]
        multi = results["MULTI_POSITIVE_JR"][str(seed)]
        per_seed.append({"seed": seed, "original_moving": original["moving"], "multi_moving": multi["moving"], "original_full": original["full"], "multi_full": multi["full"]})
        paired.append({"seed": seed, **_audit(original["raw"], multi["raw"])})

    def metric_summary(population: str, metric: str, objective: str) -> dict[str, float]:
        return _summary([float(results[objective][str(seed)][population][metric]) for seed in SEEDS])

    deltas = {"moving_accuracy": [per_seed[i]["multi_moving"]["accuracy"] - per_seed[i]["original_moving"]["accuracy"] for i in range(3)], "moving_macro_f1": [per_seed[i]["multi_moving"]["macro_f1"] - per_seed[i]["original_moving"]["macro_f1"] for i in range(3)], "full_accuracy": [per_seed[i]["multi_full"]["accuracy"] - per_seed[i]["original_full"]["accuracy"] for i in range(3)], "full_macro_f1": [per_seed[i]["multi_full"]["macro_f1"] - per_seed[i]["original_full"]["macro_f1"] for i in range(3)]}
    summary = {"moving": {"original_accuracy": metric_summary("moving", "accuracy", "ORIGINAL_JR"), "multi_accuracy": metric_summary("moving", "accuracy", "MULTI_POSITIVE_JR"), "original_macro_f1": metric_summary("moving", "macro_f1", "ORIGINAL_JR"), "multi_macro_f1": metric_summary("moving", "macro_f1", "MULTI_POSITIVE_JR"), "delta_accuracy": _summary(deltas["moving_accuracy"]), "delta_macro_f1": _summary(deltas["moving_macro_f1"])}, "full": {"original_accuracy": metric_summary("full", "accuracy", "ORIGINAL_JR"), "multi_accuracy": metric_summary("full", "accuracy", "MULTI_POSITIVE_JR"), "original_macro_f1": metric_summary("full", "macro_f1", "ORIGINAL_JR"), "multi_macro_f1": metric_summary("full", "macro_f1", "MULTI_POSITIVE_JR"), "delta_accuracy": _summary(deltas["full_accuracy"]), "delta_macro_f1": _summary(deltas["full_macro_f1"])}}
    wins = {"accuracy": int(sum(value > 0 for value in deltas["moving_accuracy"])), "macro_f1": int(sum(value > 0 for value in deltas["moving_macro_f1"]))}
    case = "CASE_A" if wins["accuracy"] >= 2 and summary["moving"]["delta_accuracy"]["mean"] > 0 and summary["moving"]["delta_macro_f1"]["mean"] >= 0 else "CASE_C"
    recommended = next((item["path"] for item in checkpoint_entries if item["objective"] == "MULTI_POSITIVE_JR" and item["seed"] == 42), None)
    output = {"experiment_id": "EXP056", "status": "COMPLETED", "split": "val", "seeds": list(SEEDS), "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows), "val_full_population": len(load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl"))}, "train_objective_stats": train_stats, "per_seed": per_seed, "summary": summary, "paired_deltas": deltas, "wins": wins, "paired_audit": paired, "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "checkpoint_manifest": checkpoint_entries, "case": case, "recommended_canonical": "MULTI_POSITIVE" if case == "CASE_A" else None, "recommended_fixed_seed": 42 if case == "CASE_A" else None, "recommended_checkpoint": recommended if case == "CASE_A" else None}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "result.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (EXP_DIR / "per_seed_metrics.json").write_text(json.dumps(per_seed, indent=2) + "\n", encoding="utf-8")
    (EXP_DIR / "paired_audit.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    (EXP_DIR / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_entries, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP056 objective stability on Train/Val")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EXP056 requires CUDA")
    print(json.dumps(run(args.data_root.resolve(), device), indent=2))


if __name__ == "__main__":
    main()
