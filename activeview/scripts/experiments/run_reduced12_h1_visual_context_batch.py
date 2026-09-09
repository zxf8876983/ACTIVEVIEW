#!/usr/bin/env python3
"""Train/Val-only reduced12 H1 visual-context representation batch.

The experiment reuses the existing Stay-aware GTMargin listwise objective and
adds only visited s0 DINO representations.  It deliberately reads the
existing DINO cache; it never renders or extracts new perception data and it
does not read policy Test artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.policy_data import RecordBalancedSampler, load_feature_statistics
from activeview.methods.active_view.utility_predictor import (
    CandidateGeometryEncoder,
    CurrentContextEncoder,
    SetUtilityRanker,
    count_parameters,
)
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import (
    _action_targets,
    _classification,
    _listwise_loss,
    evaluate_selector,
)

SEED = 42
NUM_CLASSES = 12
BASE_CURRENT_DIM = 271
GEOMETRY_DIM = 11
DINO_DIM = 768
SPATIAL_DIM = 64
EPOCHS = 20
BATCH_SIZE = 256
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_visual_context_batch"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_dino_store(data_root: Path) -> tuple[dict[tuple[str, str, str, int], int], np.ndarray, dict[str, Any]]:
    cache_dir = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history"
    embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.ndim != 3 or embeddings.shape[1:] != (16, DINO_DIM):
        raise ValueError(f"unexpected DINO shape: {embeddings.shape}")
    lookup: dict[tuple[str, str, str, int], int] = {}
    with (cache_dir / "manifest.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
            if key in lookup:
                raise ValueError(f"duplicate DINO key: {key}")
            lookup[key] = index
    if len(lookup) != len(embeddings):
        raise ValueError("DINO manifest and embedding count differ")
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    if bool(summary.get("future_candidate_rgb_used", True)):
        raise ValueError("DINO cache is marked as future-candidate RGB")
    return lookup, embeddings, summary


def _subset_cache(cache: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    index = np.asarray(indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        result[key] = value[index] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] >= int(index.max(initial=-1)) + 1 else value
    return result


def _covered_rows(
    rows: Sequence[Mapping[str, Any]],
    allowed_episode_ids: set[str],
    dino_lookup: Mapping[tuple[str, str, str, int], int],
) -> tuple[list[dict[str, Any]], list[int]]:
    selected: list[dict[str, Any]] = []
    indices: list[int] = []
    for index, row in enumerate(rows):
        episode_id = str(row["episode_id"])
        if episode_id not in allowed_episode_ids:
            continue
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        if key not in dino_lookup:
            raise ValueError(f"DINO cache missing a Stage-D covered row: {key}")
        selected.append(dict(row))
        indices.append(index)
    if not selected:
        raise RuntimeError("no DINO-covered rows selected")
    return selected, indices


class VisualDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        cache: Mapping[str, np.ndarray],
        stats: Mapping[str, np.ndarray],
        dino_lookup: Mapping[tuple[str, str, str, int], int],
        dino_embeddings: np.ndarray,
        branch: str,
        visual_mean: np.ndarray | None = None,
        visual_std: np.ndarray | None = None,
    ) -> None:
        self.rows = list(rows)
        self.cache = cache
        self.current_mean = stats["current_mean"]
        self.current_std = stats["current_std"]
        self.geometry_mean = stats["geometry_mean"]
        self.geometry_std = stats["geometry_std"]
        self.dino_lookup = dino_lookup
        self.dino_embeddings = dino_embeddings
        self.branch = branch
        self.visual_mean = visual_mean
        self.visual_std = visual_std
        self.max_candidates = int(cache["candidate_logp"].shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        current = (np.asarray(row["current_feature"], dtype=np.float32) - self.current_mean) / self.current_std
        geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - self.geometry_mean) / self.geometry_std
        count = len(row["candidate_viewpoint_ids"])
        padded_geometry = np.zeros((self.max_candidates, GEOMETRY_DIM), dtype=np.float32)
        padded_geometry[:count] = geometry
        item: dict[str, Any] = {
            "current_feature": torch.from_numpy(current),
            "candidate_geometry": torch.from_numpy(padded_geometry),
            "candidate_mask": torch.from_numpy(np.asarray(self.cache["candidate_mask"][index], dtype=bool)),
            "current_logp": torch.from_numpy(np.asarray(self.cache["current_logp"][index], dtype=np.float32)),
            "true_logp": torch.from_numpy(np.asarray(self.cache["candidate_logp"][index], dtype=np.float32)),
            "label_id": int(row["label_id"]),
            "episode_id": str(row["episode_id"]),
        }
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        tokens = np.asarray(self.dino_embeddings[self.dino_lookup[key]], dtype=np.float32)
        if self.branch in ("dino_mean", "dino_meanmax"):
            visual = tokens.mean(axis=0) if self.branch == "dino_mean" else np.concatenate([tokens.mean(axis=0), tokens.max(axis=0)])
            if self.visual_mean is not None and self.visual_std is not None:
                visual = (visual - self.visual_mean) / self.visual_std
            item["visual"] = torch.from_numpy(visual.astype(np.float32))
        else:
            item["dino_tokens"] = torch.from_numpy(tokens)
        return item


def _collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_keys = tuple(key for key in items[0] if isinstance(items[0][key], torch.Tensor))
    result: dict[str, Any] = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    result["label_id"] = torch.tensor([int(item["label_id"]) for item in items], dtype=torch.long)
    result["episode_id"] = [str(item["episode_id"]) for item in items]
    return result


class ExpandedVisualRanker(nn.Module):
    def __init__(self, visual_dim: int) -> None:
        super().__init__()
        self.ranker = SetUtilityRanker(BASE_CURRENT_DIM + visual_dim, GEOMETRY_DIM)
        self.stay_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([batch["current_feature"], batch["visual"]], dim=1)
        candidate = self.ranker(current, batch["candidate_geometry"], batch["candidate_mask"])
        stay = self.stay_head(self.ranker.current_encoder(current)).squeeze(-1)
        return stay, candidate


class SpatialVisualRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(DINO_DIM, SPATIAL_DIM), nn.GELU())
        self.ranker = SetUtilityRanker(BASE_CURRENT_DIM + SPATIAL_DIM, GEOMETRY_DIM)
        self.stay_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.projector(batch["dino_tokens"]).mean(dim=1)
        current = torch.cat([batch["current_feature"], visual], dim=1)
        candidate = self.ranker(current, batch["candidate_geometry"], batch["candidate_mask"])
        stay = self.stay_head(self.ranker.current_encoder(current)).squeeze(-1)
        return stay, candidate


class CandidateConditionedSpatialRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(DINO_DIM, SPATIAL_DIM), nn.GELU())
        self.current_encoder = CurrentContextEncoder(BASE_CURRENT_DIM + SPATIAL_DIM)
        self.geometry_encoder = CandidateGeometryEncoder(GEOMETRY_DIM)
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + SPATIAL_DIM, 128), nn.GELU(), nn.Linear(128, 1))
        self.stay_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.projector(batch["dino_tokens"]).mean(dim=1)
        context = self.current_encoder(torch.cat([batch["current_feature"], visual], dim=1))
        geometry = self.geometry_encoder(batch["candidate_geometry"])
        visual_tokens = visual.unsqueeze(1).expand(-1, geometry.size(1), -1)
        context_tokens = context.unsqueeze(1).expand(-1, geometry.size(1), -1)
        candidate = self.fusion(torch.cat([context_tokens, geometry, visual_tokens], dim=-1)).squeeze(-1)
        stay = self.stay_head(context).squeeze(-1)
        return stay, candidate


def _model(branch: str) -> nn.Module:
    if branch == "dino_mean":
        return ExpandedVisualRanker(DINO_DIM)
    if branch == "dino_meanmax":
        return ExpandedVisualRanker(2 * DINO_DIM)
    if branch == "dino_spatial":
        return SpatialVisualRanker()
    if branch == "candidate_conditioned_spatial":
        return CandidateConditionedSpatialRanker()
    raise ValueError(f"unknown visual branch: {branch}")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _loss(stay: torch.Tensor, candidates: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    scores = torch.cat([stay[:, None], candidates], dim=1)
    valid = torch.cat([torch.ones((batch["candidate_mask"].size(0), 1), dtype=torch.bool, device=scores.device), batch["candidate_mask"]], dim=1)
    targets = _action_targets(batch, "margin_listwise")
    return _listwise_loss(scores, targets, valid)


def _score(model: nn.Module, dataset: VisualDataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    stays: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            stay, candidate = model(_move_batch(batch, device))
            stays.append(stay.cpu().numpy())
            candidates.append(candidate.cpu().numpy())
    return np.concatenate(stays), np.concatenate(candidates)


def _train_branch(
    branch: str,
    train_set: VisualDataset,
    val_set: VisualDataset,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    _seed()
    model = _model(branch).to(device)
    sampler = RecordBalancedSampler(train_set.rows, episodes_per_record=16, seed=SEED)
    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=_collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            stay, candidate = model(batch)
            loss = _loss(stay, candidate, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for raw_batch in DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0):
                batch = _move_batch(raw_batch, device)
                val_losses.append(float(_loss(*model(batch), batch).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        print(f"[visual:{branch}] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "branch": branch, "epoch": epoch, "seed": SEED}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "branch": branch, "seed": SEED, "epochs": EPOCHS, "batch_size": batch_size,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "selected_epoch": best_epoch, "best_val_loss": best_loss,
        "parameter_count": count_parameters(model), "checkpoint": str(checkpoint.resolve()),
        "elapsed_seconds": time.time() - started, "history": history, "test_used": False,
        "objective": "Stay-aware GTMargin listwise (unchanged)",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _visual_vectors(rows: Sequence[Mapping[str, Any]], lookup: Mapping[tuple[str, str, str, int], int], embeddings: np.ndarray, kind: str) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        tokens = np.asarray(embeddings[lookup[key]], dtype=np.float32)
        values.append(tokens.mean(axis=0) if kind == "dino_mean" else np.concatenate([tokens.mean(axis=0), tokens.max(axis=0)]))
    return np.asarray(values, dtype=np.float32)


def _normalise(train: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((train - mean) / std).astype(np.float32), ((value - mean) / std).astype(np.float32), (mean, std)


def _make_cache(cache: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[np.asarray(indices)] for key, value in cache.items()}


def _metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metric[key] for key in ("n", "accuracy", "macro_f1", "move_rate", "stay_rate", "s0_error_correction_rate", "s0_correct_harm_rate", "mean_selected_gt_true_logp", "mean_selected_gt_margin", "mean_regret_to_gt_logp_oracle", "p90_regret_to_gt_logp_oracle") if key in metric}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    feature_root = policy_root / "stage_c"
    train_rows_all = load_jsonl(feature_root / "features/train.jsonl")
    val_rows_all = load_jsonl(feature_root / "features/val.jsonl")
    stage_d_train = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/train.jsonl")}
    stage_d_val = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    train_cache_all = _read_npz(data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/train_candidate_true_logp.npz")
    val_cache_all = _read_npz(data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz")
    if len(train_rows_all) != len(train_cache_all["labels"]) or len(val_rows_all) != len(val_cache_all["labels"]):
        raise ValueError("Stage-C rows and true-logp cache are not aligned")
    dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root)
    train_rows, train_indices = _covered_rows(train_rows_all, stage_d_train, dino_lookup)
    val_rows, val_indices = _covered_rows(val_rows_all, stage_d_val, dino_lookup)
    train_cache = _make_cache(train_cache_all, train_indices)
    val_cache = _make_cache(val_cache_all, val_indices)
    if not np.array_equal(train_cache["labels"], np.asarray([int(row["label_id"]) for row in train_rows])):
        raise ValueError("Train labels do not align with true-logp cache")
    if not np.array_equal(val_cache["labels"], np.asarray([int(row["label_id"]) for row in val_rows])):
        raise ValueError("Val labels do not align with true-logp cache")
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    output = EXPERIMENT
    checkpoint_root = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_visual_context_batch"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Train covered={len(train_rows)} Val moving covered={len(val_rows)} device={device}", flush=True)
    branch_names = ("dino_mean", "dino_meanmax", "dino_spatial", "candidate_conditioned_spatial")
    visual_norms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    vector_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("dino_mean", "dino_meanmax"):
        train_vec = _visual_vectors(train_rows, dino_lookup, dino_embeddings, kind)
        val_vec = _visual_vectors(val_rows, dino_lookup, dino_embeddings, kind)
        train_norm, val_norm, (mean, std) = _normalise(train_vec, val_vec)
        vector_cache[kind] = (train_norm, val_norm)
        visual_norms[kind] = (mean, std)
    branch_metrics: dict[str, Any] = {}
    training: dict[str, Any] = {}
    for branch in branch_names:
        if branch in ("dino_mean", "dino_meanmax"):
            train_set = VisualDataset(train_rows, train_cache, stats, dino_lookup, dino_embeddings, branch, *visual_norms[branch])
            val_set = VisualDataset(val_rows, val_cache, stats, dino_lookup, dino_embeddings, branch, *visual_norms[branch])
        else:
            train_set = VisualDataset(train_rows, train_cache, stats, dino_lookup, dino_embeddings, branch)
            val_set = VisualDataset(val_rows, val_cache, stats, dino_lookup, dino_embeddings, branch)
        ckpt = checkpoint_root / f"{branch}_best.pth"
        summary_path = output / f"{branch}_training.json"
        if ckpt.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[visual:{branch}] skip completed checkpoint", flush=True)
        else:
            summary = _train_branch(branch, train_set, val_set, device, ckpt, summary_path, args.batch_size)
        training[branch] = summary
        model = _model(branch).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        stays, candidates = _score(model, val_set, device, args.batch_size)
        branch_metrics[branch] = evaluate_selector(val_rows, val_cache, stays, candidates, branch)
        partial = {"completed_branch": branch, "metrics_moving": {k: _metric_summary(v) for k, v in branch_metrics.items()}, "test_used": False}
        (output / "branch_results.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    previous = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))
    references_moving = {
        "FrozenStageCv0": previous["metrics_moving"]["FrozenStageCv0"],
        "GTMargin Listwise": previous["metrics_moving"]["margin_listwise"],
        "AnyCorrect Oracle": previous["metrics_moving"]["AnyCorrect Oracle"],
    }
    references_full = {
        "FrozenStageCv0": previous["metrics_full"]["FrozenStageCv0"],
        "GTMargin Listwise": previous["metrics_full"]["margin_listwise"],
        "AnyCorrect Oracle": previous["metrics_full"]["AnyCorrect Oracle"],
    }
    moving = {**references_moving, **branch_metrics}
    result = {
        "experiment_id": "REDUCED12_H1_VISUAL_CONTEXT_BATCH",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "metrics_moving": moving,
        "metrics_full_reference": references_full,
        "metrics_full_visual": {"status": "NOT_AVAILABLE", "reason": "Existing visited s0 DINO cache covers moving Val only; no DINO fallback or regeneration was used.", "covered_contexts": len(val_rows), "full_val_contexts": len(val_rows_all)},
        "population": {"stage_c_train_rows": len(train_rows_all), "stage_c_val_rows": len(val_rows_all), "visual_train_rows": len(train_rows), "visual_val_moving_rows": len(val_rows), "stage_d_train_rows": len(stage_d_train), "stage_d_val_rows": len(stage_d_val)},
        "training": training,
        "dino_cache": {**dino_summary, "manifest_entries": len(dino_lookup), "future_candidate_dino_used": False, "regenerated": False},
        "test_used": False,
        "test_read": False,
        "skeleton_regenerated": False,
        "rgb_regenerated": False,
        "dino_regenerated": False,
        "protocol": {"objective": "Stay-aware GTMargin listwise", "candidate_budget": "ALL_LEGAL", "terminal_observation": "real archived skeleton through frozen ST-GCN", "visual_input": "visited s0 DINO only"},
    }
    result["per_class_metrics"] = {label: {name: metric["per_class"][label] for name, metric in moving.items() if isinstance(metric, Mapping) and "per_class" in metric} for label in LABELS}
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "per_class_metrics.json").write_text(json.dumps(result["per_class_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    frozen = float(moving["FrozenStageCv0"]["accuracy"])
    visual_best = max(branch_names, key=lambda name: float(moving[name]["accuracy"]))
    visual_gain = 100.0 * (float(moving[visual_best]["accuracy"]) - frozen)
    lines = [
        "# Reduced12 H1 visual-context representation batch",
        "",
        "Train/Val only. No policy Test was read and no skeleton, RGB or DINO data was regenerated.",
        "",
        f"Visual DINO coverage: Train {len(train_rows)}/{len(train_rows_all)} Stage-C rows and moving Val {len(val_rows)}/{len(val_rows_all)} rows. The existing cache does not cover non-moving Full-Val contexts; Full-Val visual metrics are therefore reported as NOT_AVAILABLE rather than using a silent fallback.",
        "",
        "## Moving Val",
        "",
        "| H1 | Accuracy | Macro-F1 | Move rate | Rescue | Harm | GTMargin regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in moving.items():
        if "accuracy" not in metric:
            continue
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} | {metric.get('s0_error_correction_rate', 0.0):.6f} | {metric.get('s0_correct_harm_rate', 0.0):.6f} | {metric.get('mean_regret_to_gt_logp_oracle', 0.0):.6f} |")
    lines.extend([
        "",
        "## Full Val reference",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ])
    for name, metric in references_full.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend([
        "",
        "Full-Val visual branch metrics are intentionally not reported because the existing visited s0 DINO cache covers only the moving-Val contexts; generating missing DINO is prohibited in this experiment.",
        "",
        f"Best visual branch: **{visual_best}**, Moving Accuracy gain over FrozenStageCv0: **{visual_gain:+.3f} pp**.",
        "",
        "## Scientific decision",
        "",
        ("The best visual-context branch reaches at least +2 pp over FrozenStageCv0, supporting a scene/occlusion visual-representation bottleneck." if visual_gain >= 2.0 else "All visual-context branches stay below +2 pp over FrozenStageCv0; this batch does not establish that a single visited s0 visual context is sufficient to predict candidate recognition outcomes."),
        "The fixed Stay-aware GTMargin listwise objective was reused without loss changes. No branch was connected to H2.",
        "",
        "## Focus classes",
        "",
        "Per-class Recall/F1 for bend, stumble, knock and touching face are in `per_class_metrics.json`.",
        "",
        "test_used=false; future_candidate_dino_used=false; skeleton/rgb/dino regenerated=false.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    result = run(args)
    print(json.dumps({"output": str(EXPERIMENT.resolve()), "best_visual": max((name for name in ("dino_mean", "dino_meanmax", "dino_spatial", "candidate_conditioned_spatial")), key=lambda name: result["metrics_moving"][name]["accuracy"]), "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
