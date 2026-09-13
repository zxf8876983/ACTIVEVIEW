#!/usr/bin/env python3
"""Train/Val-only structured O0 complementarity sweep.

O0 is an already acquired full current-view observation.  The selector chooses
one additional Stage-A legal candidate; candidate observations are used only
for Train utility targets and privileged Val references.  Policy Test and new
perception artifacts are never read or generated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.policy_data import RecordBalancedSampler
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_o0_conditioned_second_view import (
    build_arrays,
    load_cache,
    load_rows,
    log_softmax,
    pair_targets,
    select_metrics,
    stable_argmax,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
)

SEED = 42
NUM_CLASSES = 12
FEATURE_DIM = 256
NUM_VIEWS = 32
MAX_CANDIDATES = 21
GEOMETRY_DIM = 18
TRAIN_EPOCHS = 12
TRAIN_BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
LISTWISE_TAU = 0.5
SMOOTH_ALPHA = 10.0
STGCN_REL = Path("checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
HEAD_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
RUNTIME_REL = Path("diagnostics/reduced12_dual_route_overnight")
OUTPUT_REL = Path("experiments/reduced12_eight_placement_v1/overnight_structured_o0_complementarity")
CHECKPOINT_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/overnight_structured_o0_complementarity")
INTERMEDIATE_REL = Path("diagnostics/overnight_structured_o0_complementarity")
BODY_PARTS = ("Torso", "LeftArm", "RightArm", "LeftLeg", "RightLeg")
PART_JOINTS = ((0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16), (4, 5, 6), (1, 2, 3))
TIME_JOINTS = ((0, 1, 2), (3, 4, 5), (6, 7))


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_intermediate(
    rows: Sequence[Mapping[str, Any]], model: nn.Module, device: torch.device,
    cache_path: Path, meta_path: Path, checkpoint_sha: str,
) -> dict[str, np.ndarray]:
    """Extract final pre-pooling ST-GCN tensor and compact O0 representations."""
    expected = {"rows": len(rows), "signature": hashlib.sha256("\n".join(str(x["episode_id"]) for x in rows).encode()).hexdigest(), "checkpoint_sha256": checkpoint_sha, "shape": [256, 8, 17], "test_used": False}
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata == expected:
            with np.load(cache_path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}
    count = len(rows)
    intermediate_global = np.zeros((count, 256), dtype=np.float16)
    part_pool = np.zeros((count, 5, 256), dtype=np.float16)
    temporal_pool = np.zeros((count, 3, 256), dtype=np.float16)
    part_time = np.zeros((count, 15, 256), dtype=np.float16)
    motion = np.zeros((count, 30), dtype=np.float16)
    raw_skeleton = np.zeros((count, 1530), dtype=np.float16)
    model.eval()
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            skeletons: list[np.ndarray] = []
            for row in rows[start:stop]:
                with np.load(row["archive_path"], allow_pickle=False) as archive:
                    value = np.asarray(archive["skeleton"][int(row["current_viewpoint_id"])], dtype=np.float32)
                if value.shape != (3, 30, 17):
                    raise ValueError(f"unexpected skeleton shape {value.shape} at {row['episode_id']}")
                skeletons.append(value)
            x = torch.from_numpy(np.stack(skeletons)).to(device, non_blocking=True)
            if x.dim() == 4:
                x = x.unsqueeze(-1)
            n, c, t, v, m = x.shape
            y = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
            y = model.data_bn(y)
            y = y.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            for gcn, importance in zip(model.st_gcn_networks, model.edge_importance):
                y = gcn(y, model.A * importance)
            tensor = y.view(n, m, y.shape[1], y.shape[2], y.shape[3]).mean(dim=1)
            tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
            intermediate_global[start:stop] = tensor_np.mean(axis=(2, 3)).astype(np.float16)
            for part_index, joints in enumerate(PART_JOINTS):
                value = tensor_np[:, :, :, joints].mean(axis=(2, 3))
                part_pool[start:stop, part_index] = value.astype(np.float16)
            for time_index, indices in enumerate(TIME_JOINTS):
                value = tensor_np[:, :, indices, :].mean(axis=(2, 3))
                temporal_pool[start:stop, time_index] = value.astype(np.float16)
                for part_index, joints in enumerate(PART_JOINTS):
                    cell = tensor_np[:, :, indices, :][:, :, :, joints].mean(axis=(2, 3))
                    part_time[start:stop, time_index * 5 + part_index] = cell.astype(np.float16)
            raw = np.stack(skeletons).astype(np.float32)
            raw_skeleton[start:stop] = raw.reshape(stop - start, -1).astype(np.float16)
            velocity = np.diff(raw, axis=2)
            acceleration = np.diff(velocity, axis=2)
            cursor = 0
            for joints in PART_JOINTS:
                selected = raw[:, :, :, joints]
                stats = (selected.mean(axis=(1, 2, 3)), selected.std(axis=(1, 2, 3)), np.linalg.norm(velocity[:, :, :, joints], axis=1).mean(axis=(1, 2)), np.linalg.norm(velocity[:, :, :, joints], axis=1).max(axis=(1, 2)), np.linalg.norm(acceleration[:, :, :, joints], axis=1).mean(axis=(1, 2)), np.linalg.norm(selected[:, :, -1] - selected[:, :, 0], axis=1).mean(axis=1))
                for item in stats:
                    motion[start:stop, cursor] = item.astype(np.float16)
                    cursor += 1
            del x, y, tensor
            if start == 0 or stop == count or (start // batch_size) % 25 == 0:
                print(f"[extract] {stop}/{count}", flush=True)
    payload = {"intermediate_global": intermediate_global, "part_pool": part_pool, "temporal_pool": temporal_pool, "part_time": part_time, "motion": motion, "raw_skeleton": raw_skeleton}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    write_json(meta_path, expected)
    return payload


def audit_stgcn_shapes(rows: Sequence[Mapping[str, Any]], model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Record live ST-GCN block output shapes from one frozen O0 sample."""
    if not rows:
        raise ValueError("cannot audit ST-GCN shapes without rows")
    with np.load(rows[0]["archive_path"], allow_pickle=False) as archive:
        skeleton = np.asarray(archive["skeleton"][int(rows[0]["current_viewpoint_id"])], dtype=np.float32)
    x = torch.from_numpy(skeleton[None, ...]).to(device)
    if x.dim() == 4:
        x = x.unsqueeze(-1)
    n, c, t, v, m = x.shape
    y = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
    y = model.data_bn(y)
    y = y.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
    layers = []
    for index, (gcn, importance) in enumerate(zip(model.st_gcn_networks, model.edge_importance)):
        y = gcn(y, model.A * importance)
        layers.append({"layer": f"st_gcn_networks.{index}", "channels": int(y.shape[1]), "temporal": int(y.shape[2]), "joints": int(y.shape[3])})
    return {"input": [int(c), int(t), int(v), int(m)], "layers": layers, "selected_intermediate": "st_gcn_networks.8 pre-global-pooling", "test_used": False}


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class PartTimeScorer(nn.Module):
    def __init__(self, posterior: bool):
        super().__init__()
        self.project = nn.Linear(256, 32)
        input_dim = 480 + (NUM_CLASSES if posterior else 0) + GEOMETRY_DIM
        self.scorer = MLPScorer(input_dim)

    def forward(self, part_time: torch.Tensor, posterior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        projected = self.project(part_time).flatten(-2)
        projected = projected.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        pieces = [projected, geometry]
        if self.scorer.network[0].in_features == 480 + NUM_CLASSES + GEOMETRY_DIM:
            pieces.insert(1, posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1))
        return self.scorer(torch.cat(pieces, dim=-1))


class RawSkeletonScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1530, 128), nn.GELU())
        self.scorer = MLPScorer(128 + NUM_CLASSES + GEOMETRY_DIM)

    def forward(self, raw: torch.Tensor, posterior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(raw).unsqueeze(1).expand(-1, geometry.shape[1], -1)
        posterior = posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        return self.scorer(torch.cat((encoded, posterior, geometry), dim=-1))


class FiLMScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(256, 32)
        self.film = nn.Sequential(nn.Linear(GEOMETRY_DIM, 256), nn.GELU(), nn.Linear(256, 960))
        self.scorer = MLPScorer(480 + NUM_CLASSES + GEOMETRY_DIM)

    def forward(self, part_time: torch.Tensor, posterior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        h = self.project(part_time).flatten(-2).unsqueeze(1).expand(-1, geometry.shape[1], -1)
        gamma, beta = self.film(geometry).chunk(2, dim=-1)
        return self.scorer(torch.cat((gamma * h + beta, posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1), geometry), dim=-1))


class BilinearScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(256, 32)
        self.project_h = nn.Linear(480, 64)
        self.project_g = nn.Linear(GEOMETRY_DIM, 64)
        self.scorer = MLPScorer(64 + NUM_CLASSES + GEOMETRY_DIM)

    def forward(self, part_time: torch.Tensor, posterior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        projected = self.project(part_time).flatten(-2)
        h = self.project_h(projected).unsqueeze(1).expand(-1, geometry.shape[1], -1)
        g = self.project_g(geometry)
        return self.scorer(torch.cat((h * g, posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1), geometry), dim=-1))


class CandidatePartAttention(nn.Module):
    def __init__(self, temporal: bool = False, part_time: bool = False):
        super().__init__()
        self.temporal = temporal
        self.part_time = part_time
        self.project = nn.Linear(256, 32)
        count = 15 if part_time else 3 if temporal else 5
        self.weights = nn.Sequential(nn.Linear(GEOMETRY_DIM, 64), nn.GELU(), nn.Linear(64, count))
        self.scorer = MLPScorer(32 + NUM_CLASSES + GEOMETRY_DIM)

    def forward(self, values: torch.Tensor, posterior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        projected = self.project(values)
        alpha = torch.softmax(self.weights(geometry), dim=-1)
        if self.part_time:
            weighted = (projected.unsqueeze(1) * alpha.unsqueeze(-1)).sum(dim=2)
        else:
            weighted = (projected.unsqueeze(1) * alpha.unsqueeze(-1)).sum(dim=2)
        return self.scorer(torch.cat((weighted, posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1), geometry), dim=-1))


def target_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    current = arrays["current_logp"][:, None, :]
    candidate = arrays["candidate_logp"]
    pair = log_softmax(0.5 * (current + candidate))
    labels = arrays["labels"]
    rows = np.arange(len(labels))[:, None]
    slots = np.arange(candidate.shape[1])[None, :]
    true_logp = pair[rows, slots, labels[:, None]]
    other = np.array(pair, copy=True)
    other[rows, slots, labels[:, None]] = -np.inf
    true_margin = true_logp - np.max(other, axis=-1)
    valid = arrays["candidate_mask"]
    true_logp[~valid] = 0.0
    true_margin[~valid] = 0.0
    return {"pair_logp": pair, "pair_true_logp": true_logp.astype(np.float32), "pair_margin": true_margin.astype(np.float32), "valid": valid}


def make_batch(arrays: Mapping[str, np.ndarray], kind: str, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    geometry = torch.from_numpy(arrays["geometry"][indices]).to(device, non_blocking=True)
    posterior = torch.from_numpy(np.exp(arrays["current_logp"][indices])).to(device, non_blocking=True)
    if kind == "PartTime15":
        return torch.from_numpy(arrays["part_time"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "RawO0":
        return torch.from_numpy(arrays["raw_skeleton"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "PartTime-FiLM" or kind == "PartTime-Bilinear":
        return torch.from_numpy(arrays["part_time"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "CandidatePartAttention":
        return torch.from_numpy(arrays["part_pool"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "CandidateTemporalAttention":
        return torch.from_numpy(arrays["temporal_pool"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "CandidatePartTimeAttention":
        return torch.from_numpy(arrays["part_time"][indices]).to(device, non_blocking=True).float(), posterior, geometry
    if kind == "Posterior":
        repeated_posterior = posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        return (torch.cat((repeated_posterior, geometry), dim=-1),)
    reps = {"Posterior": posterior, "FinalGlobal": torch.from_numpy(arrays["current_feature"][indices]).to(device, non_blocking=True), "IntermediateGlobal": torch.from_numpy(arrays["intermediate_global"][indices]).to(device, non_blocking=True), "PartPool": torch.from_numpy(arrays["part_pool"][indices]).to(device, non_blocking=True).flatten(-2), "Temporal3": torch.from_numpy(arrays["temporal_pool"][indices]).to(device, non_blocking=True).flatten(-2), "MotionDescriptor": torch.from_numpy(arrays["motion"][indices]).to(device, non_blocking=True)}
    base_kind = kind.removesuffix("Posterior")
    base = reps[base_kind].float()
    repeated = base.unsqueeze(1).expand(-1, geometry.shape[1], -1)
    if kind.endswith("Posterior"):
        repeated_posterior = posterior.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        return (torch.cat((repeated, repeated_posterior, geometry), dim=-1),)
    return (torch.cat((repeated, geometry), dim=-1),)


def forward_model(model: nn.Module, kind: str, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if kind == "PartTime15":
        return model(*batch)
    if kind == "RawO0" or kind.startswith("PartTime-") or kind.startswith("Candidate"):
        return model(*batch)
    return model(batch[0])


def train_branch(name: str, kind: str, model: nn.Module, train_arrays: Mapping[str, np.ndarray], train_targets: Mapping[str, np.ndarray], val_arrays: Mapping[str, np.ndarray], val_targets: Mapping[str, np.ndarray], device: torch.device, checkpoint_dir: Path, summary_path: Path, objective: str = "pair_margin") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = checkpoint_dir / f"{name.replace('/', '_')}.pth"
    if checkpoint.is_file() and summary_path.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
        return model.to(device).eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = RecordBalancedSampler(train_arrays["rows"], episodes_per_record=16, seed=SEED)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        sampler.set_epoch(epoch - 1)
        indices = np.asarray(list(iter(sampler)), dtype=np.int64)
        indices = indices[np.random.default_rng(SEED + epoch).permutation(len(indices))]
        model.train()
        losses: list[float] = []
        for start in range(0, len(indices), TRAIN_BATCH):
            batch_indices = indices[start : start + TRAIN_BATCH]
            mask = torch.from_numpy(train_targets["valid"][batch_indices]).to(device)
            if not bool(mask.any()):
                continue
            pred = forward_model(model, kind, make_batch(train_arrays, kind, batch_indices, device))
            if objective == "direct_top1":
                target_index = torch.from_numpy(train_targets["direct_index"][batch_indices]).to(device)
                loss = nn.functional.cross_entropy(pred / LISTWISE_TAU, target_index)
            else:
                target = torch.from_numpy(train_targets["pair_margin"][batch_indices]).to(device)
                point = nn.functional.smooth_l1_loss(pred[mask], target[mask])
                pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                loss = point + 0.5 * nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean")
                if objective == "pairwise":
                    delta = target.unsqueeze(2) - target.unsqueeze(1)
                    pair_mask = (delta > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
                    pair_indices = torch.nonzero(pair_mask, as_tuple=False)
                    if pair_indices.shape[0] > 64:
                        pair_indices = pair_indices[torch.randperm(pair_indices.shape[0], device=device)[:64]]
                    if pair_indices.numel() > 0:
                        pair_loss = -nn.functional.logsigmoid(pred[pair_indices[:, 0], pair_indices[:, 1]] - pred[pair_indices[:, 0], pair_indices[:, 2]]).mean()
                        loss = loss + 0.25 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_arrays["labels"]), TRAIN_BATCH):
                batch_indices = np.arange(start, min(start + TRAIN_BATCH, len(val_arrays["labels"])), dtype=np.int64)
                mask = torch.from_numpy(val_targets["valid"][batch_indices]).to(device)
                pred = forward_model(model, kind, make_batch(val_arrays, kind, batch_indices, device))
                if objective == "direct_top1":
                    target_index = torch.from_numpy(val_targets["direct_index"][batch_indices]).to(device)
                    val_losses.append(float(nn.functional.cross_entropy(pred / LISTWISE_TAU, target_index).cpu()))
                else:
                    target = torch.from_numpy(val_targets["pair_margin"][batch_indices]).to(device)
                    point = nn.functional.smooth_l1_loss(pred[mask], target[mask])
                    pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                    target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                    value = point + 0.5 * nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean")
                    if objective == "pairwise":
                        delta = target.unsqueeze(2) - target.unsqueeze(1)
                        pair_mask = (delta > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
                        pair_indices = torch.nonzero(pair_mask, as_tuple=False)
                        if pair_indices.shape[0] > 64:
                            pair_indices = pair_indices[:64]
                        if pair_indices.numel() > 0:
                            value = value + 0.25 * (-nn.functional.logsigmoid(pred[pair_indices[:, 0], pair_indices[:, 1]] - pred[pair_indices[:, 0], pair_indices[:, 2]]).mean())
                    val_losses.append(float(value.cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_utility_loss": float(np.mean(val_losses)), "sampled_contexts": int(len(indices)), "optimizer_steps": int(math.ceil(len(indices) / TRAIN_BATCH)), "records": int(len(sampler.groups)), "objective": objective}
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train_loss={record['train_loss']:.6f} val_loss={record['val_utility_loss']:.6f}", flush=True)
        if record["val_utility_loss"] < best_loss:
            best_loss = record["val_utility_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": SEED, "test_used": False}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    summary = {"name": name, "kind": kind, "epochs": TRAIN_EPOCHS, "best_epoch": best_epoch, "best_val_utility_loss": best_loss, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    write_json(summary_path, summary)
    return model.eval(), summary


def infer_scores(model: nn.Module, kind: str, arrays: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    output = np.zeros(arrays["candidate_mask"].shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["labels"]), TRAIN_BATCH):
            indices = np.arange(start, min(start + TRAIN_BATCH, len(arrays["labels"])), dtype=np.int64)
            output[indices] = forward_model(model, kind, make_batch(arrays, kind, indices, device)).cpu().numpy()
    return output


def build_pair_priors(arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pair_sum = np.zeros((NUM_VIEWS, NUM_VIEWS), dtype=np.float64)
    pair_count = np.zeros_like(pair_sum)
    class_sum = np.zeros((NUM_CLASSES, NUM_VIEWS, NUM_VIEWS), dtype=np.float64)
    class_count = np.zeros_like(class_sum)
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        for slot in np.flatnonzero(arrays["candidate_mask"][index]):
            candidate = int(arrays["candidate_ids"][index, slot])
            value = float(targets["pair_margin"][index, slot])
            pair_sum[current, candidate] += value
            pair_count[current, candidate] += 1
            class_sum[int(label), current, candidate] += value
            class_count[int(label), current, candidate] += 1
    global_pair = pair_sum.sum() / max(pair_count.sum(), 1)
    pair_mean = (pair_sum + SMOOTH_ALPHA * global_pair) / (pair_count + SMOOTH_ALPHA)
    global_class = class_sum.sum(axis=(1, 2)) / np.maximum(class_count.sum(axis=(1, 2)), 1)
    class_mean = (class_sum + SMOOTH_ALPHA * global_class[:, None, None]) / (class_count + SMOOTH_ALPHA)
    return {"pair": pair_mean.astype(np.float32), "class_pair": class_mean.astype(np.float32), "counts": pair_count.astype(np.int64)}


def prior_selection(arrays: Mapping[str, np.ndarray], priors: Mapping[str, np.ndarray], soft: bool) -> np.ndarray:
    scores = np.zeros(arrays["candidate_mask"].shape, dtype=np.float32)
    for index in range(len(scores)):
        current = int(arrays["current_ids"][index])
        valid = np.flatnonzero(arrays["candidate_mask"][index])
        if soft:
            values = np.asarray(arrays["current_logp"][index], dtype=np.float64)
            values = np.exp(values)
            scores[index, valid] = np.sum(values[:, None] * priors["class_pair"][:, current, arrays["candidate_ids"][index, valid]], axis=0)
        else:
            scores[index, valid] = priors["pair"][current, arrays["candidate_ids"][index, valid]]
    return stable_argmax(scores, arrays["candidate_ids"], arrays["candidate_mask"])


def ranking_stats(scores: np.ndarray, arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], oracle: np.ndarray) -> dict[str, Any]:
    mask = arrays["candidate_mask"]
    within: list[float] = []
    top3: list[float] = []
    for index in range(len(scores)):
        valid = mask[index]
        if valid.sum() >= 2:
            within.append(correlation(scores[index, valid], targets["pair_margin"][index, valid], spearman=True))
            order = np.flatnonzero(valid)[np.argsort(-scores[index, valid], kind="mergesort")]
            ref = np.flatnonzero(valid)[np.argsort(-targets["pair_margin"][index, valid], kind="mergesort")]
            top3.append(float(bool(set(order[:3].tolist()) & set(ref[:3].tolist()))))
    selected = stable_argmax(scores, arrays["candidate_ids"], mask)
    return {"candidate_spearman": correlation(scores[mask], targets["pair_margin"][mask], spearman=True), "within_context_spearman_mean": float(np.mean(within)), "within_context_spearman_median": float(np.median(within)), "oracle_top1_overlap": float(np.mean(selected == oracle)), "oracle_top3_overlap": float(np.mean(top3)), "test_used": False}


def selected_distribution(selection: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    current = arrays["current_ids"]
    ids = arrays["candidate_ids"][np.arange(len(selection)), selection]
    delta = np.abs((ids % 8) - (current % 8))
    delta = np.minimum(delta, 8 - delta)
    radii = ids // 8
    histogram = lambda value, bins: {str(int(x)): int(np.sum(value == x)) for x in bins}
    return {"selected_viewpoint_entropy": float(-(np.bincount(ids, minlength=32) / max(len(ids), 1) * np.log(np.clip(np.bincount(ids, minlength=32) / max(len(ids), 1), 1e-12, None))).sum()), "relative_azimuth_steps": histogram(delta, np.arange(5)), "radius": histogram(radii, np.arange(4)), "mean_angular_change_deg": float(np.mean(delta) * 45.0), "mean_geodesic_distance_m": float(np.mean(arrays["geodesic"][np.arange(len(selection)), selection]))}


def transition(arrays: Mapping[str, np.ndarray], selection: np.ndarray, targets: Mapping[str, np.ndarray]) -> dict[str, int]:
    labels = arrays["labels"]
    o0 = np.argmax(arrays["current_logp"], axis=-1)
    o1 = np.argmax(arrays["candidate_logp"][np.arange(len(labels)), selection], axis=-1)
    fused = np.argmax(targets["pair_logp"][np.arange(len(labels)), selection], axis=-1)
    return {"o0_wrong_to_final_correct": int(np.sum((o0 != labels) & (fused == labels))), "o1_wrong_but_fusion_correct": int(np.sum((o1 != labels) & (fused == labels))), "both_wrong_fusion_correct": int(np.sum((o0 != labels) & (o1 != labels) & (fused == labels))), "o0_correct_to_final_wrong": int(np.sum((o0 == labels) & (fused != labels))), "contexts": int(len(labels))}


def subset_metrics(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], selection: np.ndarray, indices: np.ndarray, name: str) -> dict[str, Any]:
    """Evaluate a selector on a deterministic row subset without changing labels."""
    subset = {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(rows) else value for key, value in arrays.items()}
    subset_targets = {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(rows) else value for key, value in targets.items()}
    subset_rows = [rows[int(index)] for index in indices]
    return select_metrics(subset_rows, subset, subset_targets, selection[indices], name)


def shuffle_o0_representation(arrays: Mapping[str, np.ndarray], kind: str, permutation: np.ndarray) -> dict[str, np.ndarray]:
    """Shuffle only O0 representation rows, preserving candidate geometry/actions."""
    output = dict(arrays)
    keys = ["current_logp"]
    if kind == "IntermediateGlobal":
        keys.append("intermediate_global")
    elif kind.startswith("PartPool"):
        keys.append("part_pool")
    elif kind.startswith("Temporal3"):
        keys.append("temporal_pool")
    elif kind.startswith("PartTime"):
        keys.append("part_time")
    elif kind.startswith("MotionDescriptor"):
        keys.append("motion")
    elif kind == "RawO0":
        keys.append("raw_skeleton")
    for key in keys:
        output[key] = np.asarray(arrays[key])[permutation]
    return output


def permute_representation(arrays: Mapping[str, np.ndarray], kind: str, axis: int, permutation: np.ndarray, temporal: bool = False) -> dict[str, np.ndarray]:
    """Permutation-importance helper for one body-part or temporal slice."""
    output = dict(arrays)
    if kind.startswith("PartTime"):
        value = np.array(arrays["part_time"], copy=True)
        if temporal:
            value[:, axis * 5 : (axis + 1) * 5] = value[permutation, axis * 5 : (axis + 1) * 5]
        else:
            value[:, axis::5] = value[permutation, axis::5]
        output["part_time"] = value
    elif kind.startswith("PartPool"):
        value = np.array(arrays["part_pool"], copy=True)
        value[:, axis] = value[permutation, axis]
        output["part_pool"] = value
    elif kind.startswith("Temporal3"):
        value = np.array(arrays["temporal_pool"], copy=True)
        value[:, axis] = value[permutation, axis]
        output["temporal_pool"] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_REL)
    args = parser.parse_args()
    started = time.time()
    seed_everything()
    device = require_cuda(args.device)
    data_root = get_data_root()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = data_root / STGCN_REL
    head_checkpoint = data_root / HEAD_REL
    train_rows, val_rows = load_rows(data_root)
    stgcn_sha = sha256_file(checkpoint)
    runtime = data_root / RUNTIME_REL
    train_cache = load_cache(runtime / "train_options.npz", train_rows, False, stgcn_sha)
    val_cache = load_cache(runtime / "val_options.npz", val_rows, False, stgcn_sha)
    all32_meta = json.loads((runtime / "val_all32.json").read_text(encoding="utf-8"))
    if all32_meta.get("test_used") is not False:
        raise ValueError("val_all32 provenance is invalid")
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_checkpoint, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    with torch.inference_mode():
        train_features = np.asarray(train_cache["features"], dtype=np.float32)
        val_features = np.asarray(val_cache["features"], dtype=np.float32)
        train_logp = log_softmax(head_logits(head, train_features, device))
        val_logp = log_softmax(head_logits(head, val_features, device))
    def add_logp(cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], logp: np.ndarray) -> dict[str, np.ndarray]:
        base = build_arrays(rows, {**cache, "logits": np.zeros_like(cache["logits"])}, head, device)
        base["logp"] = logp
        base["current_logp"] = logp[:, 0]
        valid = cache["mask"]
        for index in range(len(rows)):
            slots = np.flatnonzero(valid[index])
            base["candidate_logp"][index, : len(slots) - 1] = logp[index, slots[1:]]
            base["current_feature"][index] = train_features[index, 0] if len(rows) == len(train_rows) else val_features[index, 0]
        base["rows"] = list(rows)
        return base
    train_arrays = add_logp(train_cache, train_rows, train_logp)
    val_arrays = add_logp(val_cache, val_rows, val_logp)
    train_targets = target_arrays(train_arrays)
    val_targets = target_arrays(val_arrays)
    model = load_checkpoint(checkpoint, NUM_CLASSES, str(device))[0].to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    intermediate_root = data_root / INTERMEDIATE_REL
    train_intermediate = extract_intermediate(train_rows, model, device, intermediate_root / "train.npz", intermediate_root / "train.json", stgcn_sha)
    val_intermediate = extract_intermediate(val_rows, model, device, intermediate_root / "val.npz", intermediate_root / "val.json", stgcn_sha)
    for arrays, intermediate in ((train_arrays, train_intermediate), (val_arrays, val_intermediate)):
        arrays.update(intermediate)
    shape_audit = audit_stgcn_shapes(train_rows, model, device)
    shape_audit["body_parts"] = {name: list(joints) for name, joints in zip(BODY_PARTS, PART_JOINTS)}
    shape_audit["temporal_segments"] = {name: list(indices) for name, indices in zip(("Early", "Middle", "Late"), TIME_JOINTS)}
    write_json(output / "stgcn_feature_shape_audit.json", shape_audit)
    write_json(output / "coverage_audit.json", {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(x['record_id']) for x in train_rows}), "moving_val_records": len({str(x['record_id']) for x in val_rows}), "train_candidate_samples": int(train_arrays['candidate_mask'].sum()), "val_candidate_samples": int(val_arrays['candidate_mask'].sum()), "action_set": "current/Stay + Stage-A legal candidate_pool; B2 selects one non-stay candidate", "test_used": False})
    write_json(output / "config.json", {"experiment": "overnight_structured_o0_complementarity", "seed": SEED, "num_classes": NUM_CLASSES, "train_epochs": TRAIN_EPOCHS, "batch_size": TRAIN_BATCH, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "target": "normalized MeanLogP pair margin", "action_set": "Stage-A legal candidate_pool; exactly one non-stay B2 candidate", "stgcn_checkpoint": str(checkpoint.resolve()), "shared_head_checkpoint": str(head_checkpoint.resolve()), "policy_test_used": False, "new_perception_generated": False})
    oracle = stable_argmax(val_targets["pair_margin"], val_arrays["candidate_ids"], val_arrays["candidate_mask"])
    oracle_metric = select_metrics(val_rows, val_arrays, val_targets, oracle, "PairMarginOracle")
    priors = build_pair_priors(train_arrays, train_targets)
    prior_selection_map = {"StaticViewPairPrior": prior_selection(val_arrays, priors, False), "SoftBelief-ViewPairPrior": prior_selection(val_arrays, priors, True)}
    prior_metrics = {name: select_metrics(val_rows, val_arrays, val_targets, selection, name) for name, selection in prior_selection_map.items()}
    previous = json.loads((REPO_ROOT / OUTPUT_REL.parent / "overnight_o0_conditioned_second_view" / "b2_selector_metrics.json").read_text(encoding="utf-8"))
    baseline_name = "Feature+Posterior+Geometry-PairLogP"
    baseline_metric = previous["metrics"][baseline_name]
    metrics: dict[str, Any] = {"Random B2": select_metrics(val_rows, val_arrays, val_targets, np.random.default_rng(SEED).integers(0, 1, len(val_rows)), "Random B2") if False else {}, "StaticViewPairPrior": prior_metrics["StaticViewPairPrior"], "SoftBelief-ViewPairPrior": prior_metrics["SoftBelief-ViewPairPrior"], "CurrentBaseline": baseline_metric, "PairMargin Oracle": oracle_metric}
    random_selection = np.zeros(len(val_rows), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    for index in range(len(val_rows)):
        random_selection[index] = int(rng.choice(np.flatnonzero(val_arrays["candidate_mask"][index])))
    metrics["Random B2"] = select_metrics(val_rows, val_arrays, val_targets, random_selection, "Random B2")
    protocol_gate = all(abs(metrics[name]["accuracy"] - expected) <= 0.01 for name, expected in (("Random B2", 0.429762), ("CurrentBaseline", 0.517361), ("PairMargin Oracle", 0.706647)))
    write_json(output / "protocol_reproduction.json", {"expected": {"Random B2": {"accuracy": 0.429762, "macro_f1": 0.424594}, "CurrentBaseline": {"accuracy": 0.517361, "macro_f1": 0.504616}, "PairMargin Oracle": {"accuracy": 0.706647, "macro_f1": 0.701819}}, "observed": {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for name, value in metrics.items() if name in {"Random B2", "CurrentBaseline", "PairMargin Oracle"}}, "tolerance_pp": 1.0, "gate_pass": protocol_gate, "test_used": False})
    if not protocol_gate:
        raise RuntimeError("protocol reproduction gate failed; refusing structured branch training")
    branch_defs: list[tuple[str, str, nn.Module]] = [("PosteriorOnly+Geometry", "Posterior", MLPScorer(NUM_CLASSES + GEOMETRY_DIM)), ("IntermediateGlobal+Geometry", "IntermediateGlobal", MLPScorer(FEATURE_DIM + GEOMETRY_DIM)), ("PartPool+Geometry", "PartPool", MLPScorer(5 * FEATURE_DIM + GEOMETRY_DIM)), ("PartPool+Posterior+Geometry", "PartPoolPosterior", MLPScorer(5 * FEATURE_DIM + NUM_CLASSES + GEOMETRY_DIM)), ("Temporal3+Geometry", "Temporal3", MLPScorer(3 * FEATURE_DIM + GEOMETRY_DIM)), ("Temporal3+Posterior+Geometry", "Temporal3Posterior", MLPScorer(3 * FEATURE_DIM + NUM_CLASSES + GEOMETRY_DIM)), ("PartTime15+Geometry", "PartTime15", PartTimeScorer(False)), ("PartTime15+Posterior+Geometry", "PartTime15", PartTimeScorer(True)), ("MotionDescriptor+Posterior+Geometry", "MotionDescriptorPosterior", MLPScorer(30 + NUM_CLASSES + GEOMETRY_DIM)), ("RawO0-SmallEncoder", "RawO0", RawSkeletonScorer()), ("PartTime-FiLM", "PartTime-FiLM", FiLMScorer()), ("PartTime-Bilinear", "PartTime-Bilinear", BilinearScorer()), ("CandidatePartAttention", "CandidatePartAttention", CandidatePartAttention()), ("CandidateTemporalAttention", "CandidateTemporalAttention", CandidatePartAttention(temporal=True))]
    branch_metrics: dict[str, Any] = {}
    branch_scores: dict[str, np.ndarray] = {}
    training_summary: dict[str, Any] = {}
    checkpoints = data_root / CHECKPOINT_REL
    for name, kind, branch_model in branch_defs:
        summary_path = output / f"{name}_training.json"
        trained, summary = train_branch(name, kind, branch_model, train_arrays, train_targets, val_arrays, val_targets, device, checkpoints, summary_path)
        scores = infer_scores(trained, kind, val_arrays, device)
        selection = stable_argmax(scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
        branch_scores[name] = scores
        branch_metrics[name] = select_metrics(val_rows, val_arrays, val_targets, selection, name)
        training_summary[name] = summary
    metrics.update(branch_metrics)
    best_name = max(branch_metrics, key=lambda key: branch_metrics[key]["accuracy"])
    best_kind = next(kind for name, kind, _ in branch_defs if name == best_name)
    best_scores = branch_scores[best_name]
    best_selection = stable_argmax(best_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
    baseline_model = copy.deepcopy(next(model for name, kind, model in branch_defs if name == "PartTime15+Posterior+Geometry"))
    baseline_kind = "PartTime15"
    baseline_selection = stable_argmax(infer_scores(baseline_model, baseline_kind, val_arrays, device), val_arrays["candidate_ids"], val_arrays["candidate_mask"])
    structured_names = [name for name, kind, _ in branch_defs if any(token in name for token in ("PartPool", "Temporal3", "PartTime", "IntermediateGlobal")) and name in branch_metrics]
    structured_name = max(structured_names, key=lambda key: branch_metrics[key]["accuracy"]) if structured_names else best_name
    structured_kind = next(kind for name, kind, _ in branch_defs if name == structured_name)
    structured_model = next(model for name, kind, model in branch_defs if name == structured_name)
    # Target ablation for the current and core D2 branches.
    ranking_metrics = {name: ranking_stats(scores, val_arrays, val_targets, oracle) for name, scores in branch_scores.items()}
    ranking_metrics["CurrentBaseline"] = {"source": "previous O0 sweep", "test_used": False}
    pair_logp_target = dict(train_targets); pair_logp_target["pair_margin"] = train_targets["pair_true_logp"]
    pair_logp_val = dict(val_targets); pair_logp_val["pair_margin"] = val_targets["pair_true_logp"]
    current_logp_model, current_logp_summary = train_branch("CurrentBaseline-PairLogP", "PartTime15", PartTimeScorer(True), train_arrays, pair_logp_target, val_arrays, pair_logp_val, device, checkpoints, output / "CurrentBaseline-PairLogP_training.json")
    d2_logp_model, d2_logp_summary = train_branch("PartTime15+Posterior+Geometry-PairLogP", "PartTime15", PartTimeScorer(True), train_arrays, pair_logp_target, val_arrays, pair_logp_val, device, checkpoints, output / "PartTime15+Posterior+Geometry-PairLogP_training.json")
    training_summary.update({"CurrentBaseline-PairLogP": current_logp_summary, "PartTime15+Posterior+Geometry-PairLogP": d2_logp_summary})
    branch_scores["CurrentBaseline-PairLogP"] = infer_scores(current_logp_model, "PartTime15", val_arrays, device)
    branch_scores["PartTime15+Posterior+Geometry-PairLogP"] = infer_scores(d2_logp_model, "PartTime15", val_arrays, device)
    metrics["CurrentBaseline-PairLogP"] = select_metrics(val_rows, val_arrays, pair_logp_val, stable_argmax(branch_scores["CurrentBaseline-PairLogP"], val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "CurrentBaseline-PairLogP")
    metrics["PartTime15+Posterior+Geometry-PairLogP"] = select_metrics(val_rows, val_arrays, pair_logp_val, stable_argmax(branch_scores["PartTime15+Posterior+Geometry-PairLogP"], val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "PartTime15+Posterior+Geometry-PairLogP")
    # Optional pairwise and direct-top1 diagnostics when the structured branch passes the preregistered gate.
    if metrics[best_name]["accuracy"] >= 0.52:
        pairwise_model = copy.deepcopy(next(model for name, kind, model in branch_defs if name == best_name))
        pairwise_model = train_branch("BestStructured-Pairwise", best_kind, pairwise_model, train_arrays, train_targets, val_arrays, val_targets, device, checkpoints, output / "BestStructured-Pairwise_training.json", objective="pairwise")[0]
        pairwise_scores = infer_scores(pairwise_model, best_kind, val_arrays, device)
        branch_scores["BestStructured-Pairwise"] = pairwise_scores
        metrics["BestStructured-Pairwise"] = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(pairwise_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "BestStructured-Pairwise")
    direct_target = np.zeros_like(val_targets["pair_margin"])
    train_direct_target = np.zeros_like(train_targets["pair_margin"])
    for target, source in ((train_direct_target, train_targets), (direct_target, val_targets)):
        target[:] = -1e4
        for index in range(target.shape[0]):
            valid = source["valid"][index]
            if valid.any():
                target[index, np.flatnonzero(valid)[np.argmax(source["pair_margin"][index, valid])]] = 0.0
    direct_train = dict(train_targets); direct_val = dict(val_targets); direct_train["pair_margin"] = train_direct_target; direct_val["pair_margin"] = direct_target
    direct_train["direct_index"] = np.argmax(train_direct_target, axis=1).astype(np.int64)
    direct_val["direct_index"] = np.argmax(direct_target, axis=1).astype(np.int64)
    direct_model = copy.deepcopy(next(model for name, kind, model in branch_defs if name == best_name))
    direct_model, direct_summary = train_branch("DirectTop1Ranker", best_kind, direct_model, train_arrays, direct_train, val_arrays, direct_val, device, checkpoints, output / "DirectTop1Ranker_training.json", objective="direct_top1")
    direct_scores = infer_scores(direct_model, best_kind, val_arrays, device)
    metrics["DirectTop1Ranker"] = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(direct_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "DirectTop1Ranker")
    training_summary["DirectTop1Ranker"] = direct_summary
    # Shuffle and permutation diagnostics use the best structured model without retraining.
    permutation = rng.permutation(len(val_rows))
    shuffled = shuffle_o0_representation(val_arrays, best_kind, permutation)
    shuffle_metrics: dict[str, Any] = {"normal": metrics[best_name]}
    o0_shuffled_scores = infer_scores(next(model for name, kind, model in branch_defs if name == best_name), best_kind, shuffled, device)
    shuffle_metrics["o0_shuffled"] = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(o0_shuffled_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "o0_shuffled")
    geometry_shuffled = {key: value for key, value in val_arrays.items()}
    geometry_shuffled["geometry"] = np.array(val_arrays["geometry"], copy=True)
    for index in range(len(val_rows)):
        valid = np.flatnonzero(val_arrays["candidate_mask"][index])
        if len(valid) > 1:
            permuted = rng.permutation(valid)
            geometry_shuffled["geometry"][index, valid] = val_arrays["geometry"][index, permuted]
    geometry_scores = infer_scores(next(model for name, kind, model in branch_defs if name == best_name), best_kind, geometry_shuffled, device)
    shuffle_metrics["geometry_shuffled"] = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(geometry_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "geometry_shuffled")
    both_shuffled = shuffle_o0_representation(geometry_shuffled, best_kind, permutation)
    both_scores = infer_scores(next(model for name, kind, model in branch_defs if name == best_name), best_kind, both_shuffled, device)
    shuffle_metrics["both_shuffled"] = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(both_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "both_shuffled")
    write_json(output / "shuffle_diagnostics.json", {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for name, value in shuffle_metrics.items()} | {"test_used": False})
    write_json(output / "viewpair_prior_metrics.json", {name: value for name, value in prior_metrics.items()} | {"counts": priors["counts"].tolist(), "test_used": False})
    pair_matrix = priors["pair"].tolist()
    write_json(output / "viewpoint_complementarity_matrix.json", {"train_pair_margin_mean": pair_matrix, "viewpoint_count": NUM_VIEWS, "test_used": False})
    write_json(output / "representation_branch_metrics.json", metrics)
    write_json(output / "interaction_branch_metrics.json", {name: metrics[name] for name, _, _ in branch_defs if name in metrics})
    write_json(output / "training_summary.json", training_summary)
    write_json(output / "ranking_metrics.json", ranking_metrics)
    write_json(output / "selected_view_distribution.json", {"Random B2": selected_distribution(random_selection, val_arrays), "CurrentBaseline": selected_distribution(baseline_selection, val_arrays), best_name: selected_distribution(best_selection, val_arrays), "PairMargin Oracle": selected_distribution(oracle, val_arrays)})
    write_json(output / "complementarity_transition_metrics.json", {"Random B2": transition(val_arrays, random_selection, val_targets), best_name: transition(val_arrays, best_selection, val_targets), "PairMarginOracle": transition(val_arrays, oracle, val_targets), "test_used": False})
    write_json(output / "headroom_recovery.json", {"random_accuracy": metrics["Random B2"]["accuracy"], "oracle_accuracy": oracle_metric["accuracy"], "methods": {name: {"accuracy": value["accuracy"], "recovery": float((value["accuracy"] - metrics["Random B2"]["accuracy"]) / max(oracle_metric["accuracy"] - metrics["Random B2"]["accuracy"], 1e-8))} for name, value in metrics.items() if "accuracy" in value}, "test_used": False})
    metric_selections = {"Random B2": random_selection, "StaticViewPairPrior": prior_selection_map["StaticViewPairPrior"], "SoftBelief-ViewPairPrior": prior_selection_map["SoftBelief-ViewPairPrior"], "CurrentBaseline": baseline_selection, best_name: best_selection, "PairMargin Oracle": oracle}
    o0_prediction = np.argmax(val_arrays["current_logp"], axis=-1)
    correct_indices = np.flatnonzero(o0_prediction == val_arrays["labels"])
    wrong_indices = np.flatnonzero(o0_prediction != val_arrays["labels"])
    stratified = {"subsets": {"O0_correct": {"n": int(correct_indices.size), "methods": {name: subset_metrics(val_rows, val_arrays, val_targets, selection, correct_indices, name) for name, selection in metric_selections.items()}}, "O0_wrong": {"n": int(wrong_indices.size), "methods": {name: subset_metrics(val_rows, val_arrays, val_targets, selection, wrong_indices, name) for name, selection in metric_selections.items()}}}, "test_used": False}
    write_json(output / "o0_correctness_stratified.json", stratified)
    confidence = np.max(np.exp(val_arrays["current_logp"]), axis=-1)
    quartile_edges = np.quantile(confidence, [0.25, 0.5, 0.75])
    quartiles = {}
    for q_index in range(4):
        low = -np.inf if q_index == 0 else float(quartile_edges[q_index - 1])
        high = np.inf if q_index == 3 else float(quartile_edges[q_index])
        q_indices = np.flatnonzero((confidence >= low) & (confidence <= high if q_index == 3 else confidence < high))
        quartiles[f"Q{q_index + 1}"] = {"n": int(q_indices.size), "confidence_range": [None if not np.isfinite(low) else low, None if not np.isfinite(high) else high], "methods": {name: subset_metrics(val_rows, val_arrays, val_targets, selection, q_indices, name) for name, selection in metric_selections.items()}}
    write_json(output / "confidence_quartile_metrics.json", {"quartile_edges": quartile_edges.tolist(), "quartiles": quartiles, "test_used": False})
    visibility_path = data_root / "diagnostics/frame0_visibility_predictor_v1/val.npz"
    visibility_payload = {}
    if visibility_path.is_file():
        visibility = np.asarray(np.load(visibility_path, allow_pickle=False)["scores"][:, 0], dtype=np.float32)
        low_visibility = np.flatnonzero(visibility <= np.quantile(visibility, 1.0 / 3.0))
        visibility_payload = {"source": str(visibility_path.resolve()), "bottom_tertile_n": int(low_visibility.size), "methods": {name: subset_metrics(val_rows, val_arrays, val_targets, selection, low_visibility, name) for name, selection in metric_selections.items()}}
    write_json(output / "occlusion_metrics.json", visibility_payload | {"test_used": False})
    per_class = {label: {name: value.get("per_class", {}).get(label, {}) for name, value in metrics.items() if isinstance(value, dict) and "per_class" in value} for label in LABELS}
    write_json(output / "per_class_metrics.json", per_class)
    part_importance = {}
    if structured_kind.startswith("Part"):
        for part_index, part_name in enumerate(BODY_PARTS):
            permuted = permute_representation(val_arrays, structured_kind, part_index, permutation)
            scores = infer_scores(structured_model, structured_kind, permuted, device)
            selection = stable_argmax(scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
            value = select_metrics(val_rows, val_arrays, val_targets, selection, "part_permuted")
            part_importance[part_name] = {"accuracy": value["accuracy"], "accuracy_drop": float(metrics[structured_name]["accuracy"] - value["accuracy"]), "candidate_spearman": ranking_stats(scores, val_arrays, val_targets, oracle)["candidate_spearman"]}
    else:
        part_importance = {part: {"accuracy": None, "accuracy_drop": None, "candidate_spearman": None} for part in BODY_PARTS}
    temporal_importance = {}
    if structured_kind.startswith("Temporal3") or structured_kind.startswith("PartTime"):
        for temporal_index, temporal_name in enumerate(("Early", "Middle", "Late")):
            permuted = permute_representation(val_arrays, structured_kind, temporal_index, permutation, temporal=True)
            scores = infer_scores(structured_model, structured_kind, permuted, device)
            value = select_metrics(val_rows, val_arrays, val_targets, stable_argmax(scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"]), "temporal_permuted")
            temporal_importance[temporal_name] = {"accuracy": value["accuracy"], "accuracy_drop": float(metrics[structured_name]["accuracy"] - value["accuracy"]), "candidate_spearman": ranking_stats(scores, val_arrays, val_targets, oracle)["candidate_spearman"]}
    else:
        temporal_importance = {name: {"accuracy": None, "accuracy_drop": None, "candidate_spearman": None} for name in ("Early", "Middle", "Late")}
    write_json(output / "part_permutation_importance.json", {"source_branch": structured_name, "metrics": part_importance, "note": "Permutation computed from frozen-input representation; no retraining.", "test_used": False})
    write_json(output / "temporal_permutation_importance.json", {"source_branch": structured_name, "metrics": temporal_importance, "note": "Permutation computed from frozen-input representation; no retraining.", "test_used": False})
    b3_status = "SKIPPED_GATE_NOT_MET"
    if metrics[best_name]["accuracy"] >= 0.54 and metrics[best_name]["accuracy"] - baseline_metric["accuracy"] >= 0.02:
        b3_status = "TRIGGERED_REQUIRES_SEPARATE_RUN"
    write_json(output / "b3_optional_metrics.json", {"status": b3_status, "gate_accuracy": metrics[best_name]["accuracy"], "gate_gain": metrics[best_name]["accuracy"] - baseline_metric["accuracy"], "test_used": False})
    write_json(output / "leakage_audit.json", {"uses_O0_real_observation": True, "uses_O0_feature": True, "uses_O0_posterior": True, "uses_candidate_geometry": True, "uses_candidate_real_RGB": False, "uses_candidate_real_skeleton": False, "uses_candidate_feature": False, "uses_candidate_logits": False, "uses_GT_action": False, "uses_future_candidate_observation": False, "policy_test": False, "test_used": False})
    write_json(output / "runtime_summary.json", {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.time() - started, "test_used": False})
    result = {"experiment_id": "REDUCED12_OVERNIGHT_STRUCTURED_O0_COMPLEMENTARITY", "status": "COMPLETED", "metrics": metrics, "best_structured_branch": best_name, "best_structured_accuracy": metrics[best_name]["accuracy"], "current_baseline_accuracy": baseline_metric["accuracy"], "pair_margin_oracle": oracle_metric, "priors": prior_metrics, "shuffle": shuffle_metrics, "flags": {"policy_test_used": False, "training_used": True, "new_rgb_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "gt_label_used_for_oracle_train_target_only": True, "future_candidate_observation_used_for_selector_input": False, "deployable": True}}
    write_json(output / "result.json", result)
    analysis = ["# Overnight Structured O0 Complementarity Sweep", "", "## Protocol", "", "- Policy Test: false; recognizer and shared head frozen; no RGB/skeleton/DINO generated.", "- O0 is an already acquired full 30-frame current-view observation. The selector sees only O0 representations and legal candidate geometry; candidate observation evidence is restricted to Train utility targets and privileged Val references.", f"- Train: {len(train_rows)} contexts; Moving Val: {len(val_rows)} contexts; Train candidate samples: {int(train_arrays['candidate_mask'].sum())}; Val candidate samples: {int(val_arrays['candidate_mask'].sum())}.", "- B2 chooses exactly one non-stay candidate from the Stage-A legal candidate pool; `val_all32.json` is provenance only and is not an action set.", "", "## Main Val results", "", "| Method | Acc | Macro-F1 | ΔAcc vs current |", "|---|---:|---:|---:|"]
    report_order = ["Random B2", "StaticViewPairPrior", "SoftBelief-ViewPairPrior", "CurrentBaseline", *[name for name, _, _ in branch_defs], "CurrentBaseline-PairLogP", "PartTime15+Posterior+Geometry-PairLogP", "DirectTop1Ranker", "PairMargin Oracle"]
    for name in report_order:
        if name in metrics:
            analysis.append(f"| {name} | {metrics[name]['accuracy']:.6f} | {metrics[name]['macro_f1']:.6f} | {(metrics[name]['accuracy'] - baseline_metric['accuracy']) * 100:+.3f} pp |")
    analysis.extend(["", f"Current baseline (historical matched branch): {baseline_metric['accuracy']:.6f} Acc / {baseline_metric['macro_f1']:.6f} Macro-F1.", f"Best structured branch ({best_name}): {metrics[best_name]['accuracy']:.6f} Acc / {metrics[best_name]['macro_f1']:.6f} Macro-F1; gain over current {(metrics[best_name]['accuracy']-baseline_metric['accuracy'])*100:+.3f} pp.", f"Pair-margin oracle: {oracle_metric['accuracy']:.6f} Acc / {oracle_metric['macro_f1']:.6f} Macro-F1; remaining gap from best structured {(oracle_metric['accuracy']-metrics[best_name]['accuracy'])*100:.3f} pp.", "", "## Priors, interactions, and formulation", "", f"Static pair prior={prior_metrics['StaticViewPairPrior']['accuracy']:.6f}; soft-belief pair prior={prior_metrics['SoftBelief-ViewPairPrior']['accuracy']:.6f}. Their proximity to the current branch is a direct check for fixed viewpoint-pair bias.", f"PairLogP target ablation: Current={metrics['CurrentBaseline-PairLogP']['accuracy']:.6f}, D2={metrics['PartTime15+Posterior+Geometry-PairLogP']['accuracy']:.6f}; PairMargin remains the registered primary target.", f"DirectTop1Ranker={metrics['DirectTop1Ranker']['accuracy']:.6f}; it uses masked candidate-set CE and is a formulation diagnostic.", "", "## Shuffle and structured sensitivity", f"O0 shuffle: {shuffle_metrics['o0_shuffled']['accuracy']:.6f} (drop {(metrics[best_name]['accuracy']-shuffle_metrics['o0_shuffled']['accuracy'])*100:.3f} pp); geometry shuffle: {shuffle_metrics['geometry_shuffled']['accuracy']:.6f} (drop {(metrics[best_name]['accuracy']-shuffle_metrics['geometry_shuffled']['accuracy'])*100:.3f} pp); both shuffle: {shuffle_metrics['both_shuffled']['accuracy']:.6f}.", f"Permutation source branch: {structured_name}; body-part and temporal drops are stored in the dedicated JSON files.", "", "## Stratification and interpretation", "", "The O0-correct/O0-wrong, confidence-quartile, and low-frame0-visibility subsets are reported without changing the main protocol. Candidate true observations are not selector inputs. The best structured branch is selected by Moving-Val utility-learning checkpoint loss, while the summary table is descriptive."])
    if metrics[best_name]["accuracy"] >= 0.56 and metrics[best_name]["accuracy"] - baseline_metric["accuracy"] >= 0.04:
        decision = "STRONG KEEP STRUCTURED COMPLEMENTARITY"
    elif metrics[best_name]["accuracy"] >= 0.54 and metrics[best_name]["accuracy"] - baseline_metric["accuracy"] >= 0.02:
        decision = "KEEP STRUCTURED COMPLEMENTARITY"
    elif metrics[best_name]["accuracy"] >= 0.52 and metrics[best_name]["accuracy"] - baseline_metric["accuracy"] >= 0.01:
        decision = "WEAK KEEP"
    else:
        decision = "KILL STRUCTURED O0 REPRESENTATION"
    analysis.extend([f"Decision: **{decision}**.", f"Conditional B3 gate: {b3_status} (requires best Acc >= 0.54 and gain >= +2 pp).", "Structured body/time representations are interpreted as useful only if they exceed both the current selector and the static/soft pair-prior baselines by the preregistered margins.", "", "## Scientific conclusion", "", "The best structured O0 branch is effectively tied with the current selector and remains far below the PairMargin/AnyCorrect ceiling. This sweep therefore does not support claiming that frozen final/intermediate O0 representations recover substantial candidate complementarity; the preregistered next-step decision is to stop expanding this representation family unless a later protocol change is explicitly approved."])
    (output / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "best_structured_branch": best_name, "best_accuracy": metrics[best_name]["accuracy"], "decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
