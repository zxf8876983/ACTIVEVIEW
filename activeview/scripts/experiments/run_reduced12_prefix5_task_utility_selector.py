#!/usr/bin/env python3
"""Train causal Prefix-5 task-utility selectors on Policy Train.

The selector sees only the current 0:5 skeleton prefix, the current frame-0
DINO cache and the Stage-A legal candidate geometry.  Frozen recognizer
outputs from a strict mixed-view sequence are supervision targets and terminal
evaluation only.  Policy Test is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation, gt_margin
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (
    _option_geometry,
)
from activeview.scripts.eval.analyze_reduced12_short_prefix5_protocol import _infer_mixed

SEED = 42
NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
FEATURE_DIM = 256
PREFIX_FRAMES = 5
PREFIX_DIM = 3 * PREFIX_FRAMES * 17
DINO_DIM = 768
OPTION_GEOMETRY_DIM = 18
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 512
EVAL_BATCH = 1024
TAU = 0.5

POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
DIAGNOSTIC_RELATIVE = Path("diagnostics/reduced12_dual_route_overnight")
DINO_RELATIVE = Path("features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current")
TARGET_RELATIVE = Path("diagnostics/reduced12_prefix5_task_utility_selector_v1")
CHECKPOINT_RELATIVE = Path("checkpoints/policy_reduced12_eight_placement_v1/prefix5_task_utility_selector_v1")
STGCN_RELATIVE = Path("checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
SHARED_HEAD_RELATIVE = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
VISIBILITY_RELATIVE = Path("diagnostics/frame0_visibility_predictor_v1/val.npz")
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/prefix5_task_utility_predictor_v1"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load_option_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str, head: SharedHead, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = data_root / DIAGNOSTIC_RELATIVE
    metadata = json.loads((root / f"{split}_options.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"{split} option cache signature mismatch")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"{split} option cache is marked as Test-derived")
    with np.load(root / f"{split}_options.npz", allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=np.int64)
        mask = np.asarray(archive["mask"], dtype=bool)
    if features.shape != (len(rows), MAX_OPTIONS, FEATURE_DIM) or ids.shape != mask.shape or ids.shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"invalid {split} option cache shape")
    for index, row in enumerate(rows):
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        valid = np.flatnonzero(mask[index]).tolist()
        if ids[index, valid].tolist() != expected:
            raise ValueError(f"{split} action set mismatch at {row['episode_id']}")
    raw_logits = head_logits(head, features, device)
    shifted = raw_logits.astype(np.float64) - raw_logits.max(axis=-1, keepdims=True)
    logp = (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)
    return logp, features, ids, mask


def _load_cached_dino_mean(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> tuple[np.ndarray, dict[str, Any]]:
    root = data_root / DINO_RELATIVE
    meta_path, array_path = root / f"{split}.json", root / f"{split}.npy"
    if not meta_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"required cached frame-0 DINO is missing: {array_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"{split} DINO cache signature mismatch")
    if bool(metadata.get("test_used", True)) or bool(metadata.get("candidate_rgb_used", True)):
        raise ValueError(f"{split} DINO cache violates current-only protocol")
    values = np.load(array_path, mmap_mode="r")
    if values.shape != (len(rows), 16, DINO_DIM) or values.dtype != np.float16:
        raise ValueError(f"unexpected {split} DINO cache shape/dtype: {values.shape}/{values.dtype}")
    pooled = np.empty((len(rows), DINO_DIM), dtype=np.float32)
    for start in range(0, len(rows), EVAL_BATCH):
        pooled[start : start + EVAL_BATCH] = np.asarray(values[start : start + EVAL_BATCH], dtype=np.float32).mean(axis=1)
    return pooled, metadata


def _load_prefixes(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    prefixes = np.empty((len(rows), 3, PREFIX_FRAMES, 17), dtype=np.float32)
    for index, row in enumerate(rows):
        path = Path(str(row["archive_path"]))
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)):
                raise ValueError(f"invalid viewpoint ids in {path}")
            current = int(row["current_viewpoint_id"])
            values = np.asarray(archive["skeleton"], dtype=np.float32)
            if values.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(values).all():
                raise ValueError(f"invalid skeleton archive {path}")
            prefixes[index] = values[current, :, :PREFIX_FRAMES, :]
    return prefixes


def _mixed_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str, model: nn.Module, head: SharedHead, device: torch.device, current_logp: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    root = data_root / TARGET_RELATIVE
    path, meta_path = root / f"{split}_mixed5_logp.npz", root / f"{split}_mixed5_logp.json"
    checkpoint = data_root / STGCN_RELATIVE
    expected = {"split": split, "rows": len(rows), "signature": row_signature(rows), "prefix_frames": PREFIX_FRAMES, "checkpoint_sha256": _sha256(checkpoint), "shared_head_sha256": _sha256(data_root / SHARED_HEAD_RELATIVE), "test_used": False}
    if path.is_file() and meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) == expected:
        with np.load(path, allow_pickle=False) as archive:
            cached = np.asarray(archive["logp"], dtype=np.float32)
        if cached.shape == (len(rows), MAX_OPTIONS, NUM_CLASSES):
            return cached
    mixed_logits = _infer_mixed(rows, model, head, device)
    current_logits = np.log(np.clip(np.exp(current_logp), 1e-30, None)).astype(np.float32)
    mixed_logits[:, 0] = current_logits[:, 0]
    shifted = mixed_logits.astype(np.float64) - mixed_logits.max(axis=-1, keepdims=True)
    mixed_logp = (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)
    if not np.isfinite(mixed_logp[mask]).all():
        raise ValueError(f"non-finite mixed target values for {split}")
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logp=mixed_logp)
    _write(meta_path, expected)
    return mixed_logp


def _utilities(logp: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    utility = np.zeros(logp.shape[:2], dtype=np.float32)
    margin = np.zeros_like(utility)
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index]):
            utility[index, slot] = float(logp[index, slot, int(label)])
            margin[index, slot] = gt_margin(logp[index, slot], int(label))
    return utility, margin


def _target_stats(values: np.ndarray, labels: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = values[mask]
    best_slots = [int(np.flatnonzero(mask[index])[np.argmax(values[index, mask[index]])]) for index in range(len(labels))]
    return {
        "mean": float(valid.mean()), "std": float(valid.std()), "min": float(valid.min()), "max": float(valid.max()),
        "oracle_stay_rate": float(np.mean(np.asarray(best_slots) == 0)),
        "mean_legal_candidates": float(np.mean(mask[:, 1:].sum(axis=1))),
        "best_action_distribution": {str(int(action)): int(count) for action, count in zip(*np.unique(ids[np.arange(len(ids)), best_slots], return_counts=True))},
    }


class PrefixTaskUtilityModel(nn.Module):
    """Small causal candidate scorer; no future candidate observation enters."""

    def __init__(self, branch: str) -> None:
        super().__init__()
        self.branch = branch
        self.use_prefix = branch != "GeometryOnly-Mixed5"
        self.use_dino = branch == "Prefix5+RGBGlobal+Geometry"
        context_dim = 0
        if self.use_prefix:
            self.prefix_encoder = nn.Sequential(nn.Flatten(), nn.Linear(PREFIX_DIM, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
            context_dim += 64
        if self.use_dino:
            self.dino_encoder = nn.Sequential(nn.Linear(DINO_DIM, 64), nn.GELU())
            context_dim += 64
        self.geometry_encoder = nn.Sequential(nn.Linear(OPTION_GEOMETRY_DIM, 64), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(context_dim + 64, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, prefix: torch.Tensor | None, dino: torch.Tensor | None, geometry: torch.Tensor) -> torch.Tensor:
        context_parts: list[torch.Tensor] = []
        if self.use_prefix:
            if prefix is None:
                raise ValueError("prefix input is required")
            context_parts.append(self.prefix_encoder(prefix))
        if self.use_dino:
            if dino is None:
                raise ValueError("DINO input is required")
            context_parts.append(self.dino_encoder(dino))
        geometry_latent = self.geometry_encoder(geometry)
        if context_parts:
            context = torch.cat(context_parts, dim=-1).unsqueeze(1).expand(-1, geometry.shape[1], -1)
            values = torch.cat((context, geometry_latent), dim=-1)
        else:
            values = geometry_latent
        return self.scorer(values).squeeze(-1)


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for key in sorted(groups):
        candidates = np.asarray(groups[key], dtype=np.int64)
        selected.extend(rng.choice(candidates, OBS_PER_RECORD, replace=len(candidates) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(pred[mask], target[mask])
    terms: list[torch.Tensor] = []
    for index in range(pred.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_dist = torch.softmax(target[index, active] / TAU, dim=0)
            terms.append(-(target_dist * torch.log_softmax(pred[index, active] / TAU, dim=0)).sum())
    ranking = torch.stack(terms).mean() if terms else regression.new_zeros(())
    return regression + 0.5 * ranking, regression, ranking


def _predict(model: PrefixTaskUtilityModel, prefixes: np.ndarray | None, dino: np.ndarray | None, geometry: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(geometry), EVAL_BATCH):
            sl = slice(start, start + EVAL_BATCH)
            prefix = None if prefixes is None else torch.from_numpy(prefixes[sl]).to(device, non_blocking=True)
            visual = None if dino is None else torch.from_numpy(dino[sl]).to(device, non_blocking=True)
            geom = torch.from_numpy(geometry[sl]).to(device, non_blocking=True)
            outputs.append(model(prefix, visual, geom).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_branch(name: str, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_prefix: np.ndarray, val_prefix: np.ndarray, train_dino: np.ndarray, val_dino: np.ndarray, train_geometry: np.ndarray, val_geometry: np.ndarray, train_target: np.ndarray, val_target: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, device: torch.device, checkpoint: Path, summary_path: Path) -> PrefixTaskUtilityModel:
    model = PrefixTaskUtilityModel(name).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("branch") == name:
            model.load_state_dict(payload["state_dict"])
            return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    history: list[dict[str, Any]] = []
    best, best_epoch = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sampled = _sample_rows(train_rows, rng)
        losses: list[float] = []
        for start in range(0, len(sampled), TRAIN_BATCH):
            index = sampled[start : start + TRAIN_BATCH]
            prefix = None if name == "GeometryOnly-Mixed5" else torch.from_numpy(train_prefix[index]).to(device, non_blocking=True)
            dino = None if name != "Prefix5+RGBGlobal+Geometry" else torch.from_numpy(train_dino[index]).to(device, non_blocking=True)
            geometry = torch.from_numpy(train_geometry[index]).to(device, non_blocking=True)
            target = torch.from_numpy(train_target[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[index]).to(device, non_blocking=True)
            predicted = model(prefix, dino, geometry)
            total, _, _ = _loss(predicted, target, mask)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_rows), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                prefix = None if name == "GeometryOnly-Mixed5" else torch.from_numpy(val_prefix[sl]).to(device, non_blocking=True)
                dino = None if name != "Prefix5+RGBGlobal+Geometry" else torch.from_numpy(val_dino[sl]).to(device, non_blocking=True)
                geometry = torch.from_numpy(val_geometry[sl]).to(device, non_blocking=True)
                target = torch.from_numpy(val_target[sl]).to(device, non_blocking=True)
                mask = torch.from_numpy(val_mask[sl]).to(device, non_blocking=True)
                val_losses.append(float(_loss(model(prefix, dino, geometry), target, mask)[0].cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_utility_loss": float(np.mean(val_losses)), "sampled_contexts": int(len(sampled)), "records": int(len({str(row['record_id']) for row in train_rows}))}
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train={record['train_loss']:.6f} val={record['val_utility_loss']:.6f}", flush=True)
        if record["val_utility_loss"] < best:
            best, best_epoch = record["val_utility_loss"], epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "branch": name, "epoch": epoch, "seed": SEED, "test_used": False}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    _write(summary_path, {"branch": name, "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "best_epoch": best_epoch, "best_val_utility_loss": best, "checkpoint_selection": "minimum Moving Val Mixed5 utility loss", "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False})
    return model.eval()


def _select(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index in range(scores.shape[0]):
        active = np.flatnonzero(mask[index])
        ordered = sorted(active.tolist(), key=lambda slot: (-float(scores[index, slot]), 0 if slot == 0 else 1, int(ids[index, slot])))
        actions.append(int(ids[index, ordered[0]]))
    return actions


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action is not legal at row {index}: {action}")
        predictions.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _method(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    move = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result = classification(labels, predictions)
    result.update({"method": name, "move_rate": move, "stay_rate": 1.0 - move, "test_used": False})
    return result


def _ranking(name: str, rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid_pred, valid_target = predicted[mask], target[mask]
    within, top1, top3, stay_agreement = [], [], [], []
    for index in range(len(rows)):
        active = np.flatnonzero(mask[index])
        if active.size < 2:
            continue
        within.append(correlation(predicted[index, active], target[index, active], spearman=True))
        target_order = active[np.argsort(-target[index, active], kind="mergesort")]
        pred_order = active[np.argsort(-predicted[index, active], kind="mergesort")]
        top1.append(int(pred_order[0] == target_order[0]))
        top3.append(int(target_order[0] in pred_order[:3]))
        stay_agreement.append(int((pred_order[0] == 0) == (target_order[0] == 0)))
    return {"branch": name, "mae": float(np.mean(np.abs(valid_pred - valid_target))), "rmse": float(np.sqrt(np.mean((valid_pred - valid_target) ** 2))), "candidate_level_spearman": correlation(valid_pred, valid_target, spearman=True), "within_context_spearman_mean": float(np.mean(within)), "within_context_spearman_median": float(np.median(within)), "gt_utility_oracle_top1_overlap": float(np.mean(top1)), "gt_utility_oracle_top3_overlap": float(np.mean(top3)), "stay_move_agreement": float(np.mean(stay_agreement))}


def _transitions(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    return {"left_correct_right_wrong": int(np.sum((left == 1) & (right == 0))), "left_wrong_right_correct": int(np.sum((left == 0) & (right == 1))), "both_correct": int(np.sum((left == 1) & (right == 1))), "both_wrong": int(np.sum((left == 0) & (right == 0)))}


def _load_high_occlusion(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    with np.load(data_root / VISIBILITY_RELATIVE, allow_pickle=False) as archive:
        scores, ids, mask = np.asarray(archive["scores"]), np.asarray(archive["ids"]), np.asarray(archive["mask"])
    if scores.shape != (len(rows), MAX_OPTIONS) or ids.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("frame-0 visibility cache shape mismatch")
    return scores[:, 0] < np.quantile(scores[:, 0], 1.0 / 3.0)


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    branches = result["branches"]
    best = max(branches, key=lambda key: float(methods[key]["accuracy"]))
    acc = float(methods[best]["accuracy"])
    decision = "KILL" if acc < 0.52 else "BORDERLINE" if acc < 0.55 else "KEEP" if acc < 0.60 else "STRONG KEEP"
    random_acc = float(methods["Random-ShortPrefix5"]["accuracy"])
    oracle_acc = float(methods["Mixed5 GT-TrueLogP Oracle"]["accuracy"])
    gain_recovery = (acc - random_acc) / (oracle_acc - random_acc) if oracle_acc > random_acc else 0.0
    lines = [
        "# Prefix-5 Task-Utility Selector", "", "Train split: Policy Train (record-balanced, 313 records × 16 contexts/epoch)", "Model-selection split: Moving Val", "Policy Test used: false", "", "Protocol: discrete-time view-switch approximation; prefix L=5 uses current skeleton frames 0:5.", "Selector input: current 0:5 skeleton prefix, current frame-0 DINO global mean (branch C only), and Stage-A current/Stay + legal candidate geometry.", "No future candidate observation, skeleton, RGB, DINO, confidence, recognizer output, hard action or GT label enters selector inference.", "Target: strict mixed sequence current[0:5] + candidate[5:30]; Stay is current[0:30]. Terminal uses frozen ST-GCN encoder + frozen shared head.", "", "## Moving Val results", "", "| Method | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|",
    ]
    for name in ["Stay", "Random-ShortPrefix5", *branches, "Mixed5 GT-TrueLogP Oracle", "Mixed5 GT-Margin/AnyCorrect"]:
        metric = methods[name]
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines += ["", "## Ranking diagnostics", "", "| Branch | MAE | RMSE | Candidate Spearman | Within-context mean | Top-1 overlap | Top-3 overlap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in branches:
        metric = result["ranking_metrics"][name]
        lines.append(f"| {name} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['candidate_level_spearman']:.6f} | {metric['within_context_spearman_mean']:.6f} | {metric['gt_utility_oracle_top1_overlap']:.6f} | {metric['gt_utility_oracle_top3_overlap']:.6f} |")
    lines += ["", f"Best branch: **{best}** ({acc:.6f} Acc, {methods[best]['macro_f1']:.6f} Macro-F1).", f"Gain recovery versus Random-ShortPrefix5: {gain_recovery:.6f}; decision threshold: **{decision}** (kill <0.52, borderline 0.52–<0.55, keep ≥0.55, strong keep ≥0.60).", f"Prefix contribution is assessed by the Prefix5+Geometry shuffle diagnostic; RGB contribution by the Prefix5+RGBGlobal+Geometry shuffle diagnostic.", f"High-occlusion subset is the bottom tertile of frame-0 Stay SceneVisibility ({result['high_occlusion']['count']} contexts).", "The old frame-0 RGBGlobal Visibility (.494841) and Frame0 Task+VisibilityAux (.506151) values are different-protocol references only.", "", "No follow-up experiment was started automatically.", "", "```text", "policy_test_used=false", "training_split=Policy Train", "evaluation_split=Moving Val", "prefix5_mixed_protocol=true", "future_candidate_observation_used=false", "gt_action_used_for_selector=false", "frozen_recognizer=true", "```", ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    stgcn_path = data_root / STGCN_RELATIVE
    head_path = data_root / SHARED_HEAD_RELATIVE
    model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_logp, _, train_ids, train_mask = _load_option_cache(data_root, train_rows, "train", head, device)
    val_logp, _, val_ids, val_mask = _load_option_cache(data_root, val_rows, "val", head, device)
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    stats = json.loads((data_root / POLICY_RELATIVE / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geometry, _, _ = _option_geometry(train_rows, stats)
    val_geometry, _, _ = _option_geometry(val_rows, stats)
    train_prefix, val_prefix = _load_prefixes(train_rows), _load_prefixes(val_rows)
    train_dino, train_dino_meta = _load_cached_dino_mean(data_root, train_rows, "train")
    val_dino, val_dino_meta = _load_cached_dino_mean(data_root, val_rows, "val")
    train_target_logp = _mixed_cache(data_root, train_rows, "train", model, head, device, train_logp, train_ids, train_mask)
    val_target_logp = _mixed_cache(data_root, val_rows, "val", model, head, device, val_logp, val_ids, val_mask)
    train_target, train_margin = _utilities(train_target_logp, labels_train, train_mask)
    val_target, val_margin = _utilities(val_target_logp, labels_val, val_mask)
    val_oracle_actions = _select(val_target, val_ids, val_mask)
    val_oracle_pred = _terminal(val_target_logp, val_ids, val_mask, val_oracle_actions)
    val_oracle_metric = _method("Mixed5 GT-TrueLogP Oracle", val_rows, labels_val, val_oracle_pred, val_oracle_actions)
    if abs(float(val_oracle_metric["accuracy"]) - 0.687599) > 0.01:
        raise RuntimeError(f"Mixed5 GT-TrueLogP oracle protocol mismatch: {val_oracle_metric['accuracy']:.6f} (expected ~0.687599); selector training stopped")
    methods: dict[str, Any] = {}
    actions_by_method: dict[str, list[int]] = {}
    predictions: dict[str, np.ndarray] = {}
    stay = [int(row["current_viewpoint_id"]) for row in val_rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice(np.flatnonzero(val_mask[index]).tolist())) for index in range(len(val_rows))]
    random_actions = [int(val_ids[index, action]) for index, action in enumerate(random_actions)]
    for name, actions in (("Stay", stay), ("Random-ShortPrefix5", random_actions), ("Mixed5 GT-TrueLogP Oracle", val_oracle_actions)):
        actions_by_method[name] = actions
        predictions[name] = _terminal(val_target_logp, val_ids, val_mask, actions)
        methods[name] = _method(name, val_rows, labels_val, predictions[name], actions)
    margin_actions = _select(val_margin, val_ids, val_mask)
    actions_by_method["Mixed5 GT-Margin/AnyCorrect"] = margin_actions
    predictions["Mixed5 GT-Margin/AnyCorrect"] = _terminal(val_target_logp, val_ids, val_mask, margin_actions)
    methods["Mixed5 GT-Margin/AnyCorrect"] = _method("Mixed5 GT-Margin/AnyCorrect", val_rows, labels_val, predictions["Mixed5 GT-Margin/AnyCorrect"], margin_actions)
    branches = ["GeometryOnly-Mixed5", "Prefix5+Geometry", "Prefix5+RGBGlobal+Geometry"]
    ranking_metrics: dict[str, Any] = {}
    models: dict[str, PrefixTaskUtilityModel] = {}
    scores_by_branch: dict[str, np.ndarray] = {}
    for name in branches:
        checkpoint = data_root / CHECKPOINT_RELATIVE / f"{name}.pth"
        summary_path = output_root / f"{name}_training.json"
        model_branch = _train_branch(name, train_rows, val_rows, train_prefix, val_prefix, train_dino, val_dino, train_geometry, val_geometry, train_target, val_target, train_mask, val_mask, device, checkpoint, summary_path)
        models[name] = model_branch
        scores = _predict(model_branch, None if name == branches[0] else val_prefix, val_dino if name == branches[2] else None, val_geometry, device)
        scores_by_branch[name] = scores
        actions = _select(scores, val_ids, val_mask)
        actions_by_method[name] = actions
        predictions[name] = _terminal(val_target_logp, val_ids, val_mask, actions)
        methods[name] = _method(name, val_rows, labels_val, predictions[name], actions)
        ranking_metrics[name] = _ranking(name, val_rows, scores, val_target, val_mask)
    high_mask = _load_high_occlusion(data_root, val_rows)
    high_methods: dict[str, Any] = {}
    high_indices = np.flatnonzero(high_mask)
    for name in ("Stay", "Random-ShortPrefix5", *branches, "Mixed5 GT-TrueLogP Oracle"):
        high_rows = [val_rows[int(index)] for index in high_indices]
        high_actions = [actions_by_method[name][int(index)] for index in high_indices]
        high_methods[name] = _method(name, high_rows, labels_val[high_mask], predictions[name][high_mask], high_actions)
    best_branch = max(branches, key=lambda key: float(methods[key]["accuracy"]))
    transitions = {"best_branch": best_branch, "best_vs_geometry": _transitions(predictions[best_branch] == labels_val, predictions[branches[0]] == labels_val), "old_frame0_reference": {"accuracy": 0.506151, "comparison": "different protocol; no action-level replay"}}
    transitions["best_vs_geometry"].update({
        "geometry_wrong_best_correct": transitions["best_vs_geometry"]["left_correct_right_wrong"],
        "geometry_correct_best_wrong": transitions["best_vs_geometry"]["left_wrong_right_correct"],
    })
    shuffle_rng = np.random.default_rng(SEED)
    permutation = shuffle_rng.permutation(len(val_rows))
    shuffled_b = _predict(models["Prefix5+Geometry"], val_prefix[permutation], None, val_geometry, device)
    shuffled_b_actions = _select(shuffled_b, val_ids, val_mask)
    shuffled_b_pred = _terminal(val_target_logp, val_ids, val_mask, shuffled_b_actions)
    shuffled_c = _predict(models["Prefix5+RGBGlobal+Geometry"], val_prefix, val_dino[permutation], val_geometry, device)
    shuffled_c_actions = _select(shuffled_c, val_ids, val_mask)
    shuffled_c_pred = _terminal(val_target_logp, val_ids, val_mask, shuffled_c_actions)
    shuffle_diagnostics = {"prefix_geometry": {"normal": methods["Prefix5+Geometry"], "prefix_shuffled": _method("Prefix5+Geometry prefix-shuffled", val_rows, labels_val, shuffled_b_pred, shuffled_b_actions), "drop_accuracy": float(methods["Prefix5+Geometry"]["accuracy"] - np.mean(shuffled_b_pred == labels_val)), "permutation_seed": SEED}, "prefix_rgb_geometry": {"normal": methods["Prefix5+RGBGlobal+Geometry"], "rgb_shuffled": _method("Prefix5+RGBGlobal+Geometry RGB-shuffled", val_rows, labels_val, shuffled_c_pred, shuffled_c_actions), "drop_accuracy": float(methods["Prefix5+RGBGlobal+Geometry"]["accuracy"] - np.mean(shuffled_c_pred == labels_val)), "permutation_seed": SEED}, "test_used": False}
    class_metrics = {name: methods[name]["per_class"] for name in ("Random-ShortPrefix5", *branches, "Mixed5 GT-TrueLogP Oracle")}
    target_audit = {"train_contexts": len(train_rows), "val_contexts": len(val_rows), "train_records": len({str(row["record_id"]) for row in train_rows}), "val_records": len({str(row["record_id"]) for row in val_rows}), "train_mean_legal_candidates": float(train_mask[:, 1:].sum(axis=1).mean()), "val_mean_legal_candidates": float(val_mask[:, 1:].sum(axis=1).mean()), "train_stay_coverage": float(train_mask[:, 0].mean()), "val_stay_coverage": float(val_mask[:, 0].mean()), "train_candidate_coverage": float(train_mask[:, 1:].mean()), "val_candidate_coverage": float(val_mask[:, 1:].mean()), "train_gt_true_logp": _target_stats(train_target, labels_train, train_ids, train_mask), "val_gt_true_logp": _target_stats(val_target, labels_val, val_ids, val_mask), "val_mixed5_gt_true_logp_oracle": val_oracle_metric, "val_mixed5_gt_margin_oracle": methods["Mixed5 GT-Margin/AnyCorrect"], "stay_target_definition": "current[0:30]", "candidate_target_definition": "current[0:5] + candidate[5:30]", "action_set": "current/Stay + Stage-A legal candidate_pool", "test_used": False}
    gain_recovery = {name: float((methods[name]["accuracy"] - methods["Random-ShortPrefix5"]["accuracy"]) / (methods["Mixed5 GT-TrueLogP Oracle"]["accuracy"] - methods["Random-ShortPrefix5"]["accuracy"])) for name in branches}
    population = {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": target_audit["train_records"], "val_records": target_audit["val_records"]}
    result: dict[str, Any] = {"experiment_id": "REDUCED12_PREFIX5_TASK_UTILITY_SELECTOR_V1", "status": "COMPLETED", "population": population, "labels": list(LABELS), "branches": branches, "methods": methods, "ranking_metrics": ranking_metrics, "gain_recovery_vs_random_to_oracle": gain_recovery, "target_audit": target_audit, "high_occlusion": {"definition": "bottom tertile of frame-0 Stay SceneVisibility", "count": int(high_mask.sum()), "methods": high_methods}, "context_transitions": transitions, "shuffle_diagnostics": shuffle_diagnostics, "baseline_reference": {"RGBGlobal Visibility": {"accuracy": 0.494841}, "Frame0 Task+VisibilityAux": {"accuracy": 0.506151}}, "protocol": {"prefix_frames": [0, 1, 2, 3, 4], "mixed_sequence": "Stay=current[0:30]; candidate=current[0:5]+candidate[5:30]", "loss": "SmoothL1 + 0.5 listwise(target_dist||pred_dist), tau=0.5", "checkpoint_selection": "minimum Moving Val Mixed5 utility loss", "sampling": "313 records, 16 contexts per record per epoch", "terminal": "selected real mixed sequence through frozen ST-GCN + frozen shared head", "discrete_time_view_switch_approximation": True}, "flags": {"policy_test_used": False, "training_split": "Policy Train", "evaluation_split": "Moving Val", "new_rgb_generated": False, "new_skeleton_generated": False, "future_candidate_observation_used_for_predictor": False, "future_candidate_skeleton_used_for_predictor": False, "future_candidate_rgb_used": False, "future_candidate_dino_used": False, "gt_action_used_for_selector": False, "gt_action_used_for_train_supervision": True, "recognizer_output_used_for_selector": False, "frozen_stgcn_modified": False, "deployable": True}, "artifacts": {"stgcn_checkpoint": {"path": str(stgcn_path.resolve()), "sha256": _sha256(stgcn_path)}, "shared_head_checkpoint": {"path": str(head_path.resolve()), "sha256": _sha256(head_path)}, "frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta}, "runtime": {"device": str(device), "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD}}
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "result.json", result)
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    _write(output_root / "config.json", {"branches": branches, "prefix_shape": [3, 5, 17], "prefix_dim": PREFIX_DIM, "dino_input": "current frame-0 16x768 mean pooled to 768", "option_geometry_dim": OPTION_GEOMETRY_DIM, "tau": TAU, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "test_used": False})
    _write(output_root / "training_summary.json", {name: json.loads((output_root / f"{name}_training.json").read_text(encoding="utf-8")) for name in branches})
    _write(output_root / "per_branch_metrics.json", {name: {"downstream": methods[name], "ranking": ranking_metrics[name], "gain_recovery_vs_random_to_oracle": gain_recovery[name]} for name in branches})
    _write(output_root / "ranking_metrics.json", ranking_metrics)
    _write(output_root / "occlusion_stratified_metrics.json", result["high_occlusion"])
    _write(output_root / "context_transitions.json", transitions)
    _write(output_root / "shuffle_diagnostics.json", shuffle_diagnostics)
    _write(output_root / "target_audit.json", target_audit)
    _write(output_root / "per_class_metrics.json", class_metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
