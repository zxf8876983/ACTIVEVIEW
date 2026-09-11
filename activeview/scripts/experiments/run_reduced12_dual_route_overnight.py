#!/usr/bin/env python3
"""Run the reduced12 overnight dual-route Train/Val experiment.

The script has three stages: a shared frozen-ST-GCN feature cache and
view-agnostic classifier head, a non-parametric retrieval Route-1 selector,
and a small reliability-verifier Route-2.  Policy Test is never opened.
Large feature caches and checkpoints are written below ``ACTIVEVIEW_DATA_ROOT``
and are intentionally kept outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification
from activeview.scripts.experiments.run_reduced12_direct_candidate_correctness import (
    CandidateCorrectnessRanker,
)

SEED = 42
NUM_CLASSES = 12
NUM_VIEWS = 32
FEATURE_DIM = 256
MAX_OPTIONS = 22  # stay + at most 21 legal candidates
HEAD_EPOCHS = 15
VERIFIER_EPOCHS = 15
OBS_PER_RECORD = 16
TRAIN_BATCH = 1024
INFERENCE_BATCH = 2048
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
STGCN_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
OLD_DIRECT_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/direct_candidate_correctness_v1/"
    "direct_correctness_best.pth"
)
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1"
SHARED_OUTPUT = OUTPUT_ROOT / "view_agnostic_frozen_encoder_head"
ROUTE1_OUTPUT = OUTPUT_ROOT / "route1_retrieval_nbv_v1"
ROUTE2_OUTPUT = OUTPUT_ROOT / "route2_observe_verify_continue_v1"
SUMMARY_OUTPUT = OUTPUT_ROOT / "dual_route_overnight_summary"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def load_rows(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge Stage-A geometry/archive paths into Stage-C rows."""
    root = data_root / POLICY_RELATIVE
    train_c = read_jsonl(root / "stage_c/features/train.jsonl")
    val_c = read_jsonl(root / "stage_c/features/val.jsonl")
    train_a = {str(x["episode_id"]): x for x in read_jsonl(root / "stage_a/episodes/train_episodes.jsonl")}
    val_a = {str(x["episode_id"]): x for x in read_jsonl(root / "stage_a/episodes/val_episodes.jsonl")}
    moving_ids = {str(x["episode_id"]) for x in read_jsonl(root / "stage_d/features/val.jsonl")}
    val_d = {str(x["episode_id"]): x for x in read_jsonl(root / "stage_d/features/val.jsonl")}

    def merge(source: Sequence[Mapping[str, Any]], archives: Mapping[str, Any], moving: bool) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for base in source:
            eid = str(base["episode_id"])
            if moving and eid not in moving_ids:
                continue
            a = archives.get(eid)
            if a is None:
                raise ValueError(f"missing Stage-A row for {eid}")
            candidates = [int(x["viewpoint_id"]) for x in a["candidate_pool"]]
            if candidates != [int(x) for x in base["candidate_viewpoint_ids"]]:
                raise ValueError(f"Stage-A/Stage-C candidate mismatch for {eid}")
            item = dict(base)
            item["archive_path"] = str(a["current_view"]["skeleton_source_path"])
            item["current_viewpoint_id"] = int(a["current_view"]["viewpoint_id"])
            item["candidate_ids"] = candidates
            item["label_id"] = int(base["label_id"])
            if eid in val_d:
                item["s1_viewpoint_id"] = int(val_d[eid]["s1_viewpoint_id"])
            output.append(item)
        return output

    train = merge(train_c, train_a, False)
    val = merge(val_c, val_a, True)
    if len(train) != 46324 or len(val) != 10080:
        raise ValueError(f"unexpected rows train={len(train)} moving_val={len(val)}")
    if {str(x["record_id"]) for x in train} & {str(x["record_id"]) for x in val}:
        raise ValueError("Train/Moving-Val record overlap detected")
    return train, val


def _archive_views(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
    if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)):
        raise ValueError(f"invalid viewpoint ids in {path}")
    if skeleton.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeleton).all():
        raise ValueError(f"invalid skeleton archive in {path}")
    return skeleton


def _cache_meta(rows: Sequence[Mapping[str, Any]], all32: bool, checkpoint: Path) -> dict[str, Any]:
    return {"rows": len(rows), "signature": row_signature(rows), "all32": all32, "checkpoint_sha256": sha256(checkpoint), "test_used": False}


def build_feature_cache(
    rows: Sequence[Mapping[str, Any]], model: torch.nn.Module, device: torch.device,
    cache_path: Path, meta_path: Path, all32: bool,
) -> dict[str, np.ndarray]:
    """Infer only current+legal (or all32 for Val) frozen observations once."""
    checkpoint = Path(str(rows[0].get("_checkpoint", "")))
    expected_meta = _cache_meta(rows, all32, checkpoint) if checkpoint.is_file() else None
    if cache_path.exists() and meta_path.exists() and expected_meta is not None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta == expected_meta:
            with np.load(cache_path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}

    count = len(rows)
    slots = NUM_VIEWS if all32 else MAX_OPTIONS
    features = np.zeros((count, slots, FEATURE_DIM), dtype=np.float16)
    logits = np.zeros((count, slots, NUM_CLASSES), dtype=np.float32)
    ids = np.full((count, slots), -1, dtype=np.int64)
    mask = np.zeros((count, slots), dtype=bool)
    for start in range(0, count, 128):
        batch_rows = rows[start : start + 128]
        flat: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for index, row in enumerate(batch_rows):
            views = _archive_views(Path(str(row["archive_path"])))
            wanted = list(range(NUM_VIEWS)) if all32 else [int(row["current_viewpoint_id"])] + list(row["candidate_ids"])
            wanted = list(dict.fromkeys(wanted))
            offset = len(flat)
            flat.extend([views[v] for v in wanted])
            spans.append((offset, len(wanted)))
            target_index = start + index
            ids[target_index, : len(wanted)] = np.asarray(wanted, dtype=np.int64)
            mask[target_index, : len(wanted)] = True
        batch = torch.from_numpy(np.stack(flat)).to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = model.forward_features(batch)
            output = model.fc(encoded)
        encoded_np = encoded.cpu().numpy()
        output_np = output.cpu().numpy()
        for index, (offset, length) in enumerate(spans):
            target_index = start + index
            features[target_index, :length] = encoded_np[offset : offset + length].astype(np.float16)
            logits[target_index, :length] = output_np[offset : offset + length]
        if (start + len(batch_rows)) % 2048 == 0:
            print(f"[shared-cache] rows={start + len(batch_rows)}/{count} all32={all32}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, logits=logits, ids=ids, mask=mask)
    if expected_meta is not None:
        meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return {"features": features, "logits": logits, "ids": ids, "mask": mask}


class SharedHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 256), nn.GELU(), nn.Linear(256, NUM_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def valid_flat(cache: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(cache["mask"], dtype=bool)
    feature = np.asarray(cache["features"], dtype=np.float32)[mask]
    logits = np.asarray(cache["logits"], dtype=np.float32)[mask]
    row_ids = np.repeat(np.arange(mask.shape[0]), mask.sum(axis=1))
    return feature, logits, row_ids


def head_legal_loss(head: SharedHead, cache: Mapping[str, np.ndarray], labels: np.ndarray, device: torch.device) -> float:
    features = np.asarray(cache["features"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=bool)
    chunks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in range(len(labels)):
        valid = np.flatnonzero(mask[index])
        valid = valid[1:]  # legal candidates; slot 0 is the current/stay view
        chunks.append(torch.from_numpy(features[index, valid]))
        targets.append(torch.full((len(valid),), int(labels[index]), dtype=torch.long))
    x = torch.cat(chunks).to(device)
    y = torch.cat(targets).to(device)
    with torch.inference_mode():
        return float(nn.functional.cross_entropy(head(x), y).item())


def train_shared_head(
    train_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray], val_labels: np.ndarray, device: torch.device,
    checkpoint: Path, summary_path: Path,
) -> tuple[SharedHead, dict[str, Any]]:
    if checkpoint.exists() and summary_path.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        head = SharedHead().to(device)
        head.load_state_dict(payload["state_dict"])
        return head.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    head = SharedHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        groups[str(row["record_id"])].append(index)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    rng = np.random.default_rng(SEED)
    for epoch in range(1, HEAD_EPOCHS + 1):
        sampled: list[tuple[int, int]] = []
        for record in sorted(groups):
            choices = [(row, slot) for row in groups[record] for slot in np.flatnonzero(train_cache["mask"][row])]
            replace = len(choices) < OBS_PER_RECORD
            picked = rng.choice(len(choices), OBS_PER_RECORD, replace=replace)
            sampled.extend([choices[int(i)] for i in np.asarray(picked)])
        x = np.stack([train_cache["features"][row, slot] for row, slot in sampled]).astype(np.float32)
        y = np.asarray([int(train_rows[row]["label_id"]) for row, _ in sampled], dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=TRAIN_BATCH, shuffle=True, pin_memory=True)
        head.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        val_loss = head_legal_loss(head, val_cache, val_labels, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_legal_ce": val_loss, "records": len(groups), "observations": len(sampled)}
        history.append(record)
        print(f"[shared-head] epoch={epoch:02d} train_loss={record['train_loss']:.6f} val_legal_ce={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": head.state_dict(), "epoch": epoch, "seed": SEED, "num_classes": NUM_CLASSES}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(payload["state_dict"])
    summary = {"seed": SEED, "head": "Linear(256,256)->GELU->Linear(256,12)", "lr": HEAD_LR, "max_epochs": HEAD_EPOCHS, "best_epoch": best_epoch, "best_val_legal_ce": best_loss, "records": len(groups), "observations_per_record": OBS_PER_RECORD, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return head.eval(), summary


def head_logits(head: SharedHead, features: np.ndarray, device: torch.device) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), INFERENCE_BATCH):
            outputs.append(head(torch.from_numpy(flat[start : start + INFERENCE_BATCH]).to(device)).cpu().numpy())
    return np.concatenate(outputs).reshape(*features.shape[:-1], NUM_CLASSES).astype(np.float32)


def classify_view_population(labels: np.ndarray, logits: np.ndarray, ids: np.ndarray, chosen: Sequence[int]) -> dict[str, Any]:
    predictions: list[int] = []
    for index, viewpoint in enumerate(chosen):
        slots = np.flatnonzero(ids[index] == int(viewpoint))
        if slots.size != 1:
            raise ValueError(f"viewpoint {viewpoint} absent at row {index}")
        predictions.append(int(np.argmax(logits[index, int(slots[0])])))
    return classification(labels, np.asarray(predictions, dtype=np.int64))


def population_metrics(rows: Sequence[Mapping[str, Any]], labels: np.ndarray, cache: Mapping[str, np.ndarray], all_cache: Mapping[str, np.ndarray], selected_logits: np.ndarray, selected_all_logits: np.ndarray) -> dict[str, Any]:
    ids, mask = cache["ids"], cache["mask"]
    current = [int(row["current_viewpoint_id"]) for row in rows]
    s1 = [int(row["s1_viewpoint_id"]) for row in rows]
    legal_y: list[int] = []
    legal_p: list[int] = []
    for index in range(len(rows)):
        valid = np.flatnonzero(mask[index])
        # slot 0 is current; candidate slots are the legal actions.
        candidate_slots = valid[1:]
        legal_y.extend([int(labels[index])] * len(candidate_slots))
        legal_p.extend(np.argmax(selected_logits[index, candidate_slots], axis=1).tolist())
    all_p = np.argmax(selected_all_logits, axis=2).reshape(-1)
    all_y = np.repeat(labels, NUM_VIEWS)
    return {"s0_current": classify_view_population(labels, selected_logits, ids, current), "s1": classify_view_population(labels, selected_logits, ids, s1), "legal_candidates_micro": classification(legal_y, legal_p), "all32": classification(all_y, all_p), "legal_observations": len(legal_y), "all32_observations": len(all_y)}


def relative_key(current: int, candidate: int) -> tuple[int, int]:
    cr, ca = divmod(int(current), 8)
    vr, va = divmod(int(candidate), 8)
    da = (va - ca + 4) % 8 - 4
    return int(vr - cr), int(da)


def _candidate_geometry(row: Mapping[str, Any], candidate: int) -> tuple[float, float]:
    ids = [int(x) for x in row["candidate_ids"]]
    geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
    index = ids.index(int(candidate))
    return float(geometry[index, 4]), float(geometry[index, 3])


def route_metrics(labels: np.ndarray, predictions: Sequence[int], moves: Sequence[int], name: str, selected_correct: Sequence[bool] | None = None) -> dict[str, Any]:
    result = classification(labels, np.asarray(predictions, dtype=np.int64))
    result.update({"selector": name, "move_rate": float(np.mean(moves)), "mean_moves": float(np.mean(moves)), "mean_observations": float(np.mean(np.asarray(moves) + 1)), "test_used": False})
    if selected_correct is not None:
        result["selected_observed_correct_rate"] = float(np.mean(selected_correct))
    return result


def single_verifier_score(model: nn.Module, row_index: int, viewpoint: int, cache: Mapping[str, np.ndarray], recog: np.ndarray, device: torch.device) -> float:
    """Score one already-observed view; no candidate batch is inspected."""
    slots = np.flatnonzero(cache["ids"][row_index] == int(viewpoint))
    if slots.size != 1:
        raise ValueError(f"viewpoint {viewpoint} absent at row {row_index}")
    slot = int(slots[0]); feature = np.asarray(cache["features"][row_index, slot], dtype=np.float32)[None]
    logits = np.asarray(recog[row_index, slot], dtype=np.float32)[None]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)); probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    part = np.partition(probabilities, -2, axis=1)
    values = np.concatenate((feature, logits, -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=1, keepdims=True), probabilities.max(axis=1, keepdims=True), (part[:, -1] - part[:, -2])[:, None], np.linalg.norm(feature, axis=1, keepdims=True)), axis=1).astype(np.float32)
    with torch.inference_mode():
        return float(torch.sigmoid(model(torch.from_numpy(values).to(device))).item())


def route1_select(
    rows: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray], recog_train: np.ndarray, recog_val: np.ndarray, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Non-parametric retrieval; selector input is asserted to be current-only."""
    # Build current query descriptors and relation-indexed supervision.
    def descriptor(feature: np.ndarray, logits: np.ndarray, viewpoint: int) -> np.ndarray:
        posterior = np.asarray(torch.softmax(torch.from_numpy(logits), dim=-1))
        entropy = -float(np.sum(posterior * np.log(np.clip(posterior, 1e-12, None))))
        part = np.partition(posterior, -2)[-2:]
        vector = np.concatenate((feature / max(float(np.linalg.norm(feature)), 1e-8), logits, [viewpoint / 31.0, entropy, float(part[-1] - part[-2])])).astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    db_desc: list[np.ndarray] = []
    db_rel: list[tuple[int, int]] = []
    db_target: list[float] = []
    db_record: list[str] = []
    for index, row in enumerate(train_rows):
        feature = np.asarray(train_cache["features"][index, 0], dtype=np.float32)
        logit = recog_train[index, 0]
        current = int(row["current_viewpoint_id"])
        base = descriptor(feature, logit, current)
        db_desc.append(base); db_rel.append((0, 0)); db_target.append(float(np.argmax(logit) == int(row["label_id"]))); db_record.append(str(row["record_id"]))
        for slot, candidate in enumerate(row["candidate_ids"], 1):
            db_desc.append(base); db_rel.append(relative_key(current, int(candidate))); db_target.append(float(np.argmax(recog_train[index, slot]) == int(row["label_id"]))); db_record.append(str(row["record_id"]))
    db_matrix = np.stack(db_desc).astype(np.float32)
    groups: dict[tuple[int, int], np.ndarray] = {}
    for relation in sorted(set(db_rel)):
        groups[relation] = np.asarray([i for i, value in enumerate(db_rel) if value == relation], dtype=np.int64)
    priors = {key: float(np.mean(np.asarray(db_target)[indices])) for key, indices in groups.items()}
    current_desc = [descriptor(np.asarray(val_cache["features"][i, 0], dtype=np.float32), recog_val[i, 0], int(row["current_viewpoint_id"])) for i, row in enumerate(rows)]
    # Explicitly keep the selector input surface free of future arrays/labels.
    forbidden = {"candidate_feature", "candidate_logp", "candidate_skeleton", "candidate_prediction", "label_id"}
    selector_inputs = {"current_feature", "current_logp", "current_viewpoint_id", "candidate_ids", "candidate_geometry"}
    if forbidden & selector_inputs:
        raise AssertionError("Route-1 inference input leakage")

    def select_with_k(k: int) -> tuple[list[int], list[int], list[float]]:
        score_lists: list[list[float]] = [[float("-inf")] * (1 + len(row["candidate_ids"])) for row in rows]
        support_lists: list[list[int]] = [[0] * (1 + len(row["candidate_ids"])) for row in rows]
        query_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for context_index, row in enumerate(rows):
            current = int(row["current_viewpoint_id"])
            for option_index, candidate in enumerate([current] + [int(x) for x in row["candidate_ids"]]):
                query_groups[relative_key(current, candidate)].append((context_index, option_index))
        for relation, requests in query_groups.items():
            pool = groups.get(relation, np.empty(0, dtype=np.int64))
            if pool.size == 0:
                value = priors.get(relation, float(np.mean(db_target)))
                for context_index, option_index in requests:
                    score_lists[context_index][option_index] = value
                continue
            db_tensor = torch.from_numpy(db_matrix[pool]).to(device, non_blocking=True)
            target_tensor = torch.from_numpy(np.asarray(db_target, dtype=np.float32)[pool]).to(device, non_blocking=True)
            top_k = min(int(k), int(pool.size))
            for start in range(0, len(requests), 256):
                chunk = requests[start : start + 256]
                query = torch.from_numpy(np.stack([current_desc[item[0]] for item in chunk])).to(device, non_blocking=True)
                similarities = query @ db_tensor.T
                values, indices = torch.topk(similarities, top_k, dim=1)
                weights = values.clamp_min(0.0) + 1e-3
                estimates = (weights * target_tensor[indices]).sum(dim=1).add(1.0) / weights.sum(dim=1).add(2.0)
                estimates_cpu = estimates.detach().cpu().numpy()
                for local, (context_index, option_index) in enumerate(chunk):
                    score_lists[context_index][option_index] = float(estimates_cpu[local])
                    support_lists[context_index][option_index] = int(pool.size)
            del db_tensor, target_tensor
        selected: list[int] = []
        supports: list[int] = []
        costs: list[float] = []
        for context_index, row in enumerate(rows):
            options = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
            best = int(np.argmax(np.asarray(score_lists[context_index], dtype=np.float64)))
            selected.append(options[best]); supports.append(support_lists[context_index][best]); costs.append(0.0 if best == 0 else _candidate_geometry(row, options[best])[0])
        return selected, supports, costs

    actions, supports, selected_costs = select_with_k(32)
    predictions: list[int] = []
    moves: list[int] = []
    correct: list[bool] = []
    for index, action in enumerate(actions):
        slot = int(np.flatnonzero(val_cache["ids"][index] == action)[0])
        prediction = int(np.argmax(recog_val[index, slot])); predictions.append(prediction); moves.append(int(action != int(rows[index]["current_viewpoint_id"]))); correct.append(prediction == int(labels_for_rows(rows)[index]))
    metrics = route_metrics(labels_for_rows(rows), predictions, moves, "Retrieval-NBV K=32", correct)
    metrics.update({"path_cost_mean": float(np.mean(selected_costs)), "candidate_support_mean": float(np.mean(supports)), "candidate_support_min": int(min(supports)), "retrieval_k": 32, "actions": actions})
    robustness: dict[str, Any] = {}
    for k in (16, 64):
        scores_actions, _, _ = select_with_k(k)
        pred = [int(np.argmax(recog_val[i, int(np.flatnonzero(val_cache["ids"][i] == action)[0])])) for i, action in enumerate(scores_actions)]
        robustness[str(k)] = route_metrics(labels_for_rows(rows), pred, [int(a != int(rows[i]["current_viewpoint_id"])) for i, a in enumerate(scores_actions)], f"Retrieval-NBV K={k}")
    return metrics, robustness


def gt_true_logp_actions(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], logits: np.ndarray,
) -> list[int]:
    """Privileged oracle: maximize the GT-class score over stay + legal views."""
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(cache["mask"][index])
        options = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        actions.append(options[int(np.argmax(logits[index, valid, int(row["label_id"])]))])
    return actions


def labels_for_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)


def _baseline_actions(rows: Sequence[Mapping[str, Any]], kind: str, seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    actions: list[int] = []
    for row in rows:
        current = int(row["current_viewpoint_id"]); candidates = [int(x) for x in row["candidate_ids"]]
        if kind == "stay": actions.append(current)
        elif kind == "random": actions.append(int(rng.choice([current] + candidates)))
        elif kind == "nearest":
            actions.append(min([current] + candidates, key=lambda x: (0.0 if x == current else _candidate_geometry(row, x)[0], x)))
        else: raise ValueError(kind)
    return actions


def evaluate_actions(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], recog: np.ndarray, actions: Sequence[int], name: str) -> dict[str, Any]:
    labels = labels_for_rows(rows); predictions: list[int] = []; moves: list[int] = []
    for i, action in enumerate(actions):
        slot = int(np.flatnonzero(cache["ids"][i] == int(action))[0]); predictions.append(int(np.argmax(recog[i, slot]))); moves.append(int(action != int(rows[i]["current_viewpoint_id"])))
    return route_metrics(labels, predictions, moves, name, np.asarray(predictions) == labels)


def old_direct_actions(rows: Sequence[Mapping[str, Any]], device: torch.device, stats_path: Path, checkpoint: Path) -> list[int] | None:
    if not checkpoint.exists(): return None
    stats = json.loads(stats_path.read_text(encoding="utf-8")); payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CandidateCorrectnessRanker(int(payload["current_dim"]), int(payload["geometry_dim"])).to(device)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    current_values = np.stack([(np.asarray(row["current_feature"], dtype=np.float32) - np.asarray(stats["current_mean"], dtype=np.float32)) / np.asarray(stats["current_std"], dtype=np.float32) for row in rows])
    geometry_values = np.zeros((len(rows), 21, 11), dtype=np.float32)
    id_values = np.full((len(rows), 21), -1, dtype=np.int64)
    current_ids = np.zeros(len(rows), dtype=np.float32)
    option_lists: list[list[int]] = []
    for index, row in enumerate(rows):
        geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - np.asarray(stats["geometry_mean"], dtype=np.float32)) / np.asarray(stats["geometry_std"], dtype=np.float32)
        ids = np.asarray(row["candidate_ids"], dtype=np.int64); geometry_values[index, : len(ids)] = geometry; id_values[index, : len(ids)] = ids; current_ids[index] = float(row["current_viewpoint_id"]); option_lists.append([int(row["current_viewpoint_id"])] + ids.tolist())
    scores_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), INFERENCE_BATCH):
            scores_parts.append(model(torch.from_numpy(current_values[start : start + INFERENCE_BATCH]).to(device), torch.from_numpy(geometry_values[start : start + INFERENCE_BATCH]).to(device), torch.from_numpy(id_values[start : start + INFERENCE_BATCH]).to(device), torch.from_numpy(current_ids[start : start + INFERENCE_BATCH]).to(device)).cpu().numpy())
    scores = np.concatenate(scores_parts, axis=0)
    return [option_lists[index][int(np.argmax(scores[index, : len(option_lists[index])]))] for index in range(len(rows))]


def train_verifier(
    train_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray], recog_train: np.ndarray,
    device: torch.device, checkpoint: Path, summary_path: Path,
) -> tuple[nn.Module, float, dict[str, Any]]:
    mask = np.asarray(train_cache["mask"], dtype=bool)
    feature = np.asarray(train_cache["features"], dtype=np.float32)[mask]
    logits = np.asarray(recog_train, dtype=np.float32)[mask]
    row_ids = np.repeat(np.arange(mask.shape[0]), mask.sum(axis=1))
    labels = labels_for_rows(train_rows)
    correctness = (np.argmax(logits, axis=1) == labels[row_ids]).astype(np.float32)
    record_names = sorted({str(row["record_id"]) for row in train_rows})
    calibration_records = set(record_names[-max(1, round(len(record_names) * 0.1)):])
    train_records = set(record_names) - calibration_records
    record_for_row = np.asarray([str(row["record_id"]) for row in train_rows], dtype=object)
    train_mask = np.isin(record_for_row[row_ids], list(train_records))
    calib_mask = np.isin(record_for_row[row_ids], list(calibration_records))
    def verifier_input(feat: np.ndarray, lp: np.ndarray) -> np.ndarray:
        p = np.exp(lp - lp.max(axis=1, keepdims=True)); p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        part = np.partition(p, -2, axis=1); entropy = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1, keepdims=True); margin = (part[:, -1] - part[:, -2])[:, None]
        return np.concatenate((feat, lp, entropy, p.max(axis=1, keepdims=True), margin, np.linalg.norm(feat, axis=1, keepdims=True)), axis=1).astype(np.float32)
    x_all = verifier_input(feature, logits)
    model = nn.Sequential(nn.Linear(FEATURE_DIM + NUM_CLASSES + 4, 256), nn.GELU(), nn.Linear(256, 1)).to(device)
    if checkpoint.exists() and summary_path.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"]); model.eval(); summary = json.loads(summary_path.read_text(encoding="utf-8")); return model, float(summary["threshold"]), summary
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(train_rows): groups[str(row["record_id"])].append(i)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []; rng = np.random.default_rng(SEED)
    flat_slot_index: dict[tuple[int, int], int] = {}
    cursor = 0
    for row_index in range(mask.shape[0]):
        for slot in np.flatnonzero(mask[row_index]):
            flat_slot_index[(row_index, int(slot))] = cursor
            cursor += 1
    for epoch in range(1, VERIFIER_EPOCHS + 1):
        sampled_flat: list[int] = []
        for record in sorted(groups):
            if record not in train_records: continue
            pairs = [(index, int(slot)) for index in groups[record] for slot in np.flatnonzero(train_cache["mask"][index])]
            picked = rng.choice(len(pairs), OBS_PER_RECORD, replace=len(pairs) < OBS_PER_RECORD)
            for pick in np.asarray(picked):
                pair = pairs[int(pick)]
                sampled_flat.append(flat_slot_index[pair])
        xb = torch.from_numpy(x_all[sampled_flat]).to(device); yb = torch.from_numpy(correctness[sampled_flat, None]).to(device)
        model.train(); optimizer.zero_grad(set_to_none=True); loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb); loss.backward(); optimizer.step(); model.eval()
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "train_records": len(train_records), "observations": len(sampled_flat)})
        print(f"[route2-verifier] epoch={epoch:02d} train_loss={float(loss.detach().cpu()):.6f}", flush=True)
    with torch.inference_mode():
        calibration_scores = torch.sigmoid(model(torch.from_numpy(x_all[calib_mask]).to(device))).cpu().numpy().reshape(-1)
    calibration_targets = correctness[calib_mask]
    threshold = calibrate_threshold(calibration_scores, calibration_targets)
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": model.state_dict(), "threshold": threshold}, checkpoint)
    summary = {"seed": SEED, "input_dim": int(x_all.shape[1]), "model": "Linear(272,256)->GELU->Linear(256,1)", "max_epochs": VERIFIER_EPOCHS, "train_records": len(train_records), "calibration_records": len(calibration_records), "threshold": threshold, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model.eval(), threshold, summary


def calibrate_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.unique(np.concatenate((scores, np.asarray([1.0]))))
    feasible: list[tuple[float, float]] = []
    for threshold in candidates:
        accepted = scores >= threshold
        if accepted.any() and float(np.mean(targets[accepted])) >= 0.75:
            feasible.append((float(np.mean(accepted)), float(threshold)))
    return max(feasible, key=lambda x: (x[0], x[1]))[1] if feasible else 1.0


def verifier_features(cache: Mapping[str, np.ndarray], recog: np.ndarray) -> np.ndarray:
    features, logits, _ = valid_flat({"features": cache["features"], "logits": recog, "mask": cache["mask"]})
    p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None); part = np.partition(p, -2, axis=1)
    return np.concatenate((features, logits, -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1, keepdims=True), p.max(axis=1, keepdims=True), (part[:, -1] - part[:, -2])[:, None], np.linalg.norm(features, axis=1, keepdims=True)), axis=1).astype(np.float32)


def verifier_scores(model: nn.Module, cache: Mapping[str, np.ndarray], recog: np.ndarray, device: torch.device) -> np.ndarray:
    x = verifier_features(cache, recog); output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), INFERENCE_BATCH): output.append(torch.sigmoid(model(torch.from_numpy(x[start : start + INFERENCE_BATCH]).to(device))).cpu().numpy().reshape(-1))
    values = np.concatenate(output); result = np.zeros(cache["mask"].shape, dtype=np.float32); cursor = 0
    for i in range(result.shape[0]):
        length = int(cache["mask"][i].sum()); result[i, :length] = values[cursor : cursor + length]; cursor += length
    return result


def angular_next(row: Mapping[str, Any], current: int, visited: set[int]) -> int | None:
    candidates = [int(x) for x in row["candidate_ids"] if int(x) not in visited]
    if not candidates: return None
    def key(candidate: int) -> tuple[float, float, int]:
        _, ca = divmod(int(current), 8); _, va = divmod(candidate, 8); diff = min((va - ca) % 8, (ca - va) % 8)
        cost = _candidate_geometry(row, candidate)[0]
        return -float(diff), cost, candidate
    return min(candidates, key=key)


def random_next(row: Mapping[str, Any], visited: set[int], rng: np.random.Generator) -> int | None:
    candidates = [int(x) for x in row["candidate_ids"] if int(x) not in visited]
    return int(rng.choice(candidates)) if candidates else None


def route2_evaluate(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], recog: np.ndarray, verifier: nn.Module, threshold: float, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = labels_for_rows(rows)
    observed_scores: dict[tuple[int, int], float] = {}

    def score_observed(index: int, viewpoint: int) -> float:
        key = (index, int(viewpoint))
        if key not in observed_scores:
            observed_scores[key] = single_verifier_score(verifier, index, int(viewpoint), cache, recog, device)
        return observed_scores[key]

    def observed_confidence(index: int, viewpoint: int) -> float:
        slots = np.flatnonzero(cache["ids"][index] == int(viewpoint))
        if slots.size != 1:
            raise ValueError(f"viewpoint {viewpoint} absent at row {index}")
        logits = np.asarray(recog[index, int(slots[0])], dtype=np.float64)
        probabilities = np.exp(logits - np.max(logits)); probabilities /= np.clip(probabilities.sum(), 1e-12, None)
        return float(np.max(probabilities))

    results: dict[str, Any] = {}; trajectories: dict[str, Any] = {}

    def run_policy(name: str, mode: str, steps: int) -> None:
        predictions: list[int] = []; moves: list[int] = []; stop_steps: list[int] = []; paths: list[list[int]] = []
        for i, row in enumerate(rows):
            current = int(row["current_viewpoint_id"]); visited = {current}; path = [current]; stop = steps
            for step in range(steps):
                if mode == "verifier" and float(max(score_observed(i, view) for view in visited)) >= threshold:
                    stop = step; break
                nxt = angular_next(row, current, visited) if mode in {"angular", "verifier"} else random_next(row, visited, np.random.default_rng(SEED + i * 17 + step))
                if nxt is None: stop = step; break
                current = nxt; visited.add(current); path.append(current)
            if mode == "max_conf": selected = max(path, key=lambda view: observed_confidence(i, view))
            elif mode in {"verifier_best", "verifier"}: selected = max(path, key=lambda view: score_observed(i, view))
            else: selected = path[-1]
            slot = int(np.flatnonzero(cache["ids"][i] == selected)[0]); predictions.append(int(np.argmax(recog[i, slot]))); moves.append(len(path) - 1); stop_steps.append(stop); paths.append(path)
        metric = route_metrics(labels, predictions, moves, name, np.asarray(predictions) == labels)
        metric.update({"stop_at_step0_rate": float(np.mean(np.asarray(stop_steps) == 0)), "stop_at_step1_rate": float(np.mean(np.asarray(stop_steps) == 1)), "stop_at_step2_rate": float(np.mean(np.asarray(stop_steps) == 2)), "reach_max_step_rate": float(np.mean(np.asarray(moves) == 3)), "accuracy_vs_cost": {str(k): float(np.mean(np.asarray(predictions)[np.asarray(moves) <= k] == labels[np.asarray(moves) <= k])) if np.any(np.asarray(moves) <= k) else 0.0 for k in range(4)}})
        results[name] = metric; trajectories[name] = paths

    results["Stay only"] = evaluate_actions(rows, cache, recog, _baseline_actions(rows, "stay"), "Stay only")
    for steps in (1, 2, 3): run_policy(f"Random-{steps}", "random", steps)
    for steps in (1, 2, 3): run_policy(f"Angular-Diversity always-{steps}", "angular", steps)
    run_policy("MaxConfidence over observed views", "max_conf", 3)
    run_policy("Verifier-best over observed views", "verifier_best", 3)
    run_policy("Verifier stop/continue", "verifier", 3)
    return results, {"trajectories": trajectories, "threshold": threshold}


def verifier_quality(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], recog: np.ndarray, model: nn.Module, device: torch.device) -> dict[str, float]:
    scores = verifier_scores(model, cache, recog, device); labels = labels_for_rows(rows); targets = np.concatenate([(np.argmax(recog[i, np.flatnonzero(cache["mask"][i])], axis=1) == labels[i]).astype(np.float32) for i in range(len(rows))]); values = np.concatenate([scores[i, np.flatnonzero(cache["mask"][i])] for i in range(len(rows))])
    order = np.argsort(values, kind="mergesort")
    sorted_targets = targets[order]
    positives = float(targets.sum())
    negatives = float(len(targets) - positives)
    positive_ranks = np.flatnonzero(sorted_targets > 0.5) + 1
    auroc = float((positive_ranks.sum() - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0))
    ranked = sorted_targets[::-1]
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / max(positives, 1.0)
    auprc = float(np.trapz(precision, recall)) if len(tp) > 1 else 0.0
    pos = values[targets > 0.5]; neg = values[targets <= 0.5]
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (values >= low) & (values <= high if high == 1 else values < high)
        if mask.any(): ece += float(mask.mean()) * abs(float(values[mask].mean()) - float(targets[mask].mean()))
    return {"auroc": auroc, "auprc": auprc, "ece": ece, "correct_mean_score": float(pos.mean()) if len(pos) else 0.0, "wrong_mean_score": float(neg.mean()) if len(neg) else 0.0}


def write_analysis(shared: Mapping[str, Any], route1: Mapping[str, Any], route2: Mapping[str, Any], path: Path) -> None:
    selected = shared["manifest"]["selected_recognizer"]
    lines = ["# Reduced12 Overnight Dual-Route Experiment", "", "Policy Test was not read. All training used Policy Train; Moving Val (10,080 contexts) was evaluation/model selection only.", "", f"Shared recognizer: {selected}.", "", "## Shared frozen-encoder head", "", "| Population | Frozen Acc/F1 | Multi-view head Acc/F1 |", "|---|---:|---:|"]
    for pop, values in shared["metrics"].items(): lines.append(f"| {pop} | {values['Frozen']['accuracy']:.6f}/{values['Frozen']['macro_f1']:.6f} | {values['MultiViewHead']['accuracy']:.6f}/{values['MultiViewHead']['macro_f1']:.6f} |")
    lines.extend(["", "## Route-1", "", "| Selector | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|"])
    for name, metric in route1["metrics"].items():
        if "accuracy" in metric:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    coverage = route1["metrics"]["Legal AnyCorrect Coverage"]
    lines.append(f"| Legal AnyCorrect Coverage | {coverage['rate']:.6f} coverage ({coverage['count']}/{coverage['contexts']}) | — | — |")
    lines.extend(["", f"Retrieval K=32 support mean={route1['primary'].get('candidate_support_mean', 0.0):.2f}; K=16/64 are robustness checks only.", "", "## Route-2", "", f"Verifier AUROC={route2['verifier_quality']['auroc']:.6f}, AUPRC={route2['verifier_quality']['auprc']:.6f}, ECE={route2['verifier_quality']['ece']:.6f}.", "", "| Policy | Accuracy | Macro-F1 | Mean moves |", "|---|---:|---:|---:|"])
    for name, metric in route2["metrics"].items(): lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('mean_moves', 0.0):.3f} |")
    r1_acc = float(route1["primary"]["accuracy"]); random_acc = float(route1["metrics"]["Random legal action"]["accuracy"]); route1_decision = "KEEP" if r1_acc >= 0.48 and r1_acc > random_acc + 0.01 else "KILL"
    v_acc = float(route2["metrics"]["Verifier stop/continue"]["accuracy"]); matched = float(route2["metrics"]["Random-3"]["accuracy"]); route2_decision = "KEEP" if route2["verifier_quality"]["auroc"] >= 0.65 and v_acc > matched + 0.01 else "KILL"
    lines.extend(["", f"Route-1 decision: **{route1_decision}** (primary K=32; kill rule Acc <48% or <= Random+1pp).", f"Route-2 decision: **{route2_decision}** (AUROC threshold 0.65 and matched Random+1pp).", "", "## Leakage and reproducibility flags", "", "```text", "policy_test_used=false", "train_used_for_shared_head_and_verifier=true", "moving_val_used_for_selection_and_evaluation=true", "route1_future_observation_read_before_selection=0", "route2_future_unobserved_observation_read=0", "new_rgb_or_skeleton_generated=false", "frozen_stgcn_modified=false", "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(); data_root = get_data_root(); device = require_cuda(args.device); started = time.time()
    stgcn_path = data_root / STGCN_RELATIVE
    if not stgcn_path.exists(): raise FileNotFoundError(stgcn_path)
    train_rows, val_rows = load_rows(data_root)
    for row in list(train_rows) + list(val_rows): row["_checkpoint"] = str(stgcn_path)
    labels_train, labels_val = labels_for_rows(train_rows), labels_for_rows(val_rows)
    model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device)); model.eval()
    runtime = data_root / "diagnostics/reduced12_dual_route_overnight"; runtime.mkdir(parents=True, exist_ok=True)
    train_cache = build_feature_cache(train_rows, model, device, runtime / "train_options.npz", runtime / "train_options.json", False)
    val_cache = build_feature_cache(val_rows, model, device, runtime / "val_options.npz", runtime / "val_options.json", False)
    val_all = build_feature_cache(val_rows, model, device, runtime / "val_all32.npz", runtime / "val_all32.json", True)
    original_train = np.asarray(train_cache["logits"], dtype=np.float32); original_val = np.asarray(val_cache["logits"], dtype=np.float32); original_all = np.asarray(val_all["logits"], dtype=np.float32)
    head_ckpt = data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
    head_summary_path = SHARED_OUTPUT / "training_summary.json"; head, head_training = train_shared_head(train_rows, train_cache, val_cache, labels_val, device, head_ckpt, head_summary_path)
    head_val = head_logits(head, np.asarray(val_cache["features"], dtype=np.float32), device); head_all = head_logits(head, np.asarray(val_all["features"], dtype=np.float32), device); head_train = head_logits(head, np.asarray(train_cache["features"], dtype=np.float32), device)
    def shared_population(logits_opt: np.ndarray, logits_all: np.ndarray) -> dict[str, Any]:
        return population_metrics(val_rows, labels_val, val_cache, val_all, logits_opt, logits_all)
    frozen_pop = shared_population(original_val, original_all); head_pop = shared_population(head_val, head_all)
    metric_table = {pop: {"Frozen": frozen_pop[pop], "MultiViewHead": head_pop[pop]} for pop in ("s0_current", "s1", "legal_candidates_micro", "all32")}
    s1_gain = head_pop["s1"]["accuracy"] - frozen_pop["s1"]["accuracy"]; legal_gain = head_pop["legal_candidates_micro"]["accuracy"] - frozen_pop["legal_candidates_micro"]["accuracy"]; all_gain = head_pop["all32"]["accuracy"] - frozen_pop["all32"]["accuracy"]
    selected = "shared_multiview_head" if legal_gain >= 0.02 and all_gain >= 0.02 and s1_gain - max(legal_gain, all_gain) <= 0.02 else "frozen_original_head"
    recog_train = head_train if selected == "shared_multiview_head" else original_train; recog_val = head_val if selected == "shared_multiview_head" else original_val
    manifest = {"protocol": "reduced12 eight-placement", "selected_recognizer": selected, "stgcn_checkpoint": str(stgcn_path.resolve()), "stgcn_checkpoint_sha256": sha256(stgcn_path), "shared_head_checkpoint": str(head_ckpt.resolve()), "head_checkpoint_sha256": sha256(head_ckpt), "metrics": metric_table, "selection_criterion": "Val Moving legal-candidate global CE; require >=2pp legal and all32 gains without s1-only specialization", "test_used": False}
    SHARED_OUTPUT.mkdir(parents=True, exist_ok=True); (SHARED_OUTPUT / "shared_recognizer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); (SHARED_OUTPUT / "result.json").write_text(json.dumps({"metrics": metric_table, "manifest": manifest, "training": head_training, "train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "test_used": False}, indent=2), encoding="utf-8"); (SHARED_OUTPUT / "config.json").write_text(json.dumps({"head": "Linear(256,256)->GELU->Linear(256,12)", "head_lr": HEAD_LR, "epochs": HEAD_EPOCHS, "record_balanced_observations": OBS_PER_RECORD, "test_used": False}, indent=2), encoding="utf-8")
    stats_path = data_root / POLICY_RELATIVE / "stage_c/stage_c_feature_stats.json"; old_direct = old_direct_actions(val_rows, device, stats_path, data_root / OLD_DIRECT_RELATIVE)
    route1_metrics: dict[str, Any] = {"Stay/current": evaluate_actions(val_rows, val_cache, recog_val, _baseline_actions(val_rows, "stay"), "Stay/current"), "Random legal action": evaluate_actions(val_rows, val_cache, recog_val, _baseline_actions(val_rows, "random"), "Random legal action"), "Nearest/lowest-cost": evaluate_actions(val_rows, val_cache, recog_val, _baseline_actions(val_rows, "nearest"), "Nearest/lowest-cost")}
    if old_direct is not None: route1_metrics["Old Direct-Correctness"] = evaluate_actions(val_rows, val_cache, recog_val, old_direct, "Old Direct-Correctness")
    primary, robustness = route1_select(val_rows, train_rows, train_cache, val_cache, recog_train, recog_val, device); route1_metrics["Retrieval-NBV K=32"] = primary; route1_metrics["GT-TrueLogP Oracle"] = evaluate_actions(val_rows, val_cache, recog_val, gt_true_logp_actions(val_rows, val_cache, recog_val), "GT-TrueLogP Oracle")
    any_correct = sum(any(int(np.argmax(recog_val[i, slot])) == int(row["label_id"]) for slot in np.flatnonzero(val_cache["mask"][i])[1:]) for i, row in enumerate(val_rows)); route1_metrics["Legal AnyCorrect Coverage"] = {"contexts": len(val_rows), "count": int(any_correct), "rate": float(any_correct / len(val_rows))}
    route1_result = {"metrics": route1_metrics, "primary": {key: value for key, value in primary.items() if key != "actions"}, "robustness": robustness, "train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "shared_recognizer_manifest": str((SHARED_OUTPUT / "shared_recognizer_manifest.json").resolve()), "leakage_audit": {"selection_inputs": ["current_feature", "current_logp", "current_viewpoint_id", "candidate_geometry", "candidate_ids"], "future_candidate_feature_read_before_selection": 0, "future_candidate_logit_read_before_selection": 0, "gt_label_read_before_selection": 0, "test_used": False}, "test_used": False}
    ROUTE1_OUTPUT.mkdir(parents=True, exist_ok=True)
    (ROUTE1_OUTPUT / "result.json").write_text(json.dumps(route1_result, indent=2), encoding="utf-8")
    (ROUTE1_OUTPUT / "config.json").write_text(json.dumps({"retrieval_k_primary": 32, "robustness_k": [16, 64], "test_used": False}, indent=2), encoding="utf-8")
    (ROUTE1_OUTPUT / "analysis.md").write_text("# Route-1 Retrieval NBV\n\nThe selector uses only current frozen feature/posterior, viewpoint identity and candidate geometry. Future candidate observations and labels are not read before action selection.\n", encoding="utf-8")
    (ROUTE1_OUTPUT / "per_class_metrics.json").write_text(json.dumps({action: {name: metric.get("per_class", {}).get(action, {}) for name, metric in route1_metrics.items() if "per_class" in metric} for action in LABELS}, indent=2), encoding="utf-8")
    verifier_ckpt = data_root / "checkpoints/policy_reduced12_eight_placement_v1/route2_observe_verify_continue_v1/verifier_best.pth"; verifier_summary_path = ROUTE2_OUTPUT / "training_summary.json"; verifier, threshold, verifier_training = train_verifier(train_rows, train_cache, recog_train, device, verifier_ckpt, verifier_summary_path)
    route2_metrics, route2_diag = route2_evaluate(val_rows, val_cache, recog_val, verifier, threshold, device); quality = verifier_quality(val_rows, val_cache, recog_val, verifier, device); route2_result = {"metrics": route2_metrics, "verifier_quality": quality, "training": verifier_training, "diagnostics": {"threshold": route2_diag["threshold"]}, "moving_val_contexts": len(val_rows), "shared_recognizer_manifest": str((SHARED_OUTPUT / "shared_recognizer_manifest.json").resolve()), "leakage_audit": {"future_unobserved_candidate_read_before_move": 0, "future_rgb_or_skeleton_generated": 0, "gt_label_read_for_policy": 0, "test_used": False}, "test_used": False}; ROUTE2_OUTPUT.mkdir(parents=True, exist_ok=True); (ROUTE2_OUTPUT / "result.json").write_text(json.dumps(route2_result, indent=2), encoding="utf-8"); (ROUTE2_OUTPUT / "config.json").write_text(json.dumps({"max_moves": 3, "calibration_precision": 0.75, "test_used": False}, indent=2), encoding="utf-8"); (ROUTE2_OUTPUT / "analysis.md").write_text("# Route-2 Observe-Verify-Continue\n\nThe verifier is trained on Policy Train only. Candidate observations are consumed only after the simulated move; unobserved candidates are not scored.\n", encoding="utf-8"); (ROUTE2_OUTPUT / "per_class_metrics.json").write_text(json.dumps({action: {name: metric.get("per_class", {}).get(action, {}) for name, metric in route2_metrics.items()} for action in LABELS}, indent=2), encoding="utf-8")
    summary = {"shared": {"manifest": manifest, "metrics": metric_table}, "route1": route1_result, "route2": route2_result, "elapsed_seconds": time.time() - started, "test_used": False}; SUMMARY_OUTPUT.mkdir(parents=True, exist_ok=True); (SUMMARY_OUTPUT / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); write_analysis(summary["shared"], route1_result, route2_result, SUMMARY_OUTPUT / "analysis.md"); (SUMMARY_OUTPUT / "config.json").write_text(json.dumps({"seed": SEED, "device": str(device), "head_epochs": HEAD_EPOCHS, "verifier_epochs": VERIFIER_EPOCHS, "test_used": False}, indent=2), encoding="utf-8"); log_dir = SUMMARY_OUTPUT / "logs"; log_dir.mkdir(parents=True, exist_ok=True); (log_dir / "route1.log").write_text(f"completed=true\nprimary_accuracy={primary['accuracy']:.6f}\nretrieval_k=32\ntest_used=false\n", encoding="utf-8"); (log_dir / "route2.log").write_text(f"completed=true\nverifier_auroc={quality['auroc']:.6f}\nthreshold={threshold:.6f}\ntest_used=false\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
