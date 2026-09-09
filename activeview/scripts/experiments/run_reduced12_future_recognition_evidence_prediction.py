#!/usr/bin/env python3
"""Train/Val-only predictors for future frozen-ST-GCN recognition evidence.

The predictors use only information available at H1: the current s0
Stage-C feature, the visited s0 DINO mean, and candidate geometry.  Existing
candidate log-probability targets are reused.  Candidate penultimate features
are inferred once from archived skeletons with the frozen reduced12 ST-GCN;
they are supervision targets, never predictor inputs.
"""

from __future__ import annotations

import argparse
import hashlib
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
from activeview.data.generation.utility_labels import file_sha256
from activeview.recognition.stgcn.model import STGCN, load_checkpoint


SEED = 42
NUM_CLASSES = 12
STGCN_FEATURE_DIM = 256
CURRENT_FEATURE_DIM = 271
DINO_DIM = 768
GEOMETRY_DIM = 11
HIDDEN_DIM = 256
EPOCHS = 20
BATCH_SIZE = 2048
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/future_recognition_evidence_prediction"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA")
    return device


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_dino(data_root: Path) -> tuple[dict[tuple[str, str, str, int], int], np.ndarray, dict[str, Any]]:
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    if embeddings.ndim != 3 or embeddings.shape[1:] != (16, DINO_DIM):
        raise ValueError(f"unexpected DINO shape: {embeddings.shape}")
    lookup: dict[tuple[str, str, str, int], int] = {}
    with (cache / "manifest.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
            if key in lookup:
                raise ValueError(f"duplicate DINO key: {key}")
            lookup[key] = index
    if len(lookup) != len(embeddings):
        raise ValueError("DINO manifest and embedding count differ")
    summary = json.loads((cache / "summary.json").read_text(encoding="utf-8"))
    if bool(summary.get("future_candidate_rgb_used", True)):
        raise ValueError("DINO cache provenance indicates future candidate RGB use")
    return lookup, embeddings, summary


def _covered_rows(
    rows: Sequence[Mapping[str, Any]],
    stage_d_ids: set[str],
    dino_lookup: Mapping[tuple[str, str, str, int], int],
) -> tuple[list[dict[str, Any]], list[int]]:
    selected: list[dict[str, Any]] = []
    indices: list[int] = []
    for index, row in enumerate(rows):
        if str(row["episode_id"]) not in stage_d_ids:
            continue
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        if key not in dino_lookup:
            raise ValueError(f"DINO cache missing visited s0 observation: {key}")
        selected.append(dict(row))
        indices.append(index)
    if not selected:
        raise RuntimeError("no DINO-covered rows selected")
    return selected, indices


def _subset_cache(cache: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {key: value[idx] if value.ndim and value.shape[0] == len(cache["labels"]) else value for key, value in cache.items()}


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    return archive_root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"


def _candidate_alignment(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> None:
    for index, row in enumerate(rows):
        valid = np.flatnonzero(cache["candidate_mask"][index])
        cache_ids = [int(value) for value in cache["candidate_ids"][index, valid]]
        row_ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        if cache_ids != row_ids:
            raise ValueError(f"candidate identity mismatch at {row['episode_id']}: cache={cache_ids} row={row_ids}")
        geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        if geometry.shape != (len(row_ids), GEOMETRY_DIM) or not np.isfinite(geometry).all():
            raise ValueError(f"invalid candidate geometry at {row['episode_id']}")


def _build_target_features(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    archive_root: Path,
    checkpoint: Path,
    output_path: Path,
    metadata_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    signature = _signature(rows)
    if output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("row_signature") == signature and metadata.get("checkpoint_sha256") == file_sha256(checkpoint):
            return np.asarray(np.load(output_path, allow_pickle=False)["candidate_feature"])

    stgcn, _ = load_checkpoint(checkpoint, NUM_CLASSES, str(device))
    max_candidates = int(cache["candidate_feature_placeholder"].shape[1]) if "candidate_feature_placeholder" in cache else int(cache["candidate_logp"].shape[1])
    target = np.zeros((len(rows), max_candidates, STGCN_FEATURE_DIM), dtype=np.float16)
    requests: dict[Path, dict[int, list[tuple[int, int]]]] = {}
    for row_index in range(len(rows)):
        path = _archive_path(archive_root, rows[row_index])
        if not path.exists():
            raise FileNotFoundError(f"missing archived skeleton: {path}")
        values = requests.setdefault(path, {})
        for candidate_index in np.flatnonzero(cache["candidate_mask"][row_index]):
            viewpoint_id = int(cache["candidate_ids"][row_index, candidate_index])
            values.setdefault(viewpoint_id, []).append((row_index, int(candidate_index)))

    started = time.time()
    skeletons: list[np.ndarray] = []
    ref_groups: list[list[tuple[int, int]]] = []

    def flush() -> None:
        if not skeletons:
            return
        batch = torch.from_numpy(np.stack(skeletons).astype(np.float32)).to(device).unsqueeze(-1)
        with torch.inference_mode():
            features = stgcn.forward_features(batch).cpu().numpy().astype(np.float16)
        for position, locations in enumerate(ref_groups):
            for row_index, candidate_index in locations:
                target[row_index, candidate_index] = features[position]
        skeletons.clear()
        ref_groups.clear()

    for archive_index, (path, requested) in enumerate(requests.items(), start=1):
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        if ids.shape != (32,) or skeleton.shape != (32, 3, 30, 17) or not np.isfinite(skeleton).all():
            raise ValueError(f"invalid archive schema: {path}")
        id_to_index = {int(value): index for index, value in enumerate(ids)}
        for viewpoint_id, locations in requested.items():
            if viewpoint_id not in id_to_index:
                raise ValueError(f"viewpoint {viewpoint_id} missing in {path}")
            skeletons.append(skeleton[id_to_index[viewpoint_id]])
            # The same archive/viewpoint can be shared by many contexts; one
            # frozen-ST-GCN inference is fanned out to all references.
            ref_groups.append(locations)
            if len(skeletons) >= batch_size:
                flush()
        flush()
        if archive_index % 250 == 0:
            print(f"target-feature archives={archive_index}/{len(requests)} elapsed={time.time() - started:.1f}s", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, candidate_feature=target)
    metadata_path.write_text(json.dumps({
        "rows": len(rows), "max_candidates": max_candidates, "row_signature": signature,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "feature_shape": list(target.shape), "dtype": "float16", "test_used": False,
        "future_skeleton_used_for_target_only": True,
    }, indent=2) + "\n", encoding="utf-8")
    return target


class EvidencePredictor(nn.Module):
    """Small two-layer GELU MLP with branch-specific evidence heads."""

    def __init__(self, input_dim: int, branch: str) -> None:
        super().__init__()
        self.branch = branch
        self.trunk = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU())
        self.logp_head = nn.Linear(HIDDEN_DIM, NUM_CLASSES) if branch in {"logp", "feature_logp"} else None
        self.feature_head = nn.Linear(HIDDEN_DIM, STGCN_FEATURE_DIM) if branch in {"feature", "feature_logp"} else None

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        latent = self.trunk(inputs)
        return (
            self.logp_head(latent) if self.logp_head is not None else None,
            self.feature_head(latent) if self.feature_head is not None else None,
        )


class _EvidenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, logp: np.ndarray, feature: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.logp = torch.from_numpy(logp.astype(np.float32))
        self.feature = torch.from_numpy(feature.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.logp[index], self.feature[index]


def _train_branch(
    branch: str,
    train_set: _EvidenceDataset,
    val_set: _EvidenceDataset,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    checkpoint: Path,
    summary_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _seed()
    model = EvidencePredictor(train_set.inputs.shape[1], branch).to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, target_logp, target_feature in train_loader:
            inputs = inputs.to(device)
            target_logp = target_logp.to(device)
            target_feature = target_feature.to(device)
            predicted_logp, predicted_feature = model(inputs)
            losses: list[torch.Tensor] = []
            if predicted_logp is not None:
                losses.append(nn.functional.smooth_l1_loss(predicted_logp, target_logp))
            if predicted_feature is not None:
                losses.append(nn.functional.smooth_l1_loss(predicted_feature, target_feature))
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, target_logp, target_feature in val_loader:
                predicted_logp, predicted_feature = model(inputs.to(device))
                losses = []
                if predicted_logp is not None:
                    losses.append(nn.functional.smooth_l1_loss(predicted_logp, target_logp.to(device)))
                if predicted_feature is not None:
                    losses.append(nn.functional.smooth_l1_loss(predicted_feature, target_feature.to(device)))
                val_losses.append(float(torch.stack(losses).sum().cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        print(f"[{branch}] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_set.inputs.shape[1]), "branch": branch, "input_mean": input_mean, "input_std": input_std, "seed": SEED, "epoch": epoch}, checkpoint)
    summary = {"branch": branch, "seed": SEED, "epochs": EPOCHS, "batch_size": batch_size, "hidden_dim": HIDDEN_DIM, "optimizer": "Adam", "learning_rate": 1e-3, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_model(checkpoint: Path, branch: str, input_dim: int, device: torch.device) -> EvidencePredictor:
    model = EvidencePredictor(input_dim, branch).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    f1_values: list[float] = []
    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(LABELS):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": int(support), "recall": recall, "f1": f1}
    return {"n": int(labels.size), "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0, "macro_f1": float(np.mean(f1_values)), "per_class": per_class, "confusion_matrix": confusion.tolist()}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _flatten_inputs(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    target_feature: np.ndarray,
    dino_lookup: Mapping[tuple[str, str, str, int], int],
    dino_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    inputs: list[np.ndarray] = []
    logp: list[np.ndarray] = []
    features: list[np.ndarray] = []
    refs: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        dino_mean = np.asarray(dino_embeddings[dino_lookup[key]], dtype=np.float32).mean(axis=0)
        current = np.asarray(row["current_feature"], dtype=np.float32)
        if current.shape != (CURRENT_FEATURE_DIM,):
            raise ValueError(f"current_feature shape mismatch: {current.shape}")
        row_ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        row_geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        valid = np.flatnonzero(cache["candidate_mask"][row_index])
        for candidate_index in valid:
            position = row_ids.index(int(cache["candidate_ids"][row_index, candidate_index]))
            inputs.append(np.concatenate([current, dino_mean, row_geometry[position]], axis=0))
            logp.append(np.asarray(cache["candidate_logp"][row_index, candidate_index], dtype=np.float32))
            features.append(np.asarray(target_feature[row_index, candidate_index], dtype=np.float32))
            refs.append((row_index, int(candidate_index)))
    return np.asarray(inputs, dtype=np.float32), np.asarray(logp, dtype=np.float32), np.asarray(features, dtype=np.float32), refs


def _predict_candidates(
    model: EvidencePredictor,
    inputs: np.ndarray,
    refs: Sequence[tuple[int, int]],
    row_count: int,
    max_candidates: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_logp = np.zeros((row_count, max_candidates, NUM_CLASSES), dtype=np.float32)
    pred_feature = np.zeros((row_count, max_candidates, STGCN_FEATURE_DIM), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            out_logp, out_feature = model(torch.from_numpy(inputs[start : start + batch_size]).to(device))
            if out_logp is not None:
                log_values = out_logp.cpu().numpy().astype(np.float32)
            else:
                log_values = np.zeros((len(refs[start : start + batch_size]), NUM_CLASSES), dtype=np.float32)
            if out_feature is not None:
                feature_values = out_feature.cpu().numpy().astype(np.float32)
            else:
                feature_values = np.zeros((len(refs[start : start + batch_size]), STGCN_FEATURE_DIM), dtype=np.float32)
            for offset, (row_index, candidate_index) in enumerate(refs[start : start + batch_size]):
                pred_logp[row_index, candidate_index] = log_values[offset]
                pred_feature[row_index, candidate_index] = feature_values[offset]
    return pred_logp, pred_feature


def _select_metrics(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    pred_logp: np.ndarray,
    pred_feature: np.ndarray,
    branch: str,
    stgcn: STGCN,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_branch = branch.removesuffix("_belief")
    use_belief_score = branch.endswith("_belief")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    true_logp = np.asarray(cache["candidate_logp"], dtype=np.float32)
    current_logp = np.asarray(cache["current_logp"], dtype=np.float32)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    if base_branch == "feature":
        with torch.inference_mode():
            flat = torch.from_numpy(pred_feature[mask]).to(device)
            pred_logp = torch.log_softmax(stgcn.fc(flat), dim=-1).cpu().numpy().astype(np.float32)
        rebuilt = np.zeros_like(pred_feature[..., :NUM_CLASSES])
        rebuilt[mask] = pred_logp
        pred_logp = rebuilt
    predicted_classes = np.argmax(pred_logp, axis=-1)
    selected_predictions: list[int] = []
    selected_ids: list[int] = []
    selected_scores: list[float] = []
    real_scores: list[float] = []
    entropy_values: list[float] = []
    belief = np.exp(current_logp)
    for row_index, label in enumerate(labels):
        valid = np.flatnonzero(mask[row_index])
        probabilities = np.exp(pred_logp[row_index, valid] - pred_logp[row_index, valid].max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if use_belief_score:
            scores = probabilities @ belief[row_index]
        else:
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
            scores = -entropy
        order = sorted(range(len(valid)), key=lambda position: (-float(scores[position]), int(cache["candidate_ids"][row_index, valid[position]])))
        chosen = int(valid[order[0]])
        selected_predictions.append(int(np.argmax(true_logp[row_index, chosen])))
        selected_ids.append(int(cache["candidate_ids"][row_index, chosen]))
        selected_scores.append(float(scores[order[0]]))
        real_scores.append(float(true_logp[row_index, chosen, label]))
        entropy_values.append(float(-np.sum(probabilities[order[0]] * np.log(np.clip(probabilities[order[0]], 1e-12, 1.0)))))
    metric = _metrics(labels, np.asarray(selected_predictions, dtype=np.int64))
    metric.update({"move_rate": 1.0, "stay_rate": 0.0, "selected_candidate_correct_rate": float(np.mean(np.asarray(selected_predictions) == labels)), "selected_candidate_ids": selected_ids, "mean_selected_predicted_score": float(np.mean(selected_scores)), "mean_selected_true_gt_logp": float(np.mean(real_scores)), "mean_predicted_entropy": float(np.mean(entropy_values))})
    return metric, {"selected_ids": selected_ids, "predicted_classes": predicted_classes[mask].tolist()}


def _candidate_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    pred_logp: np.ndarray,
    pred_feature: np.ndarray,
    refs: Sequence[tuple[int, int]],
    branch: str,
    stgcn: STGCN,
    device: torch.device,
) -> dict[str, Any]:
    if branch == "feature":
        with torch.inference_mode():
            predicted_flat = torch.log_softmax(stgcn.fc(torch.from_numpy(pred_feature[cache["candidate_mask"]]).to(device)), dim=-1).cpu().numpy()
        pred_logp = np.zeros_like(cache["candidate_logp"], dtype=np.float32)
        pred_logp[cache["candidate_mask"]] = predicted_flat
    target = np.asarray(cache["candidate_logp"], dtype=np.float32)[cache["candidate_mask"]]
    predicted = np.asarray(pred_logp, dtype=np.float32)[cache["candidate_mask"]]
    true_classes = target.argmax(axis=1)
    pred_classes = predicted.argmax(axis=1)
    top3 = np.argsort(predicted, axis=1)[:, -3:]
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    gt_scores_true: list[float] = []
    gt_scores_pred_entropy: list[float] = []
    gt_scores_pred_belief: list[float] = []
    predicted_best_overlap_entropy = 0
    predicted_best_overlap_belief = 0
    cursor = 0
    for row_index, row in enumerate(rows):
        valid = np.flatnonzero(cache["candidate_mask"][row_index])
        count = len(valid)
        probs = np.exp(predicted[cursor : cursor + count] - predicted[cursor : cursor + count].max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        entropy_scores = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        belief_scores = probs @ np.exp(cache["current_logp"][row_index])
        gt_scores_true.extend(cache["candidate_logp"][row_index, valid, labels[row_index]].tolist())
        gt_scores_pred_entropy.extend((-entropy_scores).tolist())
        gt_scores_pred_belief.extend(belief_scores.tolist())
        real_best = int(valid[np.argmax(cache["candidate_logp"][row_index, valid, labels[row_index]])])
        entropy_best = int(valid[np.argmax(-entropy_scores)])
        belief_best = int(valid[np.argmax(belief_scores)])
        predicted_best_overlap_entropy += int(entropy_best == real_best)
        predicted_best_overlap_belief += int(belief_best == real_best)
        cursor += count
    predicted_feature_flat = pred_feature[cache["candidate_mask"]]
    target_feature = None
    if branch in {"feature", "feature_logp"}:
        target_feature = np.asarray(predicted_feature_flat)
    diagnostics = {
        "candidate_count": int(len(target)),
        "logp_pearson": float(_pearson(predicted.reshape(-1), target.reshape(-1))),
        "logp_spearman": float(_spearman(predicted.reshape(-1), target.reshape(-1))),
        "top1_class_agreement": float(np.mean(pred_classes == true_classes)),
        "top3_class_agreement": float(np.mean([int(true_classes[i] in top3[i]) for i in range(len(true_classes))])),
        "score_spearman_vs_gt_true_logp": {"entropy": float(_spearman(gt_scores_pred_entropy, gt_scores_true)), "belief": float(_spearman(gt_scores_pred_belief, gt_scores_true))},
        "predicted_top_vs_real_best_gt_logp_overlap": {"entropy": float(predicted_best_overlap_entropy / len(rows)), "belief": float(predicted_best_overlap_belief / len(rows))},
    }
    if branch in {"feature", "feature_logp"}:
        # The caller adds the actual target feature array to diagnostics.
        diagnostics["feature_target_available"] = True
    return diagnostics


def _reference_metrics(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = json.loads((repo_root / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))
    visual = json.loads((repo_root / "experiments/reduced12_eight_placement_v1/h1_visual_context_batch/result.json").read_text(encoding="utf-8"))
    moving = {"S0-only": previous["metrics_moving"]["S0-only"], "FrozenStageCv0": previous["metrics_moving"]["FrozenStageCv0"], "Candidate-Conditioned Spatial H1": visual["metrics_moving"]["candidate_conditioned_spatial"], "SceneVisibility": json.loads((repo_root / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/result.json").read_text(encoding="utf-8"))["methods"]["SceneVisibility"], "AnyCorrect Oracle": previous["metrics_moving"]["AnyCorrect Oracle"]}
    full = {"S0-only": previous["metrics_full"]["S0-only"], "FrozenStageCv0": previous["metrics_full"]["FrozenStageCv0"], "AnyCorrect Oracle": previous["metrics_full"]["AnyCorrect Oracle"]}
    return moving, full


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root().resolve()
    device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    diagnostics_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    train_rows_all = load_jsonl(policy_root / "stage_c/features/train.jsonl")
    val_rows_all = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    stage_d_train_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/train.jsonl")}
    stage_d_val_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    dino_lookup, dino_embeddings, dino_summary = _load_dino(data_root)
    train_rows, train_indices = _covered_rows(train_rows_all, stage_d_train_ids, dino_lookup)
    val_rows, val_indices = _covered_rows(val_rows_all, stage_d_val_ids, dino_lookup)
    train_logp_cache = _subset_cache(_read_npz(diagnostics_root / "train_candidate_true_logp.npz"), train_indices)
    val_logp_cache = _subset_cache(_read_npz(diagnostics_root / "val_all_candidate_true_logp.npz"), val_indices)
    _candidate_alignment(train_rows, train_logp_cache)
    _candidate_alignment(val_rows, val_logp_cache)
    archive_root = data_root / "datasets/offline/habitat-train/00006-00087"
    stgcn_checkpoint = data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
    output = args.output_dir.resolve() if args.output_dir else EXPERIMENT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    target_root = data_root / "diagnostics/reduced12_future_recognition_evidence_prediction"
    train_target = _build_target_features(train_rows, train_logp_cache, archive_root, stgcn_checkpoint, target_root / "train_target_features.npz", target_root / "train_target_features.json", device, args.inference_batch_size)
    val_target = _build_target_features(val_rows, val_logp_cache, archive_root, stgcn_checkpoint, target_root / "val_moving_target_features.npz", target_root / "val_moving_target_features.json", device, args.inference_batch_size)
    train_inputs_raw, train_target_logp, train_target_feature, _ = _flatten_inputs(train_rows, train_logp_cache, train_target, dino_lookup, dino_embeddings)
    val_inputs_raw, val_target_logp, val_target_feature, val_refs = _flatten_inputs(val_rows, val_logp_cache, val_target, dino_lookup, dino_embeddings)
    input_mean = train_inputs_raw.mean(axis=0).astype(np.float32)
    input_std = train_inputs_raw.std(axis=0).astype(np.float32)
    input_std[input_std < 1e-6] = 1.0
    train_inputs = ((train_inputs_raw - input_mean) / input_std).astype(np.float32)
    val_inputs = ((val_inputs_raw - input_mean) / input_std).astype(np.float32)
    train_set = _EvidenceDataset(train_inputs, train_target_logp, train_target_feature)
    val_set = _EvidenceDataset(val_inputs, val_target_logp, val_target_feature)
    checkpoint_root = data_root / "checkpoints/policy_reduced12_eight_placement_v1/future_recognition_evidence_prediction"
    branches = ("logp", "feature", "feature_logp")
    training: dict[str, Any] = {}
    selectors: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    stgcn, _ = load_checkpoint(stgcn_checkpoint, NUM_CLASSES, str(device))
    for branch in branches:
        checkpoint = checkpoint_root / f"{branch}_best.pth"
        summary_path = output / f"{branch}_training.json"
        if checkpoint.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[{branch}] skip existing checkpoint", flush=True)
        else:
            summary = _train_branch(branch, train_set, val_set, input_mean, input_std, checkpoint, summary_path, device, args.batch_size)
        training[branch] = summary
        model = _load_model(checkpoint, branch, train_inputs.shape[1], device)
        pred_logp, pred_feature = _predict_candidates(model, val_inputs, val_refs, len(val_rows), int(val_logp_cache["candidate_logp"].shape[1]), device, args.batch_size)
        metric_entropy, _ = _select_metrics(val_rows, val_logp_cache, pred_logp, pred_feature, branch, stgcn, device)
        metric_entropy["selector"] = f"{branch.title()}-Entropy"
        metric_belief, _ = _select_metrics(val_rows, val_logp_cache, pred_logp, pred_feature, branch + "_belief", stgcn, device)
        metric_belief["selector"] = f"{branch.title()}-BeliefScore"
        selectors[f"{branch}_entropy"] = metric_entropy
        selectors[f"{branch}_belief"] = metric_belief
        branch_diag = _candidate_diagnostics(val_rows, val_logp_cache, pred_logp, pred_feature, val_refs, branch, stgcn, device)
        if branch in {"feature", "feature_logp"}:
            predicted_feature = pred_feature[val_logp_cache["candidate_mask"]]
            target_feature_flat = val_target[val_logp_cache["candidate_mask"]].astype(np.float32)
            branch_diag["feature_cosine_mean"] = float(np.mean(np.sum(predicted_feature * target_feature_flat, axis=1) / (np.linalg.norm(predicted_feature, axis=1) * np.linalg.norm(target_feature_flat, axis=1) + 1e-8)))
            branch_diag["feature_classification_agreement"] = float(np.mean(np.argmax(pred_logp[val_logp_cache["candidate_mask"]], axis=1) == np.argmax(val_target_logp, axis=1)))
        diagnostics[branch] = branch_diag
        np.savez_compressed(output / f"{branch}_val_predictions.npz", predicted_logp=pred_logp, predicted_feature=pred_feature)
    references_moving, references_full = _reference_metrics(REPO_ROOT)
    best_name = max(selectors, key=lambda name: float(selectors[name]["accuracy"]))
    frozen_accuracy = float(references_moving["FrozenStageCv0"]["accuracy"])
    frozen_f1 = float(references_moving["FrozenStageCv0"]["macro_f1"])
    result = {
        "experiment_id": "REDUCED12_FUTURE_RECOGNITION_EVIDENCE_PREDICTION",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "population": {"stage_c_train_rows": len(train_rows_all), "dino_covered_train_contexts": len(train_rows), "stage_c_val_rows": len(val_rows_all), "moving_val_contexts": len(val_rows), "train_candidate_examples": len(train_inputs), "moving_val_candidate_examples": len(val_inputs)},
        "references_moving": references_moving,
        "references_full": references_full,
        "metrics_full": {**references_full, "evidence_predictors": {"status": "NOT_AVAILABLE", "reason": "visited s0 DINO cache covers Moving Val only"}},
        "selectors_moving": selectors,
        "best_selector": {"name": best_name, "accuracy": selectors[best_name]["accuracy"], "macro_f1": selectors[best_name]["macro_f1"], "delta_vs_frozen_accuracy_pp": 100.0 * (selectors[best_name]["accuracy"] - frozen_accuracy), "delta_vs_frozen_macro_f1_pp": 100.0 * (selectors[best_name]["macro_f1"] - frozen_f1), "delta_vs_candidate_conditioned_spatial_accuracy_pp": 100.0 * (selectors[best_name]["accuracy"] - references_moving["Candidate-Conditioned Spatial H1"]["accuracy"]), "delta_vs_candidate_conditioned_spatial_macro_f1_pp": 100.0 * (selectors[best_name]["macro_f1"] - references_moving["Candidate-Conditioned Spatial H1"]["macro_f1"])},
        "training": training,
        "diagnostics": diagnostics,
        "target_feature_caches": {"train": str((target_root / "train_target_features.npz").resolve()), "val_moving": str((target_root / "val_moving_target_features.npz").resolve())},
        "full_val_visual_predictor_metrics": {"status": "NOT_AVAILABLE", "reason": "Existing visited s0 DINO cache covers Stage-D moving Val only; no DINO regeneration or silent fallback was used."},
        "dino_cache": {**dino_summary, "regenerated": False, "future_candidate_dino_used": False},
        "frozen_stgcn_checkpoint": str(stgcn_checkpoint.resolve()),
        "frozen_stgcn_checkpoint_sha256": file_sha256(stgcn_checkpoint),
        "protocol": {"inputs": ["s0 frozen ST-GCN feature (256-D)", "s0 frozen logp/current feature", "visited s0 DINO mean (16x768 pooled to 768-D)", "candidate geometry (11-D)"], "targets": ["archived candidate frozen ST-GCN logp", "archived candidate frozen ST-GCN feature"], "selectors": ["negative predicted entropy", "s0-belief expected confidence"], "terminal_observation": "selected real archived candidate skeleton through frozen ST-GCN", "deployable": True},
        "test_used": False,
        "train_used_for_supervised_evidence_prediction": True,
        "future_candidate_skeleton_used_for_target_only": True,
        "future_candidate_rgb_used": False,
        "future_candidate_dino_used": False,
        "gt_action_used": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_branch = {name: {"entropy": selectors[f"{name}_entropy"], "belief": selectors[f"{name}_belief"], "diagnostics": diagnostics[name]} for name in branches}
    (output / "per_branch_metrics.json").write_text(json.dumps(per_branch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Reduced12 Future Recognition Evidence Prediction", "", "Train/Val only; no policy Test was read. Existing visited s0 DINO and archived skeleton caches were reused; no RGB/skeleton/DINO regeneration was performed.", "", "## Moving Val references", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name, metric in references_moving.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend(["", "## Moving Val evidence selectors", "", "| Selector | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |", "|---|---:|---:|---:|---:|"])
    for name, metric in selectors.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {100.0 * (metric['accuracy'] - frozen_accuracy):+.3f}pp | {100.0 * (metric['macro_f1'] - frozen_f1):+.3f}pp |")
    lines.extend(["", "## Full Val", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, metric in references_full.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend(["| Evidence predictors | NOT_AVAILABLE | NOT_AVAILABLE |", "", f"Best evidence selector: **{best_name}**. Evidence selectors follow the requested direct argmax over legal candidates (no added stay score). Full-Val evidence predictors are **NOT_AVAILABLE** because the existing DINO cache covers only Moving Val; no fallback/regeneration was used.", "", "## Diagnostics", ""])
    for branch in branches:
        item = diagnostics[branch]
        lines.append(f"- **{branch}**: logp Pearson/Spearman {item['logp_pearson']:.6f}/{item['logp_spearman']:.6f}; top-1/top-3 class agreement {item['top1_class_agreement']:.6f}/{item['top3_class_agreement']:.6f}; score Spearman entropy/belief {item['score_spearman_vs_gt_true_logp']['entropy']:.6f}/{item['score_spearman_vs_gt_true_logp']['belief']:.6f}.")
    lines.extend(["", "## Scientific judgment", "", ("At least one simple evidence predictor exceeds the Candidate-Conditioned Spatial reference, supporting direct future recognizer-evidence prediction as a direction." if result["best_selector"]["delta_vs_candidate_conditioned_spatial_accuracy_pp"] >= 2.0 else "Simple evidence predictors do not establish a clear >=2pp gain over Candidate-Conditioned Spatial; s0+DINO+geometry may be insufficient for candidate-specific future evidence."), "No larger network, hyperparameter sweep, or formal WM/JR modification was performed.", "", "`test_used=false`; `future_candidate_skeleton_used_for_target_only=true`; `future_candidate_rgb_used=false`; `future_candidate_dino_used=false`; `deployable=true`."])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"], "best_selector": result["best_selector"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
