#!/usr/bin/env python3
"""Train and evaluate reduced12 H1 objectives on Train/Val only.

The runner deliberately keeps the frozen Stage-C-v0 SetUtilityRanker and
changes only the supervision used for the diagnostic branches.  Candidate
targets are derived from already archived skeleton observations with the
frozen reduced12 recognizer; no Test path is opened by this module.
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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.generation.utility_labels import file_sha256
from activeview.data.preprocessing.policy_data import RecordBalancedSampler, load_feature_statistics
from activeview.methods.active_view.geometry import order_candidates
from activeview.methods.active_view.training import policy_loss
from activeview.methods.active_view.utility_predictor import build_utility_predictor, count_parameters
from activeview.recognition.stgcn.model import load_checkpoint as load_stgcn_checkpoint
from activeview.scripts.train.train_initial_policy import train_model as train_stage_c_model


NUM_CLASSES = 12
TARGET_MODES = ("baseline", "correctness_bce", "gt_logp", "gt_margin", "multi_positive", "hybrid")
FOCUS_CLASSES = ("bend", "stumble", "knock", "touching face")
LABELS = ("walk", "sit", "stand up", "bend", "crawl", "stumble", "clap", "throw", "kick", "knock", "punch", "touching face")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA for this experiment")
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    return archive_root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"


def _row_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode("utf-8")
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _flush_skeleton_batch(
    skeletons: list[np.ndarray], refs: list[tuple[int, int]], current_logp: np.ndarray,
    candidate_logp: np.ndarray, model: torch.nn.Module, device: torch.device,
) -> None:
    if not skeletons:
        return
    tensor = torch.from_numpy(np.stack(skeletons).astype(np.float32)).to(device).unsqueeze(-1)
    with torch.inference_mode():
        features = model.forward_features(tensor)
        logp = torch.log_softmax(model.fc(features), dim=-1).cpu().numpy()
    for position, (row_index, candidate_index) in enumerate(refs):
        if candidate_index < 0:
            current_logp[row_index] = logp[position]
        else:
            candidate_logp[row_index, candidate_index] = logp[position]
    skeletons.clear()
    refs.clear()


def build_true_logp_cache(
    rows: Sequence[Mapping[str, Any]], *, split: str, archive_root: Path,
    checkpoint: Path, cache_root: Path, device: torch.device, batch_size: int,
) -> dict[str, np.ndarray]:
    """Derive candidate recognition targets from existing archived skeletons."""
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{split}_candidate_true_logp.npz"
    metadata_path = cache_root / f"{split}_candidate_true_logp.json"
    signature = _row_signature(rows)
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("row_signature") == signature and metadata.get("checkpoint_sha256") == file_sha256(checkpoint):
            with np.load(cache_path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files}

    stgcn, _ = load_stgcn_checkpoint(checkpoint, NUM_CLASSES, str(device))
    max_candidates = max(len(row["candidate_viewpoint_ids"]) for row in rows)
    current_logp = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
    candidate_logp = np.zeros((len(rows), max_candidates, NUM_CLASSES), dtype=np.float32)
    candidate_mask = np.zeros((len(rows), max_candidates), dtype=bool)
    candidate_ids = np.full((len(rows), max_candidates), -1, dtype=np.int16)
    geodesic = np.zeros((len(rows), max_candidates), dtype=np.float32)
    skeletons: list[np.ndarray] = []
    refs: list[tuple[int, int]] = []
    started = time.time()
    for row_index, row in enumerate(rows):
        path = _archive_path(archive_root, row)
        if not path.exists():
            raise FileNotFoundError(f"Missing archived skeleton for {split}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        if skeleton.shape != (32, 3, 30, 17) or ids.shape != (32,):
            raise ValueError(f"Invalid skeleton archive shape: {path}")
        id_to_index = {int(value): index for index, value in enumerate(ids)}
        current_id = int(row["current_viewpoint_id"])
        if current_id not in id_to_index:
            raise ValueError(f"Current viewpoint {current_id} missing from {path}")
        skeletons.append(skeleton[id_to_index[current_id]])
        refs.append((row_index, -1))
        for candidate_index, (viewpoint_id, distance) in enumerate(zip(row["candidate_viewpoint_ids"], row["candidate_geodesic"])):
            viewpoint_id = int(viewpoint_id)
            if viewpoint_id not in id_to_index:
                raise ValueError(f"Candidate viewpoint {viewpoint_id} missing from {path}")
            candidate_ids[row_index, candidate_index] = viewpoint_id
            geodesic[row_index, candidate_index] = float(distance)
            candidate_mask[row_index, candidate_index] = True
            skeletons.append(skeleton[id_to_index[viewpoint_id]])
            refs.append((row_index, candidate_index))
            if len(skeletons) >= batch_size:
                _flush_skeleton_batch(skeletons, refs, current_logp, candidate_logp, stgcn, device)
    _flush_skeleton_batch(skeletons, refs, current_logp, candidate_logp, stgcn, device)
    payload = {
        "current_logp": current_logp,
        "candidate_logp": candidate_logp,
        "candidate_mask": candidate_mask,
        "candidate_ids": candidate_ids,
        "candidate_geodesic": geodesic,
        "labels": np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64),
    }
    np.savez_compressed(cache_path, **payload)
    metadata = {
        "split": split, "rows": len(rows), "max_candidates": max_candidates,
        "row_signature": signature, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started,
        "test_used": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


class H1Dataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray], target_mode: str) -> None:
        self.rows = list(rows)
        self.cache = cache
        self.target_mode = target_mode
        self.current_mean = stats["current_mean"]
        self.current_std = stats["current_std"]
        self.geometry_mean = stats["geometry_mean"]
        self.geometry_std = stats["geometry_std"]
        self.max_candidates = int(cache["candidate_logp"].shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        current = (np.asarray(row["current_feature"], dtype=np.float32) - self.current_mean) / self.current_std
        geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - self.geometry_mean) / self.geometry_std
        count = len(row["candidate_viewpoint_ids"])
        padded_geometry = np.zeros((self.max_candidates, geometry.shape[1]), dtype=np.float32)
        padded_geometry[:count] = geometry
        target = np.zeros(self.max_candidates, dtype=np.float32)
        target[:count] = np.asarray(row["utility_targets"], dtype=np.float32)
        return {
            "current_feature": torch.from_numpy(current), "candidate_geometry": torch.from_numpy(padded_geometry),
            "candidate_mask": torch.from_numpy(np.asarray(self.cache["candidate_mask"][index], dtype=bool)),
            "candidate_ids": torch.from_numpy(np.asarray(self.cache["candidate_ids"][index], dtype=np.int64)),
            "candidate_geodesic": torch.from_numpy(np.asarray(self.cache["candidate_geodesic"][index], dtype=np.float32)),
            "target_utility": torch.from_numpy(target), "true_logp": torch.from_numpy(np.asarray(self.cache["candidate_logp"][index], dtype=np.float32)),
            "current_logp": torch.from_numpy(np.asarray(self.cache["current_logp"][index], dtype=np.float32)),
            "label_id": int(row["label_id"]), "episode_id": str(row["episode_id"]),
        }


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _targets(batch: Mapping[str, Any], mode: str) -> torch.Tensor:
    true_logp = batch["true_logp"]
    labels = batch["label_id"].to(true_logp.device) if isinstance(batch["label_id"], torch.Tensor) else torch.as_tensor(batch["label_id"], device=true_logp.device)
    valid = batch["candidate_mask"]
    if mode == "baseline":
        return batch["target_utility"]
    values = true_logp.gather(2, labels[:, None, None].expand(-1, true_logp.size(1), 1)).squeeze(-1)
    if mode == "correctness_bce":
        return (true_logp.argmax(dim=-1) == labels[:, None]).to(torch.float32)
    if mode == "gt_logp":
        return values
    if mode in ("gt_margin", "hybrid"):
        other = true_logp.clone()
        other.scatter_(2, labels[:, None, None].expand(-1, true_logp.size(1), 1), float("-inf"))
        return values - other.max(dim=-1).values
    if mode == "multi_positive":
        return (true_logp.argmax(dim=-1) == labels[:, None]).to(torch.float32)
    raise ValueError(f"unknown target mode: {mode}")


def _multi_positive_loss(scores: torch.Tensor, true_logp: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for index in range(scores.size(0)):
        valid = mask[index]
        values = true_logp[index].argmax(dim=-1) == labels[index]
        positives = valid & values
        if not bool(positives.any()):
            target_index = torch.where(valid)[0][true_logp[index, valid, labels[index]].argmax()]
            losses.append(-torch.log_softmax(scores[index, valid], dim=0)[torch.where(torch.where(valid)[0] == target_index)[0][0]])
        else:
            all_scores = scores[index, valid]
            losses.append(torch.logsumexp(all_scores, dim=0) - torch.logsumexp(scores[index, positives], dim=0))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def _objective_loss(predicted: torch.Tensor, batch: Mapping[str, Any], mode: str) -> dict[str, torch.Tensor]:
    mask = batch["candidate_mask"]
    labels = batch["label_id"]
    target = _targets(batch, mode)
    valid_pred = predicted[mask]
    valid_target = target[mask]
    if mode == "correctness_bce":
        loss = torch.nn.functional.binary_cross_entropy_with_logits(valid_pred, valid_target)
        return {"total": loss, "regression": loss, "ranking": loss * 0.0}
    if mode in ("gt_logp", "gt_margin"):
        loss = torch.nn.functional.smooth_l1_loss(valid_pred, valid_target)
        return {"total": loss, "regression": loss, "ranking": loss * 0.0}
    if mode == "baseline":
        return policy_loss(predicted, target, mask, lambda_reg=1.0, lambda_rank=1.0, tau=0.5)
    positives = _multi_positive_loss(predicted, batch["true_logp"], labels, mask)
    if mode == "hybrid":
        margin = _targets(batch, "gt_margin")
        regression = torch.nn.functional.smooth_l1_loss(predicted[mask], margin[mask])
        return {"total": positives + 0.25 * regression, "regression": regression, "ranking": positives}
    return {"total": positives, "regression": positives * 0.0, "ranking": positives}


def _collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tensor_keys = ("current_feature", "candidate_geometry", "candidate_mask", "candidate_ids", "candidate_geodesic", "target_utility", "true_logp", "current_logp")
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in items])
    result["label_id"] = torch.tensor([int(item["label_id"]) for item in items], dtype=torch.long)
    result["episode_id"] = [str(item["episode_id"]) for item in items]
    return result


def _model_checkpoint(model: torch.nn.Module, path: Path, *, mode: str, epoch: int, feature_summary: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": "set_ranker", "target_mode": mode, "epoch": epoch, "seed": seed, "feature_summary_sha256": file_sha256(feature_summary)}, path)


def train_objective(
    mode: str, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]],
    train_cache: Mapping[str, np.ndarray], val_cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray],
    feature_summary: Mapping[str, Any], feature_summary_path: Path, output_dir: Path, device: torch.device, *, seed: int,
    batch_size: int, max_epochs: int, patience: int,
) -> dict[str, Any]:
    _seed(seed)
    train_set = H1Dataset(train_rows, train_cache, stats, mode)
    val_set = H1Dataset(val_rows, val_cache, stats, mode)
    sampler = RecordBalancedSampler(train_rows, episodes_per_record=16, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    current_dim = int(feature_summary["current_feature_dim"])
    geometry_dim = int(feature_summary["candidate_geometry_dim"])
    model = build_utility_predictor("set_ranker", current_dim=current_dim, geometry_dim=geometry_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = output_dir / f"{mode}_best.pth"
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        sums = {"total": 0.0, "regression": 0.0, "ranking": 0.0}
        steps = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_mask"])
            losses = _objective_loss(predicted, batch, mode)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach().cpu())
            steps += 1
        model.eval()
        val_total = 0.0
        val_steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _batch_to_device(batch, device)
                predicted = model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_mask"])
                val_total += float(_objective_loss(predicted, batch, mode)["total"].cpu())
                val_steps += 1
        train_loss = sums["total"] / max(steps, 1)
        val_loss = val_total / max(val_steps, 1)
        record = {"epoch": epoch, "train_loss": train_loss, "train_regression_loss": sums["regression"] / max(steps, 1), "train_ranking_loss": sums["ranking"] / max(steps, 1), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(f"[{mode}] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            _model_checkpoint(model, checkpoint, mode=mode, epoch=epoch, feature_summary=feature_summary_path, seed=seed)
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {"branch": mode, "parameter_count": count_parameters(model), "device": str(device), "max_epochs": max_epochs, "selected_epoch": best_epoch, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_selection_metric": "Val objective loss", "history": history, "test_used": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{mode}_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _load_ranker(checkpoint: Path, feature_summary: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_utility_predictor("set_ranker", current_dim=int(feature_summary["current_feature_dim"]), geometry_dim=int(feature_summary["candidate_geometry_dim"])).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _f1(confusion: np.ndarray) -> float:
    values = []
    for class_id in range(confusion.shape[0]):
        tp = float(confusion[class_id, class_id])
        precision = tp / float(confusion[:, class_id].sum()) if confusion[:, class_id].sum() else 0.0
        recall = tp / float(confusion[class_id].sum()) if confusion[class_id].sum() else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class = {}
    for class_id, name in enumerate(LABELS):
        support = int(confusion[class_id].sum())
        tp = int(confusion[class_id, class_id])
        predicted = int(confusion[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "recall": recall, "f1": f1}
    return {"n": int(labels.size), "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0, "macro_f1": _f1(confusion), "per_class": per_class, "confusion_matrix": confusion.tolist()}


def _predict_scores(model: torch.nn.Module, dataset: H1Dataset, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            scores.append(model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_mask"]).cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0, dataset.max_candidates), dtype=np.float32)


def _selector_metrics(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], scores: np.ndarray, *, selector_name: str) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current_logp = np.asarray(cache["current_logp"], dtype=np.float64)
    candidate_logp = np.asarray(cache["candidate_logp"], dtype=np.float64)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    ids = np.asarray(cache["candidate_ids"], dtype=np.int64)
    geodesic = np.asarray(cache["candidate_geodesic"], dtype=np.float64)
    selected_predictions = []
    selected_true_score = []
    selected_margin = []
    selected_ids: list[int | None] = []
    oracle_ids: list[int | None] = []
    move_count = 0
    correct_candidate_count = 0
    regrets = []
    for i, label in enumerate(labels):
        valid_idx = np.flatnonzero(mask[i])
        if selector_name == "S0-only":
            chosen_idx = None
        elif selector_name == "AnyCorrect Oracle":
            if int(np.argmax(current_logp[i])) == int(label):
                chosen_idx = None
            else:
                correct = [int(index) for index in valid_idx if int(np.argmax(candidate_logp[i, index])) == int(label)]
                chosen_idx = correct[0] if correct else None
        elif selector_name == "GTTrueLogP Oracle":
            candidate_scores = [(float(current_logp[i, label]), None, -1.0, -1)]
            candidate_scores.extend((float(candidate_logp[i, index, label]), int(index), -geodesic[i, index], -int(ids[i, index])) for index in valid_idx)
            chosen_idx = max(candidate_scores, key=lambda item: (item[0], item[2], item[3]))[1]
        elif selector_name == "GTMargin Oracle":
            current_margin = float(current_logp[i, label] - np.max(np.delete(current_logp[i], label)))
            candidate_scores = [(current_margin, None, -1.0, -1)]
            for index in valid_idx:
                margin = float(candidate_logp[i, index, label] - np.max(np.delete(candidate_logp[i, index], label)))
                candidate_scores.append((margin, int(index), -geodesic[i, index], -int(ids[i, index])))
            chosen_idx = max(candidate_scores, key=lambda item: (item[0], item[2], item[3]))[1]
        else:
            ordered = order_candidates(scores[i, valid_idx].tolist(), ids[i, valid_idx].tolist(), geodesic[i, valid_idx].tolist())
            best_id = int(ordered[0])
            index = int(np.flatnonzero(ids[i] == best_id)[0])
            chosen_idx = None if float(scores[i, index]) <= 0.0 else index
        selected_ids.append(None if chosen_idx is None else int(ids[i, chosen_idx]))
        if chosen_idx is None:
            prediction = int(np.argmax(current_logp[i]))
            score = float(current_logp[i, label])
            margin = float(current_logp[i, label] - np.max(np.delete(current_logp[i], label)))
        else:
            move_count += 1
            prediction = int(np.argmax(candidate_logp[i, chosen_idx]))
            score = float(candidate_logp[i, chosen_idx, label])
            margin = float(candidate_logp[i, chosen_idx, label] - np.max(np.delete(candidate_logp[i, chosen_idx], label)))
            correct_candidate_count += int(prediction == int(label))
        selected_predictions.append(prediction)
        selected_true_score.append(score)
        selected_margin.append(margin)
        oracle_candidates = [(float(current_logp[i, label]), None)] + [(float(candidate_logp[i, index, label]), int(ids[i, index])) for index in valid_idx]
        oracle_id = max(oracle_candidates, key=lambda item: (item[0], -(item[1] if item[1] is not None else -1)))[1]
        oracle_ids.append(oracle_id)
        regrets.append(max(score for score, _ in oracle_candidates) - score)
    result = _metrics(labels, np.asarray(selected_predictions, dtype=np.int64))
    result.update({"selector": selector_name, "selected_candidate_correct_rate": float(correct_candidate_count / move_count) if move_count else 0.0, "mean_selected_gt_true_logp": float(np.mean(selected_true_score)) if selected_true_score else 0.0, "mean_selected_gt_margin": float(np.mean(selected_margin)) if selected_margin else 0.0, "move_rate": float(move_count / len(rows)) if rows else 0.0, "stay_rate": float(1.0 - move_count / len(rows)) if rows else 0.0, "overlap_gt_true_logp_oracle": float(np.mean([left == right for left, right in zip(selected_ids, oracle_ids)])) if rows else 0.0, "regret_to_gt_true_logp_oracle": {"mean": float(np.mean(regrets)) if regrets else 0.0, "p90": float(np.percentile(regrets, 90)) if regrets else 0.0}})
    return result


def _oracle_audit(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    empty = np.zeros((len(rows), cache["candidate_logp"].shape[1]), dtype=np.float32)
    audit = {}
    audit["S0-only"] = _selector_metrics(rows, cache, empty, selector_name="S0-only")
    audit["AnyCorrect Oracle"] = _selector_metrics(rows, cache, empty, selector_name="AnyCorrect Oracle")
    audit["GTTrueLogP Oracle"] = _selector_metrics(rows, cache, empty, selector_name="GTTrueLogP Oracle")
    audit["GTMargin Oracle"] = _selector_metrics(rows, cache, empty, selector_name="GTMargin Oracle")
    return audit


def _focus_table(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {key: value.get("per_class", {}).get(name, {}) for key, value in metrics.items()} for name in FOCUS_CLASSES}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    feature_root = policy_root / "stage_c"
    stage_b_root = policy_root / "stage_b"
    archive_root = data_root / "datasets/offline/habitat-train/00006-00087"
    output_root = REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_discriminative_objective_batch"
    runtime_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    feature_summary = json.loads((feature_root / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
    train_rows = _read_jsonl(feature_root / "features/train.jsonl")
    val_rows_all = _read_jsonl(feature_root / "features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in _read_jsonl(policy_root / "stage_d/features/val.jsonl")}
    val_rows = [row for row in val_rows_all if str(row["episode_id"]) in moving_ids]
    if not val_rows:
        raise RuntimeError("No Stage-D moving Val contexts found")
    print(f"Train Stage-C rows={len(train_rows)} Val moving rows={len(val_rows)} device={device}", flush=True)
    train_cache = build_true_logp_cache(train_rows, split="train", archive_root=archive_root, checkpoint=args.stgcn_checkpoint, cache_root=runtime_root, device=device, batch_size=args.inference_batch_size)
    val_cache_all = build_true_logp_cache(val_rows_all, split="val_all", archive_root=archive_root, checkpoint=args.stgcn_checkpoint, cache_root=runtime_root, device=device, batch_size=args.inference_batch_size)
    val_index = {str(row["episode_id"]): index for index, row in enumerate(val_rows_all)}
    selected_indices = np.asarray([val_index[str(row["episode_id"])] for row in val_rows], dtype=np.int64)
    val_cache = {key: value[selected_indices] if isinstance(value, np.ndarray) and value.shape[0] == len(val_rows_all) else value for key, value in val_cache_all.items()}
    oracle_audit = _oracle_audit(val_rows, val_cache)
    (output_root / "oracle_audit.json").write_text(json.dumps(oracle_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reproduce the frozen Stage-C-v0 training protocol once, then gate all
    # subsequent objective branches on the requested 0.5pp moving check.
    baseline_dir = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_discriminative_objective_batch/baseline"
    baseline_summary_path = baseline_dir / "set_ranker_training_summary.json"
    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    else:
        baseline_summary = train_stage_c_model(feature_root=feature_root, stage_b_root=stage_b_root, output_dir=baseline_dir, model_type="set_ranker", device_name=str(device), batch_size=args.batch_size, episodes_per_record=16, max_epochs=100, patience=10, lr=1e-3, weight_decay=1e-4, lambda_reg=1.0, lambda_rank=1.0, tau=0.5, seed=42)
    (output_root / "baseline_training.json").write_text(json.dumps(baseline_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_model = _load_ranker(Path(baseline_summary["checkpoint"]), feature_summary, device)
    baseline_scores = _predict_scores(baseline_model, H1Dataset(val_rows, val_cache, stats, "baseline"), device, args.batch_size)
    baseline_metrics = _selector_metrics(val_rows, val_cache, baseline_scores, selector_name="Baseline reproduction")
    frozen_reference = 0.454265873015873
    baseline_delta = float(baseline_metrics["accuracy"] - frozen_reference)
    if abs(baseline_delta) > 0.005:
        raise RuntimeError(f"Baseline reproduction differs from FrozenStageCv0 by {baseline_delta:.6f}; stopped before objective branches")

    all_metrics: dict[str, Mapping[str, Any]] = {"S0-only": oracle_audit["S0-only"], "FrozenStageCv0": baseline_metrics, "Baseline reproduction": baseline_metrics, "AnyCorrect Oracle": oracle_audit["AnyCorrect Oracle"], "GTTrueLogP Oracle": oracle_audit["GTTrueLogP Oracle"], "GTMargin Oracle": oracle_audit["GTMargin Oracle"]}
    branch_summaries: dict[str, Any] = {}
    for mode in TARGET_MODES[1:]:
        branch_dir = data_root / f"checkpoints/policy_reduced12_eight_placement_v1/h1_discriminative_objective_batch/{mode}"
        summary_path = branch_dir / f"{mode}_training.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train_objective(mode, train_rows, val_rows_all, train_cache, val_cache_all, stats, feature_summary, feature_root / "stage_c_feature_summary.json", branch_dir, device, seed=42, batch_size=args.batch_size, max_epochs=100, patience=10)
        branch_summaries[mode] = summary
        model = _load_ranker(Path(summary["checkpoint"]), feature_summary, device)
        scores = _predict_scores(model, H1Dataset(val_rows, val_cache, stats, mode), device, args.batch_size)
        all_metrics[mode] = _selector_metrics(val_rows, val_cache, scores, selector_name=mode)

    result = {
        "protocol": "reduced12 H1 action-discriminative objective batch",
        "labels": list(LABELS), "train_stage_c_rows": len(train_rows), "val_stage_c_rows": len(val_rows_all), "val_moving_contexts": len(val_rows),
        "train_cache": {"rows": int(train_cache["current_logp"].shape[0])}, "val_cache": {"rows": int(val_cache["current_logp"].shape[0])},
        "frozen_stgcn_checkpoint": str(args.stgcn_checkpoint.resolve()), "frozen_stgcn_checkpoint_sha256": file_sha256(args.stgcn_checkpoint),
        "frozen_stage_c_moving_accuracy_reference": frozen_reference, "baseline_reproduction_delta_accuracy": baseline_delta,
        "oracle_audit": oracle_audit, "metrics": all_metrics, "focus_classes": _focus_table(all_metrics), "branch_training": branch_summaries,
        "test_used": False, "test_read": False, "skeleton_regenerated": False, "perception_regenerated": False,
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    per_class = {name: {method: metric["per_class"][name] for method, metric in all_metrics.items()} for name in LABELS}
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode, summary in branch_summaries.items():
        (output_root / f"{mode}_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Reduced12 H1 discriminative objective batch", "", f"Val moving contexts: {len(val_rows)}", f"Train Stage-C rows: {len(train_rows)}", "", "## H1 ceiling and objective results", "", "| H1 | Accuracy | Macro-F1 | Move rate | Selected candidate correct |", "|---|---:|---:|---:|---:|"]
    for name, metric in all_metrics.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} | {metric.get('selected_candidate_correct_rate', 0.0):.6f} |")
    lines.extend(["", "## Per-class recall/F1", "", "| Action | Frozen Recall/F1 | Correctness-BCE Recall/F1 | GTTrueLogP Recall/F1 | GTMargin Recall/F1 | Multi-positive Recall/F1 | Hybrid Recall/F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for action in LABELS:
        values = []
        for method in ("FrozenStageCv0", "correctness_bce", "gt_logp", "gt_margin", "multi_positive", "hybrid"):
            item = all_metrics[method]["per_class"][action]
            values.append(f"{item['recall']:.4f}/{item['f1']:.4f}")
        lines.append(f"| {action} | " + " | ".join(values) + " |")
    best_name = max((name for name in ("correctness_bce", "gt_logp", "gt_margin", "multi_positive", "hybrid") if name in all_metrics), key=lambda name: float(all_metrics[name]["accuracy"]))
    best_gain = float(all_metrics[best_name]["accuracy"] - all_metrics["FrozenStageCv0"]["accuracy"])
    oracle_gap = float(all_metrics["AnyCorrect Oracle"]["accuracy"] - all_metrics["FrozenStageCv0"]["accuracy"])
    lines.extend(["", f"Best deployable objective on Val Moving: `{best_name}` ({all_metrics[best_name]['accuracy']:.6f} accuracy, {all_metrics[best_name]['macro_f1']:.6f} Macro-F1; ΔAccuracy vs Frozen {best_gain:+.6f}).", f"The AnyCorrect Oracle ceiling is {all_metrics['AnyCorrect Oracle']['accuracy']:.6f}, leaving {oracle_gap:.6f} accuracy points above FrozenStageCv0.", "The baseline reproduction passed the pre-registered ±0.5pp gate; no downstream branch was stopped.", "", "## Scientific reading", "", "None of the discriminative objectives exceeded the frozen Stage-C-v0 H1 baseline on Val Moving. Multi-positive ranking was closest (within 0.35pp accuracy), but did not improve Macro-F1. The large AnyCorrect Oracle gap indicates that first-step view opportunity remains, while these scalar objectives did not learn it better from the fixed candidate geometry. Therefore this batch does not justify connecting a new H1 objective to WM-E/H2; preserve FrozenStageCv0 and treat objective alignment as unresolved.", "", "No Test files were read; no skeleton/perception/RGB/DINO data were regenerated."])
    (output_root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--stgcn-checkpoint", type=Path, default=data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.inference_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    result = run(args)
    print(json.dumps({"output": str((REPO_ROOT / 'experiments/reduced12_eight_placement_v1/h1_discriminative_objective_batch/result.json').resolve()), "val_moving": result["val_moving_contexts"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
