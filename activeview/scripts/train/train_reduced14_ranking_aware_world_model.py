#!/usr/bin/env python3
"""Train the ranking-aware reduced14 WM-E and evaluate it on Val.

The old WM-E checkpoint initializes the pose world model.  This run adds a
candidate recognition head and fixed KL/ranking terms, then rebuilds only the
Train/Val counterfactual caches and retrains the frozen-protocol
Pretrained-Frozen history JR on the new imagined recognition scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.world_model.model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
    world_model_loss,
)
from activeview.scripts.data.build_reduced14_counterfactual_cache import build_split
from activeview.scripts.eval.analyze_reduced14_wm_e import _correlation
from activeview.scripts.train.train_reduced14_history_aware_jr import (
    _HistoryDataset,
    _history_inputs,
    _load_prior_methods,
    _method_metrics,
    _select_history_actions,
    _sha256 as jr_sha256,
)
from activeview.scripts.train.train_reduced14_joint_revision import _examples, _orders
from activeview.scripts.train.train_reduced14_world_model import _rgb_lookup, _sources
from activeview.methods.joint_revision.pretrained_history_aware import (
    PretrainedHistoryAwareJointRevision,
)


SEED = 42
NUM_CLASSES = 14
VIEW_COUNT = 32
WM_EPOCHS = 12
WM_BATCH_SIZE = 64
WM_WORKERS = 4
WM_LR = 1e-3
WM_WEIGHT_DECAY = 1e-4
RECOGNITION_WEIGHT = 0.1
RANKING_WEIGHT = 0.2
MAX_RANK_PAIRS = 16
JR_EPOCHS = 20
JR_BATCH_SIZE = 512
JR_LR = 1e-3
JR_WEIGHT_DECAY = 1e-4
JR_LAMBDA_ID = 0.2
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/ranking_aware_wm_e"
DATASET_NAME = "policy_reduced14_kneel_eight_placement_v1"
CHECKPOINT_DIR_NAME = "activeview_reduced14_eight_placement_v1"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _pairwise_ranking_loss(
    predicted_logp: torch.Tensor,
    true_logp: torch.Tensor,
    legal_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Use at most a fixed number of legal within-context candidate pairs."""
    terms: list[torch.Tensor] = []
    batch_size = predicted_logp.size(0)
    for batch_index in range(batch_size):
        valid = torch.nonzero(legal_mask[batch_index], as_tuple=False).flatten().tolist()
        pair_count = 0
        for left_offset, left in enumerate(valid):
            for right in valid[left_offset + 1 :]:
                target_delta = true_logp[batch_index, left, labels[batch_index]] - true_logp[batch_index, right, labels[batch_index]]
                if float(target_delta.detach()) == 0.0:
                    continue
                sign = 1.0 if float(target_delta.detach()) > 0.0 else -1.0
                pred_delta = predicted_logp[batch_index, left, labels[batch_index]] - predicted_logp[batch_index, right, labels[batch_index]]
                terms.append(torch.nn.functional.softplus(-sign * pred_delta))
                pair_count += 1
                if pair_count >= MAX_RANK_PAIRS:
                    break
            if pair_count >= MAX_RANK_PAIRS:
                break
    if not terms:
        return predicted_logp.sum() * 0.0
    return torch.stack(terms).mean()


def _train_step(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[float, float, float, float]:
    kwargs = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
    }
    kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
    kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    candidate_mask = batch["candidate_mask"].to(device)
    labels = batch["label_id"].to(device)
    prediction, recognition_logits = model(
        **kwargs,
        return_recognition=True,
    )
    valid = candidate_mask.reshape(-1)
    predicted_flat = prediction.reshape(-1, 3, 30, 17)[valid]
    truth_flat = target.reshape(-1, 3, 30, 17)[valid]
    pose_total, pose, velocity = world_model_loss(predicted_flat, truth_flat)
    with torch.no_grad():
        true_logp = torch.log_softmax(
            teacher(truth_flat), dim=-1
        ).reshape(prediction.size(0), prediction.size(1), NUM_CLASSES)
    predicted_logp = torch.log_softmax(recognition_logits, dim=-1)
    valid_logp = predicted_logp.reshape(-1, NUM_CLASSES)[valid]
    valid_true_logp = true_logp.reshape(-1, NUM_CLASSES)[valid]
    recognition = torch.nn.functional.kl_div(
        torch.log_softmax(valid_logp, dim=-1), valid_true_logp.exp(), reduction="batchmean"
    )
    ranking = _pairwise_ranking_loss(
        predicted_logp,
        true_logp,
        batch["legal_candidate_mask"].to(device) & candidate_mask,
        labels,
    )
    loss = pose_total + RECOGNITION_WEIGHT * recognition + RANKING_WEIGHT * ranking
    return loss, pose, recognition, ranking


def _build_wm_loader(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    workers: int,
) -> DataLoader:
    candidate_orders = _orders(data_root, rows)
    legal_by_context = {
        (str(row["scene_id"]), str(row["region"]), str(row["record_id"])): tuple(
            int(value) for value in candidate_orders[str(row["episode_id"])]
        )
        for row in rows
    }
    dataset = LazyWorldModelContextDataset(
        rows,
        _sources(data_root, [dict(row) for row in rows]),
        use_belief=True,
        rgb_lookup=_rgb_lookup(data_root),
        target_scope="all",
        cache_size=64,
        legal_candidate_ids=legal_by_context,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_world_model_context,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _evaluate_wm_model(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    loader: DataLoader,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate recognition-head ranking on Val without writing a cache."""
    model.eval()
    predicted_scores: list[float] = []
    true_scores: list[float] = []
    agreements: list[bool] = []
    context_hits: list[tuple[bool, bool]] = []
    oracle_contexts: list[bool] = []
    row_offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch_count = len(batch["context_key"])
            kwargs = {
                name: batch[name].to(device, non_blocking=True)
                for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
            }
            kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
            kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
            _prediction, recognition_logits = model(**kwargs, return_recognition=True)
            predicted_logp = torch.log_softmax(recognition_logits, dim=-1)
            target = batch["target_skeleton"].to(device, non_blocking=True)
            truth_flat = target.reshape(-1, 3, 30, 17)
            true_logp = torch.log_softmax(teacher(truth_flat), dim=-1).reshape(batch_count, -1, NUM_CLASSES)
            legal_mask = batch["legal_candidate_mask"].to(device) & batch["candidate_mask"].to(device)
            labels = batch["label_id"].to(device)
            for index in range(batch_count):
                legal = torch.nonzero(legal_mask[index], as_tuple=False).flatten()
                label = labels[index]
                pred = predicted_logp[index, legal]
                truth = true_logp[index, legal]
                predicted_scores.extend(pred[:, label].cpu().tolist())
                true_scores.extend(truth[:, label].cpu().tolist())
                agreements.extend((pred.argmax(dim=1) == truth.argmax(dim=1)).cpu().tolist())
                positive = truth.argmax(dim=1) == label
                oracle_contexts.append(bool(positive.any()))
                ranking = pred[:, label].argsort(descending=True)
                context_hits.append((bool(positive[ranking[0]]), bool(positive[ranking[: min(3, ranking.numel())]].any())))
            row_offset += batch_count
    pred_array = np.asarray(predicted_scores, dtype=np.float64)
    true_array = np.asarray(true_scores, dtype=np.float64)
    pearson, spearman = _correlation(pred_array, true_array)
    top1 = np.asarray([item[0] for item in context_hits], dtype=bool)
    top3 = np.asarray([item[1] for item in context_hits], dtype=bool)
    oracle_exists = np.asarray(oracle_contexts, dtype=bool)
    return {
        "recognition_agreement": float(np.mean(agreements)) if agreements else None,
        "pearson": pearson,
        "spearman": spearman,
        "top1_positive_hit": float(np.mean(top1)) if top1.size else None,
        "top3_positive_hit": float(np.mean(top3)) if top3.size else None,
        "oracle_positive_contexts": int(oracle_exists.sum()),
        "top1_when_oracle_positive": float(np.mean(top1[oracle_exists])) if oracle_exists.any() else None,
        "top3_when_oracle_positive": float(np.mean(top3[oracle_exists])) if oracle_exists.any() else None,
        "legal_candidate_samples": int(len(predicted_scores)),
        "contexts": int(len(context_hits)),
    }


def _train_wm(
    data_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    old_checkpoint: Path,
    new_checkpoint: Path,
) -> dict[str, Any]:
    _seed()
    model = CandidateObservationWorldModel(
        use_belief=True,
        use_rgb=True,
        residual=False,
        num_classes=NUM_CLASSES,
        use_recognition_head=True,
    ).to(device)
    old_payload = torch.load(old_checkpoint, map_location=device, weights_only=False)
    loaded = model.load_state_dict(old_payload["model_state_dict"], strict=False)
    if set(loaded.unexpected_keys) or set(loaded.missing_keys) != {
        "recognition_head.weight",
        "recognition_head.bias",
    }:
        raise ValueError(f"unexpected old WM-E checkpoint mismatch: {loaded}")
    teacher_checkpoint = data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth"
    from activeview.recognition.stgcn.model import load_checkpoint

    teacher, _ = load_checkpoint(teacher_checkpoint, NUM_CLASSES, str(device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    loader = _build_wm_loader(data_root, train_rows, WM_BATCH_SIZE, WM_WORKERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY)
    val_loader = _build_wm_loader(data_root, val_rows, WM_BATCH_SIZE, 2)
    history: list[dict[str, float]] = []
    best_key = (-np.inf, -np.inf)
    best_epoch = 0
    new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, WM_EPOCHS + 1):
        model.train()
        totals: list[tuple[float, float, float, float]] = []
        for batch in loader:
            loss, pose, recognition, ranking = _train_step(model, teacher, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append((float(loss.detach().cpu()), float(pose.detach().cpu()), float(recognition.detach().cpu()), float(ranking.detach().cpu())))
        values = np.asarray(totals, dtype=np.float64)
        stats = {
            "epoch": epoch,
            "loss": float(values[:, 0].mean()),
            "pose_loss": float(values[:, 1].mean()),
            "recognition_kl": float(values[:, 2].mean()),
            "ranking_loss": float(values[:, 3].mean()),
        }
        history.append(stats)
        val_metrics = _evaluate_wm_model(model, teacher, val_loader, val_rows, device)
        stats.update({f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, (float, int))})
        key = (
            float(val_metrics["spearman"] if val_metrics["spearman"] is not None else -1.0),
            float(val_metrics["top1_positive_hit"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dict": model.state_dict(),
                    "variant": "E_ranking_aware_recognition",
                    "epoch": epoch,
                    "seed": SEED,
                    "num_classes": NUM_CLASSES,
                    "recognition_weight": RECOGNITION_WEIGHT,
                    "ranking_weight": RANKING_WEIGHT,
                    "val_metrics": val_metrics,
                },
                new_checkpoint,
            )
        print(
            f"WM-E ranking epoch {epoch}/{WM_EPOCHS} loss={stats['loss']:.6f} "
            f"pose={stats['pose_loss']:.6f} kl={stats['recognition_kl']:.6f} rank={stats['ranking_loss']:.6f} "
            f"val_spearman={val_metrics['spearman']}",
            flush=True,
        )
    selected = torch.load(new_checkpoint, map_location=device, weights_only=False)
    return {
        "epochs": WM_EPOCHS,
        "batch_size": WM_BATCH_SIZE,
        "workers": WM_WORKERS,
        "learning_rate": WM_LR,
        "weight_decay": WM_WEIGHT_DECAY,
        "recognition_weight": RECOGNITION_WEIGHT,
        "ranking_weight": RANKING_WEIGHT,
        "max_rank_pairs": MAX_RANK_PAIRS,
        "seed": SEED,
        "train_contexts": len(train_rows),
        "train_targets": len(train_rows) * VIEW_COUNT,
        "history": history,
        "final_loss": history[-1]["loss"],
        "best_epoch": best_epoch,
        "best_val_metrics": selected.get("val_metrics", {}),
        "elapsed_seconds": time.perf_counter() - started,
        "old_checkpoint": str(old_checkpoint.resolve()),
        "new_checkpoint": {"path": str(new_checkpoint.resolve()), "sha256": _sha256(new_checkpoint)},
    }


def _evaluate_wm_val(
    data_root: Path,
    val_rows: Sequence[Mapping[str, Any]],
    val_cache_path: Path,
    wm_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache = _load_npz(val_cache_path)
    predictions = np.asarray(cache["imagined_logp"], dtype=np.float32)
    real = np.asarray(cache["true_logp"], dtype=np.float32)
    val_orders = _orders(data_root, val_rows)
    legal_scores: list[tuple[float, float, bool]] = []
    context_hits: list[tuple[bool, bool]] = []
    for index, row in enumerate(val_rows):
        label = int(row["label_id"])
        legal = [int(v) for v in val_orders[str(row["episode_id"])]]
        predicted = predictions[index, legal]
        truth = real[index, legal]
        predicted_class = predicted.argmax(axis=1)
        truth_class = truth.argmax(axis=1)
        legal_scores.extend(
            (float(predicted[pos, label]), float(truth[pos, label]), bool(predicted_class[pos] == truth_class[pos]))
            for pos in range(len(legal))
        )
        positive = truth_class == label
        ranking = np.argsort(-predicted[:, label], kind="stable")
        context_hits.append((bool(positive[ranking[0]]), bool(np.any(positive[ranking[: min(3, len(ranking))]]))))
    pred_scores = np.asarray([v[0] for v in legal_scores], dtype=np.float64)
    true_scores = np.asarray([v[1] for v in legal_scores], dtype=np.float64)
    pearson, spearman = _correlation(pred_scores, true_scores)
    exists = np.asarray([bool(np.any(real[i, val_orders[str(row["episode_id"])]].argmax(axis=1) == int(row["label_id"]))) for i, row in enumerate(val_rows)], dtype=bool)
    top1 = np.asarray([v[0] for v in context_hits], dtype=bool)
    top3 = np.asarray([v[1] for v in context_hits], dtype=bool)
    return {
        "contexts": len(val_rows),
        "legal_candidate_samples": len(legal_scores),
        "recognition_agreement": float(np.mean([v[2] for v in legal_scores])),
        "pearson": pearson,
        "spearman": spearman,
        "top1_positive_hit": float(np.mean(top1)),
        "top3_positive_hit": float(np.mean(top3)),
        "oracle_positive_contexts": int(exists.sum()),
        "top1_when_oracle_positive": float(np.mean(top1[exists])) if exists.any() else None,
        "top3_when_oracle_positive": float(np.mean(top3[exists])) if exists.any() else None,
        "source": "recognition_head logits in rebuilt cache",
        "checkpoint": str(wm_checkpoint.resolve()),
    }


def _train_pretrained_frozen_jr(
    data_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray],
    device: torch.device,
    identity_checkpoint: Path,
    wm_checkpoint: Path,
    new_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed()
    train_orders = _orders(data_root, train_rows)
    val_orders = _orders(data_root, val_rows)
    train_base, train_stats = _examples(train_rows, train_cache, train_orders)
    val_base, _ = _examples(val_rows, val_cache, val_orders)
    train_history = _history_inputs(train_rows, train_cache)
    val_history = _history_inputs(val_rows, val_cache)
    train_arrays = (*train_base, train_history)
    val_arrays = (*val_base, val_history)
    identity_payload = torch.load(identity_checkpoint, map_location=device, weights_only=False)
    identity_state = identity_payload.get("model_state_dict", identity_payload["state_dict"])
    model = PretrainedHistoryAwareJointRevision(NUM_CLASSES).to(device)
    model.history_identity.load_state_dict(identity_state)
    for parameter in model.history_identity.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=JR_LR,
        weight_decay=JR_WEIGHT_DECAY,
    )
    loader = DataLoader(_HistoryDataset(train_arrays), batch_size=JR_BATCH_SIZE, shuffle=True)
    history: list[dict[str, float]] = []
    best_key = (-np.inf, -np.inf)
    best_epoch = 0
    new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, JR_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            history_input = batch["history"].float().to(device)
            scores, posterior, identity_logits = model(current, candidates, mask, history_input)
            mask_float = mask.float()
            valid_scores = scores.masked_fill(~mask, -1e9)
            all_lse = torch.logsumexp(valid_scores, dim=1)
            positive_mask = batch["positive"].bool().to(device) & mask
            positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1)
            fallback = batch["fallback"].long().to(device)
            has_positive = positive_mask.any(dim=1)
            main = torch.where(has_positive, all_lse - positive_lse, nn.functional.cross_entropy(valid_scores, fallback, reduction="none"))
            positive = batch["positive"].float().to(device)
            bce = (nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none") * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1)
            labels = batch["label"].long().to(device)
            target = labels.unsqueeze(1).expand(-1, posterior.size(1))
            post_all = nn.functional.cross_entropy(posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none").reshape_as(scores)
            post = (post_all * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1)
            jr_loss = (main + 0.25 * bce + 0.05 * post).mean()
            identity_loss = nn.functional.cross_entropy(identity_logits, labels)
            loss = jr_loss + JR_LAMBDA_ID * identity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        actions, _identity_predictions = _select_history_actions(model, val_arrays, val_rows, val_orders, device)
        terminal = _method_metrics(actions, val_rows, val_cache)
        stats = {"epoch": epoch, "loss": float(np.mean(losses)), "val_accuracy": terminal["terminal"]["accuracy"], "val_macro_f1": terminal["terminal"]["macro_f1"]}
        history.append(stats)
        print(f"Ranking-aware PretrainedFrozenJR epoch {epoch}/{JR_EPOCHS} loss={stats['loss']:.6f} val_acc={stats['val_accuracy']:.6f}", flush=True)
        key = (float(stats["val_accuracy"]), float(stats["val_macro_f1"]))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "state_dict": model.state_dict(), "num_classes": NUM_CLASSES, "seed": SEED, "epoch": epoch, "lambda_id": JR_LAMBDA_ID, "wm_checkpoint": str(wm_checkpoint.resolve()), "val_terminal": terminal["terminal"]}, new_checkpoint)
    payload = torch.load(new_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    actions, identity_predictions = _select_history_actions(model, val_arrays, val_rows, val_orders, device)
    method = _method_metrics(actions, val_rows, val_cache)
    identity_predictions = np.asarray(identity_predictions)
    labels = np.asarray(val_base[-1], dtype=np.int64)
    identity_head = {"count": int(labels.size), "accuracy": float(np.mean(identity_predictions == labels))}
    f1s: list[float] = []
    matrix = np.bincount(labels * NUM_CLASSES + identity_predictions, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls]); precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0; recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0; f1s.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    identity_head["macro_f1"] = float(np.mean(f1s))
    return method, {"train_stats": train_stats, "epochs": JR_EPOCHS, "batch_size": JR_BATCH_SIZE, "learning_rate": JR_LR, "weight_decay": JR_WEIGHT_DECAY, "lambda_identity": JR_LAMBDA_ID, "seed": SEED, "best_epoch": best_epoch, "best_val_accuracy": best_key[0], "best_val_macro_f1": best_key[1], "final_loss": history[-1]["loss"], "history": history, "identity_head": identity_head, "checkpoint": {"path": str(new_checkpoint.resolve()), "sha256": _sha256(new_checkpoint)}}


def _sha256(path: Path) -> str:
    return jr_sha256(path)


def _load_old_results() -> tuple[dict[str, Any], dict[str, Any]]:
    wm_result_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/wm_e_diagnostics/result.json"
    jr_result_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/pretrained_history_jr/result.json"
    return json.loads(wm_result_path.read_text(encoding="utf-8")), json.loads(jr_result_path.read_text(encoding="utf-8"))


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    old_wm = result["old_wm_val"]
    new_wm = result["ranking_aware_wm_val"]
    methods = result["methods"]
    lines = [
        "# Ranking-aware Recognition WM-E (Train/Val)",
        "",
        "The WM-E pose/velocity objective was retained and augmented with a 14-way candidate recognition head, KL recognition loss (0.1), and fixed within-context pairwise logistic ranking loss (0.2). The old WM-E initialized the new run; its checkpoint was not modified.",
        "",
        f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}. Test was not read.",
        "",
        "## WM-E candidate diagnostics",
        "",
        "| Version | Agreement | Pearson | Spearman | Top-1 | Top-3 | Oracle-positive Top-1 | Oracle-positive Top-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Old WM-E | {old_wm['candidate_recognition_agreement']['overall']['agreement']:.6f} | {old_wm['true_class_probability_correlation']['overall']['pearson']:.6f} | {old_wm['true_class_probability_correlation']['overall']['spearman']:.6f} | {old_wm['candidate_ranking_positive_hit']['top1_positive_hit_rate']:.6f} | {old_wm['candidate_ranking_positive_hit']['top3_positive_hit_rate']:.6f} | {old_wm['candidate_ranking_positive_hit']['top1_positive_hit_rate_when_oracle_exists']:.6f} | {old_wm['candidate_ranking_positive_hit']['top3_positive_hit_rate_when_oracle_exists']:.6f} |",
        f"| Ranking-aware head | {new_wm['recognition_agreement']:.6f} | {new_wm['pearson']:.6f} | {new_wm['spearman']:.6f} | {new_wm['top1_positive_hit']:.6f} | {new_wm['top3_positive_hit']:.6f} | {new_wm['top1_when_oracle_positive']:.6f} | {new_wm['top3_when_oracle_positive']:.6f} |",
        "",
        "## JR comparison",
        "",
        "| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("NormalMultiPositiveJR", "PretrainedFrozenJR_old_WM_E", "PretrainedFrozenJR_ranking_aware_WM_E", "PrivilegedJR", "GTLabelPrivilegedJR", "SafeOracle"):
        item = methods[name]
        lines.append(f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |")
    old = methods["PretrainedFrozenJR_old_WM_E"]["terminal"]
    new = methods["PretrainedFrozenJR_ranking_aware_WM_E"]["terminal"]
    lines.extend([
        "",
        f"Ranking-aware WM-E minus old WM-E for Pretrained-Frozen JR: ΔAccuracy={new['accuracy'] - old['accuracy']:+.6f}, ΔMacro-F1={new['macro_f1'] - old['macro_f1']:+.6f}.",
        f"Ranking-aware WM-E candidate ranking changes: ΔSpearman={new_wm['spearman'] - old_wm['true_class_probability_correlation']['overall']['spearman']:+.6f}, ΔTop-1={new_wm['top1_positive_hit'] - old_wm['candidate_ranking_positive_hit']['top1_positive_hit_rate']:+.6f}.",
        "The new JR was retrained with the same Pretrained-Frozen history identity architecture and objective; only the imagined candidate recognition cache changed. The Val terminal metrics are checkpoint-selected on Val and should be interpreted as Train/Val evidence, not Test evidence.",
        "",
        "Leakage audit: `test_used=false`, no Test path was accessed, no old WM-E/JR/ST-GCN checkpoint was overwritten, and no future observation beyond the existing cache protocol was introduced.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    policy_root = data_root.resolve() / "datasets" / DATASET_NAME
    train_rows = load_jsonl(policy_root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    if any(str(row.get("policy_split", "")).lower() != "train" for row in train_rows):
        raise ValueError("Train rows must carry policy_split=train")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val rows must carry policy_split=val")
    old_checkpoint = data_root.resolve() / "checkpoints" / CHECKPOINT_DIR_NAME / "wm_e_last.pth"
    new_checkpoint = data_root.resolve() / "checkpoints" / CHECKPOINT_DIR_NAME / "wm_e_ranking_aware_recognition_best.pth"
    wm_training = _train_wm(data_root.resolve(), train_rows, val_rows, device, old_checkpoint, new_checkpoint)
    cache_root = policy_root / "counterfactual_cache" / "ranking_aware_wm_e"
    train_cache_path = cache_root / "train.npz"
    val_cache_path = cache_root / "val.npz"
    build_split(data_root.resolve(), "train", new_checkpoint, train_cache_path, device, 16, 2)
    build_split(data_root.resolve(), "val", new_checkpoint, val_cache_path, device, 16, 2)
    ranking_wm_val = _evaluate_wm_val(data_root.resolve(), val_rows, val_cache_path, new_checkpoint, device)
    old_wm_result, old_jr_result = _load_old_results()
    old_train_cache = _load_npz(policy_root / "counterfactual_cache/train.npz")
    old_val_cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    new_train_cache = _load_npz(train_cache_path)
    new_val_cache = _load_npz(val_cache_path)
    identity_checkpoint = data_root.resolve() / "checkpoints" / CHECKPOINT_DIR_NAME / "pretrained_history_identity_best.pth"
    new_jr_checkpoint = data_root.resolve() / "checkpoints" / CHECKPOINT_DIR_NAME / "ranking_aware_pretrained_frozen_jr_best.pth"
    new_jr_method, new_jr_training = _train_pretrained_frozen_jr(data_root.resolve(), train_rows, val_rows, new_train_cache, new_val_cache, device, identity_checkpoint, new_checkpoint, new_jr_checkpoint)
    prior_methods = old_jr_result["methods"]
    normal_method = prior_methods["NormalMultiPositiveJR"]
    old_pretrained = prior_methods["PretrainedFrozenJR"]
    methods = {
        "NormalMultiPositiveJR": normal_method,
        "PretrainedFrozenJR_old_WM_E": old_pretrained,
        "PretrainedFrozenJR_ranking_aware_WM_E": new_jr_method,
        "PrivilegedJR": prior_methods["PrivilegedJR"],
        "GTLabelPrivilegedJR": prior_methods["GTLabelPrivilegedJR"],
        "SafeOracle": prior_methods["SafeOracle"],
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_RANKING_AWARE_WM_E",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows)},
        "wm_training": wm_training,
        "ranking_aware_wm_val": ranking_wm_val,
        "old_wm_val": old_wm_result,
        "jr_training": new_jr_training,
        "methods": methods,
        "comparisons": {
            "ranking_aware_jr_minus_old_wm_jr": {"accuracy": new_jr_method["terminal"]["accuracy"] - old_pretrained["terminal"]["accuracy"], "macro_f1": new_jr_method["terminal"]["macro_f1"] - old_pretrained["terminal"]["macro_f1"]},
            "ranking_aware_wm_spearman_delta": ranking_wm_val["spearman"] - old_wm_result["true_class_probability_correlation"]["overall"]["spearman"],
            "ranking_aware_wm_top1_delta": ranking_wm_val["top1_positive_hit"] - old_wm_result["candidate_ranking_positive_hit"]["top1_positive_hit_rate"],
        },
        "protocol": {
            "wm_architecture": "existing CandidateObservationWorldModel + Linear(128,14) candidate recognition head",
            "pose_loss": "existing SmoothL1 pose + 0.25 velocity",
            "recognition_loss": "0.1 * KL(predicted_logp, frozen ST-GCN true distribution)",
            "ranking_loss": "0.2 * pairwise logistic loss on legal candidates, max 16 pairs/context",
            "wm_epochs": WM_EPOCHS,
            "jr": "Pretrained-Frozen History-aware JR, unchanged architecture/config",
            "candidate_budget": "ALL_LEGAL",
            "imagined_cache_source": "recognition_head logits for ranking-aware checkpoint",
        },
        "artifacts": {
            "train_cache": str(train_cache_path.resolve()),
            "val_cache": str(val_cache_path.resolve()),
            "old_cache_untouched": True,
            "old_wm_checkpoint_untouched": True,
            "old_jr_checkpoint_untouched": True,
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_wm_checkpoint_selection": True,
            "val_used_for_jr_checkpoint_selection": True,
            "true_candidate_logp_used_only_as_target": True,
            "formal_wm_e_overwritten": False,
            "formal_jr_overwritten": False,
            "formal_stgcn_modified": False,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Ranking-aware WM-E requires CUDA; CPU fallback is disabled")
    run(args.data_root.resolve(), device)


if __name__ == "__main__":
    main()
