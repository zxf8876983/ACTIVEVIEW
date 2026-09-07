#!/usr/bin/env python3
"""Train and evaluate Action-Discriminative WM-E on reduced14 Train/Val.

The old WM-E pose/velocity objective is retained.  Optional candidate heads
predict the frozen ST-GCN feature and posterior, while a frozen History
Identity encoder supplies the belief-consistency target.  Test is deliberately
not reachable from this entry point.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import (
    candidate_order,
    context_key,
    load_pairwise_and_azimuths,
    relative_view_descriptor,
)
from activeview.methods.joint_revision.history_aware import HistoryIdentityEncoder
from activeview.methods.world_model.model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
    world_model_loss,
)
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced14_h1_disambiguation import (
    _candidate_beliefs,
    _h1_orders,
    _label_names,
    _selector_metrics,
    _source_map,
)
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import _load_npz
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import _validate_rows_cache
from activeview.scripts.eval.analyze_reduced14_wm_e import _correlation
from activeview.scripts.train.train_reduced14_world_model import _sources


SEED = 42
NUM_CLASSES = 14
VIEW_COUNT = 32
WM_EPOCHS = 12
WM_BATCH_SIZE = 64
WM_WORKERS = 4
WM_LR = 1e-3
WM_WEIGHT_DECAY = 1e-4
REC_WEIGHT = 0.1
FEATURE_WEIGHT = 0.1
BELIEF_WEIGHT = 0.2
CONTEXT_BATCH_SIZE = 32
CANDIDATE_CHUNK = 8
DATASET_NAME = "policy_reduced14_kneel_eight_placement_v1"
CHECKPOINT_DIR = "activeview_reduced14_eight_placement_v1"
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/action_discriminative_wm_e"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _filtered_rgb_lookup(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], np.ndarray]:
    """Load only visited Train/Val RGB embeddings, never the Test rows.

    The shared RGB manifest has no split field and is interleaved.  We derive
    the exact required keys from the supplied Train/Val rows, then materialize
    only those embedding rows from the memmap.  Test RGB values are never
    indexed or loaded.
    """
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced14_eight_placement/initial_history"
    needed = {
        (*context_key(row), int(viewpoint))
        for row in rows
        for viewpoint in (int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]))
    }
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    lookup: dict[tuple[str, str, str, int], np.ndarray] = {}
    with (cache / "manifest.jsonl").open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            metadata = json.loads(line)
            key = (
                str(metadata["scene_id"]),
                str(metadata["region"]),
                str(metadata["record_id"]),
                int(metadata["viewpoint_id"]),
            )
            if key in needed:
                lookup[key] = np.asarray(embeddings[position])
                if len(lookup) == len(needed):
                    break
    missing = needed.difference(lookup)
    if missing:
        raise ValueError(f"missing Train/Val RGB embeddings: {sorted(missing)[:3]}")
    return lookup


def _load_history_identity(data_root: Path, device: torch.device) -> tuple[HistoryIdentityEncoder, Path]:
    path = data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth"
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"invalid history identity checkpoint: {path}")
    model = HistoryIdentityEncoder(NUM_CLASSES).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def _build_loader(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray],
    batch_size: int,
    workers: int,
    shuffle: bool,
    candidate_context_lookup: Mapping[tuple[str, ...], np.ndarray] | None = None,
) -> DataLoader:
    sources = _sources(data_root, [dict(row) for row in rows])
    pairwise, azimuths = load_pairwise_and_azimuths(
        data_root, rows, sources,
        pair_root=data_root / "datasets" / DATASET_NAME / "pairwise_viewpoint_geodesic",
    )
    legal_by_context: dict[tuple[str, str, str], tuple[int, ...]] = {}
    for row in rows:
        scene_region = (str(row["scene_id"]), str(row["region"]))
        current = int(row["s1_viewpoint_id"])
        legal_by_context[context_key(row)] = tuple(candidate_order(
            row, current, {int(row["s0_viewpoint_id"]), current},
            pairwise[scene_region], azimuths[scene_region],
        ))
    dataset = LazyWorldModelContextDataset(
        rows, sources, use_belief=True, rgb_lookup=rgb_lookup,
        target_scope="all", legal_candidate_ids=legal_by_context, cache_size=64,
        candidate_context_lookup=candidate_context_lookup,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=collate_world_model_context, pin_memory=True,
        persistent_workers=workers > 0,
    )


def _load_model_from_old(old_checkpoint: Path, device: torch.device) -> CandidateObservationWorldModel:
    payload = torch.load(old_checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"invalid old WM-E checkpoint: {old_checkpoint}")
    model = CandidateObservationWorldModel(
        use_belief=True, use_rgb=True, residual=False, num_classes=NUM_CLASSES,
        use_action_discriminative_heads=True,
    ).to(device)
    loaded = model.load_state_dict(state, strict=False)
    expected = {
        "action_feature_head.weight", "action_feature_head.bias",
        "action_logit_head.weight", "action_logit_head.bias",
    }
    if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
        raise ValueError(f"unexpected old WM-E state mismatch: {loaded}")
    return model


def _belief_inputs(
    s0_feature: torch.Tensor,
    candidate_feature: torch.Tensor,
    s0_logp: torch.Tensor,
    candidate_logp: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([s0_feature, candidate_feature, s0_logp, candidate_logp], dim=-1)


def _train_step(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    identity: HistoryIdentityEncoder,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    kwargs = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
    }
    kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
    kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    valid = batch["candidate_mask"].to(device).reshape(-1)
    prediction, predicted_feature, predicted_logits = model(
        **kwargs, return_action_discriminative=True,
    )
    predicted_flat = prediction.reshape(-1, 3, 30, 17)[valid]
    target_flat = target.reshape(-1, 3, 30, 17)[valid]
    pose_total, pose, velocity = world_model_loss(predicted_flat, target_flat)
    with torch.no_grad():
        true_feature = teacher.forward_features(target_flat)
        true_logp = torch.log_softmax(teacher.fc(true_feature), dim=-1)
        s0_feature = teacher.forward_features(kwargs["history_skeleton"][:, 0])
        s0_logp = torch.log_softmax(teacher.fc(s0_feature), dim=-1)
    pred_feature = predicted_feature.reshape(-1, 256)[valid]
    pred_logp = torch.log_softmax(predicted_logits.reshape(-1, NUM_CLASSES)[valid], dim=-1)
    rec = F.kl_div(pred_logp, true_logp.exp(), reduction="batchmean")
    feat = (1.0 - F.cosine_similarity(pred_feature, true_feature, dim=-1)).mean()
    repeat = int(pred_feature.shape[0] // s0_feature.shape[0])
    real_input = _belief_inputs(
        s0_feature.repeat_interleave(repeat, dim=0), true_feature,
        s0_logp.repeat_interleave(repeat, dim=0), true_logp,
    )
    pred_input = _belief_inputs(
        s0_feature.repeat_interleave(repeat, dim=0), pred_feature,
        s0_logp.repeat_interleave(repeat, dim=0), pred_logp,
    )
    with torch.no_grad():
        real_belief = torch.softmax(identity(real_input)[1], dim=-1)
    belief = F.kl_div(torch.log_softmax(identity(pred_input)[1], dim=-1), real_belief, reduction="batchmean")
    loss = pose_total + REC_WEIGHT * rec + FEATURE_WEIGHT * feat + BELIEF_WEIGHT * belief
    return loss, {
        "loss": float(loss.detach().cpu()), "pose_loss": float(pose.detach().cpu()),
        "velocity_loss": float(velocity.detach().cpu()), "recognition_kl": float(rec.detach().cpu()),
        "feature_cosine_loss": float(feat.detach().cpu()), "belief_kl": float(belief.detach().cpu()),
    }


def _evaluate_wm(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    identity: HistoryIdentityEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    agreement_head: list[bool] = []
    agreement_skeleton: list[bool] = []
    true_scores: list[float] = []
    pred_scores: list[float] = []
    cosine_values: list[float] = []
    belief_kls: list[float] = []
    pred_entropies: list[float] = []
    true_entropies: list[float] = []
    pred_margins: list[float] = []
    true_margins: list[float] = []
    top_hits: list[tuple[bool, bool]] = []
    oracle_exists: list[bool] = []
    with torch.inference_mode():
        for batch in loader:
            kwargs = {name: batch[name].to(device, non_blocking=True) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor", "history_belief", "history_rgb")}
            prediction, predicted_feature, predicted_logits = model(**kwargs, return_action_discriminative=True)
            target = batch["target_skeleton"].to(device, non_blocking=True)
            batch_size, candidate_count = prediction.shape[:2]
            valid = batch["candidate_mask"].to(device) & batch["legal_candidate_mask"].to(device)
            target_flat = target.reshape(-1, 3, 30, 17)
            pred_flat = prediction.reshape(-1, 3, 30, 17)
            true_feature = teacher.forward_features(target_flat)
            true_logp = torch.log_softmax(teacher.fc(true_feature), dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            skeleton_logp = torch.log_softmax(teacher(pred_flat), dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            pred_logp = torch.log_softmax(predicted_logits, dim=-1)
            s0_feature = teacher.forward_features(kwargs["history_skeleton"][:, 0])
            s0_logp = torch.log_softmax(teacher.fc(s0_feature), dim=-1)
            repeated_s0_feature = s0_feature[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 256)
            repeated_s0_logp = s0_logp[:, None, :].expand(-1, candidate_count, -1).reshape(-1, NUM_CLASSES)
            pred_belief = torch.softmax(identity(_belief_inputs(repeated_s0_feature, predicted_feature.reshape(-1, 256), repeated_s0_logp, pred_logp.reshape(-1, NUM_CLASSES)))[1], dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            real_belief = torch.softmax(identity(_belief_inputs(repeated_s0_feature, true_feature, repeated_s0_logp, true_logp.reshape(-1, NUM_CLASSES)))[1], dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            labels = batch["label_id"].to(device)
            for i in range(batch_size):
                mask = valid[i]
                if not bool(mask.any()):
                    continue
                p = pred_logp[i, mask]; t = true_logp[i, mask]; sk = skeleton_logp[i, mask]
                label = int(labels[i])
                agreement_head.extend((p.argmax(dim=-1) == t.argmax(dim=-1)).cpu().tolist())
                agreement_skeleton.extend((sk.argmax(dim=-1) == t.argmax(dim=-1)).cpu().tolist())
                pred_scores.extend(p[:, label].cpu().tolist()); true_scores.extend(t[:, label].cpu().tolist())
                cosine_values.extend(F.cosine_similarity(predicted_feature[i, mask], true_feature.reshape(batch_size, candidate_count, 256)[i, mask], dim=-1).cpu().tolist())
                rb, pb = real_belief[i, mask], pred_belief[i, mask]
                belief_kls.extend((rb * (rb.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(dim=-1).cpu().tolist())
                p_ent = -(pb * pb.clamp_min(1e-12).log()).sum(dim=-1).cpu().numpy()
                t_ent = -(rb * rb.clamp_min(1e-12).log()).sum(dim=-1).cpu().numpy()
                p_mar = (torch.sort(pb, dim=-1).values[:, -1] - torch.sort(pb, dim=-1).values[:, -2]).cpu().numpy()
                t_mar = (torch.sort(rb, dim=-1).values[:, -1] - torch.sort(rb, dim=-1).values[:, -2]).cpu().numpy()
                pred_entropies.extend(p_ent.tolist()); true_entropies.extend(t_ent.tolist()); pred_margins.extend(p_mar.tolist()); true_margins.extend(t_mar.tolist())
                positive = t.argmax(dim=-1).cpu().numpy() == label
                ranking = np.argsort(-p[:, label].cpu().numpy(), kind="stable")
                top_hits.append((bool(positive[ranking[0]]), bool(np.any(positive[ranking[: min(3, len(ranking))]]))))
                oracle_exists.append(bool(positive.any()))
    pearson, spearman = _correlation(np.asarray(pred_scores), np.asarray(true_scores))
    ent_pearson, ent_spearman = _correlation(np.asarray(pred_entropies), np.asarray(true_entropies))
    mar_pearson, mar_spearman = _correlation(np.asarray(pred_margins), np.asarray(true_margins))
    top1 = np.asarray([v[0] for v in top_hits], dtype=bool); top3 = np.asarray([v[1] for v in top_hits], dtype=bool); exists = np.asarray(oracle_exists, dtype=bool)
    return {
        "legal_candidate_samples": len(pred_scores),
        "recognition_agreement": float(np.mean(agreement_head)),
        "skeleton_recognition_agreement": float(np.mean(agreement_skeleton)),
        "pearson": pearson, "spearman": spearman,
        "feature_cosine_similarity": float(np.mean(cosine_values)),
        "belief_kl": float(np.mean(belief_kls)),
        "belief_entropy_pearson": ent_pearson, "belief_entropy_spearman": ent_spearman,
        "belief_margin_pearson": mar_pearson, "belief_margin_spearman": mar_spearman,
        "top1_positive_hit": float(np.mean(top1)), "top3_positive_hit": float(np.mean(top3)),
        "oracle_positive_contexts": int(exists.sum()),
        "top1_when_oracle_positive": float(np.mean(top1[exists])) if exists.any() else None,
        "top3_when_oracle_positive": float(np.mean(top3[exists])) if exists.any() else None,
        "contexts": int(len(top_hits)),
    }


def _load_context_inputs(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]],
    candidate_context_lookup: Mapping[tuple[str, ...], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    sources = _source_map(data_root, rows)
    s0_skeletons = np.empty((len(rows), 3, 30, 17), dtype=np.float32)
    s0_descriptors = np.empty((len(rows), 9), dtype=np.float32)
    max_count = max((len(orders[str(row["episode_id"])]) for row in rows), default=0)
    context_dim = 0
    if candidate_context_lookup:
        first_context = np.asarray(next(iter(candidate_context_lookup.values())), dtype=np.float32)
        if first_context.ndim != 1:
            raise ValueError(f"candidate context descriptors must be 1-D, got {first_context.shape}")
        context_dim = int(first_context.size)
    descriptors = np.zeros((len(rows), max_count, 9 + context_dim), dtype=np.float32)
    counts: list[int] = []
    for index, row in enumerate(rows):
        with np.load(sources[context_key(row)], allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64); skeleton = np.asarray(archive["skeleton"], dtype=np.float32); positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        by_id = {int(value): pos for pos, value in enumerate(ids.tolist())}; s0_id = int(row["s0_viewpoint_id"])
        current = positions[by_id[s0_id]]; s0_skeletons[index] = skeleton[by_id[s0_id]]; s0_descriptors[index] = relative_view_descriptor(positions, current, s0_id)
        candidates = [int(v) for v in orders[str(row["episode_id"])]]; counts.append(len(candidates))
        for offset, candidate in enumerate(candidates):
            descriptor = relative_view_descriptor(positions, current, candidate)
            if candidate_context_lookup is not None:
                try:
                    key = (*context_key(row), int(candidate))
                    extra_value = candidate_context_lookup.get(key)
                    if extra_value is None:
                        extra_value = candidate_context_lookup[(str(row["scene_id"]), str(row["region"]), int(candidate))]
                    extra = np.asarray(extra_value, dtype=np.float32)
                except KeyError as exc:
                    raise ValueError(f"missing candidate context descriptor {exc}") from exc
                if extra.shape != (context_dim,):
                    raise ValueError(f"invalid candidate context descriptor shape: {extra.shape}")
                descriptor = np.concatenate([descriptor, extra], axis=0)
            descriptors[index, offset] = descriptor
    return s0_skeletons, s0_descriptors, descriptors, counts


def _load_s0_rgb(
    rows: Sequence[Mapping[str, Any]],
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray],
) -> np.ndarray:
    result = np.asarray([
        rgb_lookup[(*context_key(row), int(row["s0_viewpoint_id"]))]
        for row in rows
    ])
    if result.shape != (len(rows), 16, 768) or result.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"invalid frozen RGB cache shape/dtype: {result.shape}, {result.dtype}")
    return result.astype(np.float32, copy=False)


def _imagined_h1_beliefs(
    model: CandidateObservationWorldModel,
    identity: HistoryIdentityEncoder,
    rows: Sequence[Mapping[str, Any]],
    s0_skeletons: np.ndarray,
    s0_descriptors: np.ndarray,
    candidate_descriptors: np.ndarray,
    candidate_counts: Sequence[int],
    logp0: np.ndarray,
    s0_rgb: np.ndarray,
    device: torch.device,
    ground_truth_labels: np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    model.eval(); beliefs: list[list[np.ndarray]] = [[] for _ in rows]; started = time.perf_counter()
    s0_features = np.stack([np.asarray(row["s0_feature"], dtype=np.float32)[:256] for row in rows])
    with torch.inference_mode():
        for start in range(0, len(rows), CONTEXT_BATCH_SIZE):
            stop = min(start + CONTEXT_BATCH_SIZE, len(rows)); batch_size = stop - start; sl = slice(start, stop)
            history_skeleton = torch.from_numpy(s0_skeletons[sl, None]).to(device)
            history_descriptor = torch.from_numpy(s0_descriptors[sl, None]).to(device)
            history_belief = torch.from_numpy(np.concatenate([logp0[sl], logp0[sl]], axis=1)).to(device)
            history_rgb = torch.from_numpy(s0_rgb[sl].astype(np.float32, copy=False)[:, None]).to(device)
            gt_action = None
            if ground_truth_labels is not None:
                labels = np.asarray(ground_truth_labels[sl], dtype=np.int64)
                if labels.shape != (batch_size,) or np.any((labels < 0) | (labels >= NUM_CLASSES)):
                    raise ValueError("ground_truth_labels must contain valid reduced14 class IDs")
                gt_action = F.one_hot(torch.from_numpy(labels), NUM_CLASSES).to(device=device, dtype=torch.float32)
            for cstart in range(0, candidate_descriptors.shape[1], CANDIDATE_CHUNK):
                cstop = min(cstart + CANDIDATE_CHUNK, candidate_descriptors.shape[1]); c = cstop - cstart
                descriptor = torch.from_numpy(candidate_descriptors[sl, cstart:cstop]).to(device)
                model_kwargs: dict[str, Any] = {
                    "history_belief": history_belief,
                    "history_rgb": history_rgb,
                }
                if gt_action is not None:
                    model_kwargs["ground_truth_action"] = gt_action
                _, predicted_feature, predicted_logits = model(
                    history_skeleton,
                    history_descriptor,
                    descriptor,
                    **model_kwargs,
                    return_action_discriminative=True,
                )
                pred_logp = torch.log_softmax(predicted_logits, dim=-1).reshape(-1, NUM_CLASSES)
                feature = predicted_feature.reshape(-1, 256)
                s0_feature = torch.from_numpy(np.repeat(s0_features[sl], c, axis=0)).to(device)
                s0_logp = torch.from_numpy(np.repeat(logp0[sl], c, axis=0)).to(device)
                result = torch.softmax(identity(_belief_inputs(s0_feature, feature, s0_logp, pred_logp))[1], dim=-1).cpu().numpy().reshape(batch_size, c, NUM_CLASSES)
                for local, global_index in enumerate(range(start, stop)):
                    valid_stop = min(cstop, int(candidate_counts[global_index]))
                    if valid_stop > cstart: beliefs[global_index].append(result[local, : valid_stop - cstart])
            if start % (CONTEXT_BATCH_SIZE * 10) == 0: print(f"H1 action-discriminative WM: {stop}/{len(rows)} contexts ({time.perf_counter() - started:.1f}s)", flush=True)
    output = [np.concatenate(parts, axis=0) if parts else np.empty((0, NUM_CLASSES), dtype=np.float32) for parts in beliefs]
    if any(len(values) != int(count) for values, count in zip(output, candidate_counts)): raise RuntimeError("action WM H1 candidate count mismatch")
    return output, {"contexts": len(rows), "candidate_hypotheses": int(sum(candidate_counts)), "elapsed_seconds": time.perf_counter() - started}


def _run_h1(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray],
    model: CandidateObservationWorldModel,
    device: torch.device,
    names: Sequence[str],
    candidate_context_lookup: Mapping[tuple[str, ...], np.ndarray] | None = None,
    ground_truth_labels: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    orders = _h1_orders(data_root, rows)
    real_beliefs, real_stats = _candidate_beliefs(data_root, rows, cache, orders, device)
    s0_skeletons, s0_desc, candidate_desc, counts = _load_context_inputs(
        data_root, rows, orders, candidate_context_lookup
    )
    s0_rgb = _load_s0_rgb(rows, rgb_lookup)
    identity, _ = _load_history_identity(data_root, device)
    imagined, inference = _imagined_h1_beliefs(
        model,
        identity,
        rows,
        s0_skeletons,
        s0_desc,
        candidate_desc,
        counts,
        np.asarray(cache["current_logp_s0"], dtype=np.float32),
        s0_rgb,
        device,
        ground_truth_labels=ground_truth_labels,
    )
    frozen: list[int] = []; real_min: list[int] = []; real_max: list[int] = []; action_min: list[int] = []; action_max: list[int] = []; oracle: list[int] = []
    for row, real, pred in zip(rows, real_beliefs, imagined):
        candidates = list(orders[str(row["episode_id"])]); frozen.append(candidates.index(int(row["s1_viewpoint_id"])))
        re = -np.sum(real * np.log(np.clip(real, 1e-12, None)), axis=1); rm = np.sort(real, axis=1)[:, -1] - np.sort(real, axis=1)[:, -2]
        pe = -np.sum(pred * np.log(np.clip(pred, 1e-12, None)), axis=1); pm = np.sort(pred, axis=1)[:, -1] - np.sort(pred, axis=1)[:, -2]
        real_min.append(int(np.argmin(re))); real_max.append(int(np.argmax(rm))); action_min.append(int(np.argmin(pe))); action_max.append(int(np.argmax(pm))); oracle.append(int(np.argmax(real[:, int(row["label_id"])])))
    s0_predictions = np.asarray(cache["current_logp_s0"], dtype=np.float32).argmax(axis=1)
    selectors = {
        "Frozen_current_H1": _selector_metrics("Frozen_current_H1", frozen, real_beliefs, rows, s0_predictions, names),
        "Real_Min_entropy_H1": _selector_metrics("Real_Min_entropy_H1", real_min, real_beliefs, rows, s0_predictions, names),
        "Real_Max_margin_H1": _selector_metrics("Real_Max_margin_H1", real_max, real_beliefs, rows, s0_predictions, names),
        "Action_Discriminative_WM_imagined_Min_entropy_H1": _selector_metrics("Action_Discriminative_WM_imagined_Min_entropy_H1", action_min, real_beliefs, rows, s0_predictions, names),
        "Action_Discriminative_WM_imagined_Max_margin_H1": _selector_metrics("Action_Discriminative_WM_imagined_Max_margin_H1", action_max, real_beliefs, rows, s0_predictions, names),
        "IdentityOracle_H1": _selector_metrics("IdentityOracle_H1", oracle, real_beliefs, rows, s0_predictions, names),
    }
    alignment = {"Action_Discriminative_WM": _alignment(imagined, real_beliefs)}
    return {"selectors": selectors, "belief_alignment": alignment, "inference": inference, "real_candidate_stats": real_stats}, {"orders": orders}


def _alignment(imagined: Sequence[np.ndarray], real: Sequence[np.ndarray]) -> dict[str, Any]:
    ie: list[float] = []; re: list[float] = []; im: list[float] = []; rm: list[float] = []; entropy_overlap = 0; margin_overlap = 0
    for pred, truth in zip(imagined, real):
        pe = -np.sum(pred * np.log(np.clip(pred, 1e-12, None)), axis=1); te = -np.sum(truth * np.log(np.clip(truth, 1e-12, None)), axis=1)
        pm = np.sort(pred, axis=1)[:, -1] - np.sort(pred, axis=1)[:, -2]; tm = np.sort(truth, axis=1)[:, -1] - np.sort(truth, axis=1)[:, -2]
        ie.extend(pe.tolist()); re.extend(te.tolist()); im.extend(pm.tolist()); rm.extend(tm.tolist()); entropy_overlap += int(np.argmin(pe) == np.argmin(te)); margin_overlap += int(np.argmax(pm) == np.argmax(tm))
    ep, es = _correlation(np.asarray(ie), np.asarray(re)); mp, ms = _correlation(np.asarray(im), np.asarray(rm)); n = len(imagined)
    return {"candidate_samples": len(ie), "entropy": {"pearson": ep, "spearman": es}, "margin": {"pearson": mp, "spearman": ms}, "candidate_top1_selection_overlap": {"min_entropy": entropy_overlap / n if n else None, "max_margin": margin_overlap / n if n else None, "contexts": n}}


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    h1 = result["h1"]["selectors"]
    lines = ["# Action-Discriminative WM-E (reduced14 Val)", "", f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}. Test was not read.", "", "## WM-E Val diagnostics", "", "| Metric | Value |", "|---|---:|"]
    for key in ("recognition_agreement", "skeleton_recognition_agreement", "pearson", "spearman", "feature_cosine_similarity", "belief_kl", "belief_entropy_pearson", "belief_entropy_spearman", "belief_margin_pearson", "belief_margin_spearman", "top1_positive_hit", "top3_positive_hit"):
        lines.append(f"| {key} | {result['wm_val'][key]} |")
    old_diagnostics = result.get("baseline_comparison", {}).get("old_wm_diagnostics")
    ranking_diagnostics = result.get("baseline_comparison", {}).get("ranking_aware_wm_diagnostics")
    if old_diagnostics or ranking_diagnostics:
        lines.extend(["", "## Candidate diagnostic comparison", "", "| WM-E | Agreement | Pearson | Spearman | Top-1 | Top-3 |", "|---|---:|---:|---:|---:|---:|"])
        for name, diagnostics in (("Old WM-E", old_diagnostics), ("Ranking-aware WM-E", ranking_diagnostics), ("Action-discriminative WM-E", result["wm_val"])):
            if not diagnostics:
                continue
            if "candidate_recognition_agreement" in diagnostics:
                agreement = diagnostics["candidate_recognition_agreement"]["overall"]["agreement"]
                pearson = diagnostics["true_class_probability_correlation"]["overall"]["pearson"]
                spearman = diagnostics["true_class_probability_correlation"]["overall"]["spearman"]
                top1 = diagnostics["candidate_ranking_positive_hit"]["top1_positive_hit_rate"]
                top3 = diagnostics["candidate_ranking_positive_hit"]["top3_positive_hit_rate"]
            else:
                agreement, pearson, spearman = diagnostics["recognition_agreement"], diagnostics["pearson"], diagnostics["spearman"]
                top1, top3 = diagnostics["top1_positive_hit"], diagnostics["top3_positive_hit"]
            lines.append(f"| {name} | {agreement:.6f} | {pearson:.6f} | {spearman:.6f} | {top1:.6f} | {top3:.6f} |")
    lines.extend(["", "## H1 history identity after real selected observation", "", "| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |", "|---|---:|---:|---:|---:|"])
    for name, item in h1.items():
        lines.append(f"| {name} | {item['identity']['accuracy']:.6f} | {item['identity']['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |")
    lines.extend(["", "## Interpretation", "", f"Action-discriminative WM-E imagined min-entropy Accuracy is {h1['Action_Discriminative_WM_imagined_Min_entropy_H1']['identity']['accuracy']:.6f}; the max-margin value is {h1['Action_Discriminative_WM_imagined_Max_margin_H1']['identity']['accuracy']:.6f}.", "The action heads directly supervise frozen ST-GCN feature/posterior targets and add frozen History Identity belief consistency; old WM-E pose/velocity supervision remains unchanged.", "If feature/belief fidelity improves while H1 remains below Frozen H1, candidate-specific scene/occlusion context—not another selector loss—should be the next representation change.", "H1 uses s0-only history against WM-E checkpoints trained with H=2 histories; the fixed interface distribution shift is recorded in result.json.", "Leakage audit: `test_used=false`; no Test data were read, ST-GCN/History Identity were frozen, and the old WM-E checkpoint was not overwritten."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed(); started = time.perf_counter(); data_root = data_root.resolve(); policy_root = data_root / "datasets" / DATASET_NAME
    train_rows = load_jsonl(policy_root / "stage_d/features/train.jsonl"); val_rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    if any(str(row.get("policy_split", "")).lower() != "train" for row in train_rows): raise ValueError("Train split contamination")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows): raise ValueError("Val split contamination")
    rgb_lookup = _filtered_rgb_lookup(data_root, [*train_rows, *val_rows])
    train_loader = _build_loader(data_root, train_rows, rgb_lookup, WM_BATCH_SIZE, WM_WORKERS, True)
    val_loader = _build_loader(data_root, val_rows, rgb_lookup, WM_BATCH_SIZE, 2, False)
    old_checkpoint = data_root / "checkpoints" / CHECKPOINT_DIR / "wm_e_last.pth"
    new_checkpoint = data_root / "checkpoints" / CHECKPOINT_DIR / "wm_e_action_discriminative_best.pth"
    new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model_from_old(old_checkpoint, device)
    teacher_checkpoint = data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth"
    teacher, _ = load_checkpoint(teacher_checkpoint, NUM_CLASSES, str(device))
    identity, identity_checkpoint = _load_history_identity(data_root, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY)
    val_history: list[dict[str, Any]] = []; train_history: list[dict[str, float]] = []; best_key = (-np.inf, -np.inf); best_epoch = 0; started_train = time.perf_counter()
    for epoch in range(1, WM_EPOCHS + 1):
        model.train(); batches: list[dict[str, float]] = []
        for batch in train_loader:
            loss, stats = _train_step(model, teacher, identity, batch, device); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); batches.append(stats)
        train_stats = {key: float(np.mean([item[key] for item in batches])) for key in batches[0]}; train_stats["epoch"] = epoch; train_history.append(train_stats)
        val_stats = _evaluate_wm(model, teacher, identity, val_loader, device); val_stats["epoch"] = epoch; val_history.append(val_stats)
        key = (float(val_stats["belief_entropy_pearson"] or -1.0), float(val_stats["belief_margin_spearman"] or -1.0))
        if key > best_key:
            best_key = key; best_epoch = epoch; torch.save({"model_state_dict": model.state_dict(), "state_dict": model.state_dict(), "variant": "E_action_discriminative", "epoch": epoch, "seed": SEED, "num_classes": NUM_CLASSES, "loss_weights": {"recognition": REC_WEIGHT, "feature": FEATURE_WEIGHT, "belief": BELIEF_WEIGHT}, "val_metrics": val_stats}, new_checkpoint)
        print(f"Action WM epoch {epoch}/{WM_EPOCHS} loss={train_stats['loss']:.6f} val_entropy_r={val_stats['belief_entropy_pearson']}", flush=True)
    payload = torch.load(new_checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"]); model.eval()
    val_cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    _validate_rows_cache(val_rows, val_cache, "val")
    if not np.array_equal(
        np.asarray(val_cache["label_id"], dtype=np.int64),
        np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64),
    ):
        raise ValueError("Val cache labels are not aligned with Val feature rows")
    names = _label_names(data_root)
    h1_result, _ = _run_h1(data_root, val_rows, val_cache, rgb_lookup, model, device, names)
    prior_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/wm_imagined_h1_disambiguation/result.json"
    prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
    prior_selectors = prior.get("selectors", {})
    for name in ("Old_WM_imagined_Min_entropy_H1", "Old_WM_imagined_Max_margin_H1", "Ranking_aware_WM_imagined_Min_entropy_H1", "Ranking_aware_WM_imagined_Max_margin_H1"):
        if name in prior_selectors: h1_result["selectors"][name] = prior_selectors[name]
    final_wm_val = val_history[best_epoch - 1]
    old_diagnostics_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/wm_e_diagnostics/result.json"
    ranking_diagnostics_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/ranking_aware_wm_e/result.json"
    old_diagnostics = json.loads(old_diagnostics_path.read_text()) if old_diagnostics_path.is_file() else None
    ranking_diagnostics = json.loads(ranking_diagnostics_path.read_text()).get("ranking_aware_wm_val") if ranking_diagnostics_path.is_file() else None
    result: dict[str, Any] = {"experiment_id": "REDUCED14_ACTION_DISCRIMINATIVE_WM_E", "status": "COMPLETED", "split": "train_val", "test_used": False, "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows), "val_candidate_hypotheses": h1_result["inference"]["candidate_hypotheses"]}, "training": {"epochs": WM_EPOCHS, "batch_size": WM_BATCH_SIZE, "workers": WM_WORKERS, "learning_rate": WM_LR, "seed": SEED, "train_history": train_history, "val_history": val_history, "best_epoch": best_epoch, "final_train_loss": train_history[-1]["loss"], "elapsed_seconds": time.perf_counter() - started_train, "checkpoint": str(new_checkpoint.resolve()), "checkpoint_sha256": _sha256(new_checkpoint)}, "wm_val": final_wm_val, "h1": h1_result, "baseline_comparison": {"previous_wm_imagined_h1": prior_selectors, "old_wm_diagnostics": old_diagnostics, "ranking_aware_wm_diagnostics": ranking_diagnostics}, "protocol": {"loss": "L_pose + 0.1*KL(true ST-GCN distribution || predicted logits) + 0.1*(1-cosine(predicted_feature,true_feature)) + 0.2*KL(real HistoryIdentity belief || predicted HistoryIdentity belief)", "predicted_feature": "256-D candidate-conditioned decoder latent head", "predicted_logits": "14-D candidate-conditioned recognition head", "teacher": "frozen ST-GCN", "history_identity": "frozen pretrained History Identity", "candidate_budget": "ALL_LEGAL", "h1_history": "s0-only; RGB/skeleton/descriptor H=1, s0 belief duplicated only for frozen 28-D belief input"}, "artifacts": {"old_wm_checkpoint": str(old_checkpoint.resolve()), "old_wm_untouched": True, "new_wm_checkpoint": str(new_checkpoint.resolve()), "stgcn_checkpoint": str(teacher_checkpoint.resolve()), "history_identity_checkpoint": str(identity_checkpoint.resolve())}, "leakage_flags": {"test_used": False, "models_trained": True, "formal_stgcn_modified": False, "formal_history_identity_modified": False, "old_wm_overwritten": False, "future_candidate_observation_used_as_wm_input": False, "real_candidate_archive_used_as_training_target": True, "real_candidate_archive_read_for_diagnostic_only": False, "rgb_lookup_scope": "Train/Val visited s0/s1 keys only"}, "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started}}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); _write_analysis(OUTPUT_DIR / "analysis.md", result); (OUTPUT_DIR / "training.json").write_text(json.dumps(result["training"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, ensure_ascii=False)); return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("Action-discriminative WM-E requires CUDA; CPU fallback is disabled")
    run(args.data_root.resolve(), device)


if __name__ == "__main__": main()
