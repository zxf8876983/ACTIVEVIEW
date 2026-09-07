#!/usr/bin/env python3
"""Val-only H1 disambiguation using frozen WM-E imagined observations.

The script compares the old and ranking-aware frozen WM-E checkpoints.  Each
legal H1 candidate is imagined from the archived s0 observation, classified by
the frozen ST-GCN, and scored by the frozen history-identity encoder.  The
selected candidate is then evaluated with its *real* archived observation.
No model is trained and this entry point intentionally has no Test path.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import (
    context_key,
    load_pairwise_and_azimuths,
    relative_view_descriptor,
)
from activeview.methods.joint_revision.history_aware import HistoryIdentityEncoder
from activeview.methods.world_model.model import CandidateObservationWorldModel
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced14_h1_disambiguation import (
    _candidate_beliefs,
    _classification_metrics,
    _h1_orders,
    _label_names,
    _source_map,
    _validate_rows_cache,
)
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import _load_npz
from activeview.data.preprocessing.rgb_cache import spatial_embedding_index


SEED = 42
NUM_CLASSES = 14
SKELETON_SHAPE = (3, 30, 17)
VIEW_COUNT = 32
CONTEXT_BATCH_SIZE = 32
CANDIDATE_CHUNK = 8
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/wm_imagined_h1_disambiguation"
DATASET_NAME = "policy_reduced14_kneel_eight_placement_v1"
CHECKPOINT_DIR = "activeview_reduced14_eight_placement_v1"


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


def _load_frozen_wm(path: Path, device: torch.device) -> CandidateObservationWorldModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if state is None:
        state = payload.get("state_dict")
    if state is None:
        raise ValueError(f"WM-E checkpoint has no state dict: {path}")
    model = CandidateObservationWorldModel(
        use_belief=True, use_rgb=True, residual=False, num_classes=NUM_CLASSES,
    ).to(device)
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys:
        raise ValueError(f"missing frozen WM-E keys in {path}: {loaded.missing_keys}")
    unexpected = set(loaded.unexpected_keys)
    if any(not key.startswith("recognition_head.") for key in unexpected):
        raise ValueError(f"unexpected non-recognition keys in frozen WM-E {path}: {sorted(unexpected)}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_context_inputs(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, int]]:
    """Load only s0 skeletons and legal H1 descriptors from Val archives."""
    sources = _source_map(data_root, rows)
    s0_skeletons = np.empty((len(rows), *SKELETON_SHAPE), dtype=np.float32)
    s0_descriptors = np.empty((len(rows), 9), dtype=np.float32)
    max_candidates = max((len(orders[str(row["episode_id"])]) for row in rows), default=0)
    candidate_descriptors = np.zeros((len(rows), max_candidates, 9), dtype=np.float32)
    candidate_counts: list[int] = []
    rgb_keys = 0
    for index, row in enumerate(rows):
        source = Path(sources[context_key(row)])
        with np.load(source, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        if skeleton.shape != (VIEW_COUNT, *SKELETON_SHAPE) or positions.shape != (VIEW_COUNT, 3):
            raise ValueError(f"invalid archived observation shape: {source}")
        by_id = {int(value): position for position, value in enumerate(ids.tolist())}
        s0_id = int(row["s0_viewpoint_id"])
        if s0_id not in by_id:
            raise ValueError(f"s0 viewpoint {s0_id} missing from {source}")
        current = positions[by_id[s0_id]]
        s0_skeletons[index] = skeleton[by_id[s0_id]]
        s0_descriptors[index] = relative_view_descriptor(positions, current, s0_id)
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        candidate_counts.append(len(candidates))
        for offset, candidate in enumerate(candidates):
            candidate_descriptors[index, offset] = relative_view_descriptor(positions, current, candidate)
        rgb_keys += 1
    stats = {
        "contexts": len(rows),
        "max_legal_h1_candidates": max_candidates,
        "candidate_hypotheses": int(sum(candidate_counts)),
        "s0_observations": rgb_keys,
    }
    return s0_skeletons, s0_descriptors, candidate_descriptors, candidate_counts, stats


def _load_s0_rgb(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Read only the Val s0 rows from the frozen spatial RGB mmap."""
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced14_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    manifest = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]
    index = spatial_embedding_index(manifest)
    keys = [(*context_key(row), int(row["s0_viewpoint_id"])) for row in rows]
    positions = []
    for key in keys:
        position = index.get(key)
        if position is None:
            raise ValueError(f"missing frozen s0 RGB embedding: {key}")
        positions.append(position)
    selected = np.asarray(embeddings[np.asarray(positions, dtype=np.int64)])
    if selected.shape != (len(rows), 16, 768) or selected.dtype != np.float16:
        raise ValueError(f"unexpected s0 RGB cache shape/dtype: {selected.shape}, {selected.dtype}")
    if not np.isfinite(selected.astype(np.float32)).all():
        raise ValueError("non-finite s0 RGB embedding")
    return selected


def _imagined_beliefs(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    s0_skeletons: np.ndarray,
    s0_descriptors: np.ndarray,
    candidate_descriptors: np.ndarray,
    candidate_counts: Sequence[int],
    current_logp_s0: np.ndarray,
    wm_path: Path,
    device: torch.device,
    s0_rgb: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Run frozen WM-E + ST-GCN + history identity for every H1 candidate."""
    wm = _load_frozen_wm(wm_path, device)
    stgcn_path = data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth"
    stgcn, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    identity_path = data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth"
    identity_payload = torch.load(identity_path, map_location=device, weights_only=False)
    identity = HistoryIdentityEncoder(NUM_CLASSES).to(device)
    identity_state = identity_payload.get("model_state_dict", identity_payload.get("state_dict"))
    if identity_state is None:
        raise ValueError(f"history identity checkpoint has no state dict: {identity_path}")
    identity.load_state_dict(identity_state)
    identity.eval()
    for parameter in identity.parameters():
        parameter.requires_grad_(False)

    s0_features = np.stack([
        np.asarray(row["s0_feature"], dtype=np.float32)[:256] for row in rows
    ])
    logp0 = np.asarray(current_logp_s0, dtype=np.float32)
    if logp0.shape != (len(rows), NUM_CLASSES) or not np.isfinite(logp0).all():
        raise ValueError(f"invalid frozen current_logp_s0 shape: {logp0.shape}")
    # RGB cache is the frozen EXP025 visited-observation cache. H1 has only s0;
    # the RGB history remains H=1 while the belief vector is duplicated only
    # to satisfy the frozen WM-E belief encoder's fixed 28-D input.
    if s0_rgb.shape != (len(rows), 16, 768):
        raise ValueError(f"invalid s0 RGB cache shape: {s0_rgb.shape}")
    beliefs: list[list[np.ndarray]] = [[] for _ in rows]
    started = time.perf_counter()
    with torch.inference_mode():
        for context_start in range(0, len(rows), CONTEXT_BATCH_SIZE):
            context_stop = min(context_start + CONTEXT_BATCH_SIZE, len(rows))
            batch = slice(context_start, context_stop)
            batch_size = context_stop - context_start
            history_skeleton = torch.from_numpy(s0_skeletons[batch, None]).to(device, non_blocking=True)
            history_descriptor = torch.from_numpy(s0_descriptors[batch, None]).to(device, non_blocking=True)
            duplicated_logp = np.concatenate([logp0[batch], logp0[batch]], axis=1)
            history_belief = torch.from_numpy(duplicated_logp).to(device, non_blocking=True)
            batch_rgb = s0_rgb[batch].astype(np.float32, copy=False)
            # RGB follows the actual H=1 history.  Only the belief vector is
            # duplicated because the frozen WM-E belief encoder was trained
            # with two posterior slots and has a fixed 28-D input.
            history_rgb = torch.from_numpy(batch_rgb[:, None]).to(device, non_blocking=True)
            for candidate_start in range(0, candidate_descriptors.shape[1], CANDIDATE_CHUNK):
                candidate_stop = min(candidate_start + CANDIDATE_CHUNK, candidate_descriptors.shape[1])
                descriptors = torch.from_numpy(candidate_descriptors[batch, candidate_start:candidate_stop]).to(device, non_blocking=True)
                prediction = wm(
                    history_skeleton=history_skeleton,
                    history_descriptor=history_descriptor,
                    candidate_descriptor=descriptors,
                    history_belief=history_belief,
                    history_rgb=history_rgb,
                )
                flat_prediction = prediction.reshape(-1, *SKELETON_SHAPE)
                imagined_feature = stgcn.forward_features(flat_prediction)
                imagined_logp = torch.log_softmax(stgcn.fc(imagined_feature), dim=-1)
                repeated_s0_feature = torch.from_numpy(np.repeat(s0_features[batch], candidate_stop - candidate_start, axis=0)).to(device)
                repeated_s0_logp = torch.from_numpy(np.repeat(logp0[batch], candidate_stop - candidate_start, axis=0)).to(device)
                history_input = torch.cat([
                    repeated_s0_feature,
                    imagined_feature,
                    repeated_s0_logp,
                    imagined_logp,
                ], dim=1)
                imagined_belief = torch.softmax(identity(history_input)[1], dim=-1).cpu().numpy().astype(np.float32)
                imagined_belief = imagined_belief.reshape(batch_size, candidate_stop - candidate_start, NUM_CLASSES)
                for local, global_index in enumerate(range(context_start, context_stop)):
                    valid_start = candidate_start
                    valid_stop = min(candidate_stop, int(candidate_counts[global_index]))
                    if valid_stop > valid_start:
                        beliefs[global_index].append(imagined_belief[local, : valid_stop - valid_start])
            if context_start % (CONTEXT_BATCH_SIZE * 10) == 0:
                elapsed = time.perf_counter() - started
                print(f"{wm_path.stem}: {context_stop}/{len(rows)} contexts ({elapsed:.1f}s)", flush=True)
    result = [np.concatenate(parts, axis=0) if parts else np.empty((0, NUM_CLASSES), dtype=np.float32) for parts in beliefs]
    if any(len(values) != int(count) for values, count in zip(result, candidate_counts)):
        raise RuntimeError("WM-E imagined candidate count mismatch")
    return result, {
        "checkpoint": str(wm_path.resolve()),
        "checkpoint_sha256": _sha256(wm_path),
        "contexts": len(rows),
        "candidate_hypotheses": int(sum(candidate_counts)),
        "elapsed_seconds": time.perf_counter() - started,
        "history_protocol": "s0-only H1; frozen WM-E H=2 interface receives duplicated s0 belief/RGB",
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.shape != x.shape:
        return None, None
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    pearson = None if denom <= 1e-12 else float(np.dot(xc, yc) / denom)
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        output = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and sorted_values[stop] == sorted_values[start]:
                stop += 1
            output[order[start:stop]] = (start + stop + 1) / 2.0
            start = stop
        return output
    rx, ry = ranks(x), ranks(y)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    rdenom = float(np.linalg.norm(rxc) * np.linalg.norm(ryc))
    spearman = None if rdenom <= 1e-12 else float(np.dot(rxc, ryc) / rdenom)
    return pearson, spearman


def _belief_alignment(
    imagined: Sequence[np.ndarray], real: Sequence[np.ndarray],
) -> dict[str, Any]:
    real_entropy: list[float] = []
    imagined_entropy: list[float] = []
    real_margin: list[float] = []
    imagined_margin: list[float] = []
    entropy_overlap = 0
    margin_overlap = 0
    for predicted, truth in zip(imagined, real):
        if predicted.shape != truth.shape:
            raise ValueError("imagined/real candidate belief shape mismatch")
        p_entropy = -np.sum(predicted * np.log(np.clip(predicted, 1e-12, None)), axis=1)
        t_entropy = -np.sum(truth * np.log(np.clip(truth, 1e-12, None)), axis=1)
        p_margin = np.sort(predicted, axis=1)[:, -1] - np.sort(predicted, axis=1)[:, -2]
        t_margin = np.sort(truth, axis=1)[:, -1] - np.sort(truth, axis=1)[:, -2]
        imagined_entropy.extend(p_entropy.tolist()); real_entropy.extend(t_entropy.tolist())
        imagined_margin.extend(p_margin.tolist()); real_margin.extend(t_margin.tolist())
        entropy_overlap += int(np.argmin(p_entropy) == np.argmin(t_entropy))
        margin_overlap += int(np.argmax(p_margin) == np.argmax(t_margin))
    entropy_pearson, entropy_spearman = _correlation(np.asarray(imagined_entropy), np.asarray(real_entropy))
    margin_pearson, margin_spearman = _correlation(np.asarray(imagined_margin), np.asarray(real_margin))
    contexts = len(imagined)
    return {
        "candidate_samples": len(imagined_entropy),
        "entropy": {"pearson": entropy_pearson, "spearman": entropy_spearman},
        "margin": {"pearson": margin_pearson, "spearman": margin_spearman},
        "candidate_top1_selection_overlap": {
            "min_entropy": float(entropy_overlap / contexts) if contexts else None,
            "max_margin": float(margin_overlap / contexts) if contexts else None,
            "contexts": contexts,
        },
    }


def _selector_metrics(
    name: str, selected: Sequence[int], real_beliefs: Sequence[np.ndarray],
    rows: Sequence[Mapping[str, Any]], s0_predictions: np.ndarray, names: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    selected_belief = np.stack([belief[index] for belief, index in zip(real_beliefs, selected)])
    predictions = selected_belief.argmax(axis=1).astype(np.int64)
    entropy = -np.sum(selected_belief * np.log(np.clip(selected_belief, 1e-12, None)), axis=1)
    wrong = s0_predictions != labels
    identity = _classification_metrics(predictions, labels, names)
    return {
        "selector": name,
        "identity": identity,
        "mean_entropy": float(entropy.mean()) if entropy.size else None,
        "s0_error_count": int(wrong.sum()),
        "s0_error_correction_rate": float(np.mean(predictions[wrong] == labels[wrong])) if wrong.any() else None,
        "stay_rate": 0.0,
    }


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{value:.6f}"

    lines = [
        "# Reduced14 WM-imagined H1 disambiguation (Val)", "",
        f"Val moving contexts: {result['population']['contexts']}; legal H1 candidate hypotheses: {result['population']['candidate_hypotheses']}. Test was not read.", "",
        "## Final history identity after real-observation evaluation", "",
        "| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |", "|---|---:|---:|---:|---:|",
    ]
    for name, item in result["selectors"].items():
        identity = item["identity"]
        lines.append(f"| {name} | {identity['accuracy']:.6f} | {identity['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |")
    lines.extend(["", "## Imagined-versus-real belief alignment", ""])
    for name, item in result["belief_alignment"].items():
        ent, margin, overlap = item["entropy"], item["margin"], item["candidate_top1_selection_overlap"]
        lines.append(f"- **{name}** entropy Pearson/Spearman = {fmt(ent['pearson'])}/{fmt(ent['spearman'])}; margin Pearson/Spearman = {fmt(margin['pearson'])}/{fmt(margin['spearman'])}; min-entropy/max-margin candidate overlap = {fmt(overlap['min_entropy'])}/{fmt(overlap['max_margin'])}.")
    old = result["selectors"]["Old_WM_imagined_Min_entropy_H1"]["identity"]["accuracy"]
    frozen = result["selectors"]["Frozen_current_H1"]["identity"]["accuracy"]
    ranking = result["selectors"]["Ranking_aware_WM_imagined_Min_entropy_H1"]["identity"]["accuracy"]
    lines.extend([
        "", "## Interpretation", "",
        f"Old WM-E imagined min-entropy changes Frozen H1 Accuracy by {old - frozen:+.6f}; ranking-aware WM-E changes it by {ranking - frozen:+.6f}.",
        "This is an H=1 s0-only deployment diagnostic against WM-E checkpoints trained with H=2 histories; the fixed interface shift is documented above. Imagined selectors are evaluated with the selected candidate's real archived observation, so their final identity metrics do not leak imagined labels or use candidate observations as WM inputs.",
        "If imagined selectors remain below Frozen H1 and privileged real-observation selectors, the limiting factor is WM-E action-discriminative representation fidelity; the next step should improve that objective rather than train another scalar H1 ranker.",
        "Leakage audit: `test_used=false`; no model was trained, no formal checkpoint was modified, and only Val archives/caches were read.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    started = time.perf_counter()
    data_root = data_root.resolve()
    policy_root = data_root / "datasets" / DATASET_NAME
    rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    _validate_rows_cache(rows, cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in rows):
        raise ValueError("WM-imagined H1 diagnostic requires explicit Val rows")
    names = _label_names(data_root)
    orders = _h1_orders(data_root, rows)
    real_beliefs, real_stats = _candidate_beliefs(data_root, rows, cache, orders, device)
    s0_skeletons, s0_desc, candidate_desc, counts, context_stats = _load_context_inputs(data_root, rows, orders)
    s0_rgb = _load_s0_rgb(data_root, rows)
    imagined_by_name: dict[str, list[np.ndarray]] = {}
    wm_paths = {
        "Old_WM": data_root / "checkpoints" / CHECKPOINT_DIR / "wm_e_last.pth",
        "Ranking_aware_WM": data_root / "checkpoints" / CHECKPOINT_DIR / "wm_e_ranking_aware_recognition_best.pth",
    }
    wm_stats: dict[str, Any] = {}
    for name, path in wm_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        imagined_by_name[name], wm_stats[name] = _imagined_beliefs(
            data_root, rows, s0_skeletons, s0_desc, candidate_desc, counts,
            np.asarray(cache["current_logp_s0"], dtype=np.float32), path, device, s0_rgb,
        )

    frozen_indices: list[int] = []
    real_min_indices: list[int] = []
    real_max_indices: list[int] = []
    oracle_indices: list[int] = []
    imagined_indices: dict[str, list[int]] = {key: [] for key in wm_paths}
    for row, real in zip(rows, real_beliefs):
        candidates = list(orders[str(row["episode_id"])])
        frozen_indices.append(candidates.index(int(row["s1_viewpoint_id"])))
        real_entropy = -np.sum(real * np.log(np.clip(real, 1e-12, None)), axis=1)
        real_margin = np.sort(real, axis=1)[:, -1] - np.sort(real, axis=1)[:, -2]
        real_min_indices.append(int(np.argmin(real_entropy)))
        real_max_indices.append(int(np.argmax(real_margin)))
        oracle_indices.append(int(np.argmax(real[:, int(row["label_id"])])))
        for key, values in imagined_by_name.items():
            entropy = -np.sum(values[len(imagined_indices[key])] * np.log(np.clip(values[len(imagined_indices[key])], 1e-12, None)), axis=1)
            margin = np.sort(values[len(imagined_indices[key])], axis=1)[:, -1] - np.sort(values[len(imagined_indices[key])], axis=1)[:, -2]
            imagined_indices[key].append(int(np.argmin(entropy)))
            imagined_indices.setdefault(f"{key}_margin", []).append(int(np.argmax(margin)))
    # The indexing above is per-context; retain a clear alias for readability.
    selectors_indices = {
        "Frozen_current_H1": frozen_indices,
        "Privileged_real_Min_entropy_H1": real_min_indices,
        "Privileged_real_Max_margin_H1": real_max_indices,
        "Old_WM_imagined_Min_entropy_H1": imagined_indices["Old_WM"],
        "Old_WM_imagined_Max_margin_H1": imagined_indices["Old_WM_margin"],
        "Ranking_aware_WM_imagined_Min_entropy_H1": imagined_indices["Ranking_aware_WM"],
        "Ranking_aware_WM_imagined_Max_margin_H1": imagined_indices["Ranking_aware_WM_margin"],
        "IdentityOracle_H1": oracle_indices,
    }
    s0_predictions = np.asarray(cache["current_logp_s0"], dtype=np.float32).argmax(axis=1)
    selectors = {
        name: _selector_metrics(name, indices, real_beliefs, rows, s0_predictions, names)
        for name, indices in selectors_indices.items()
    }
    alignment = {
        key: _belief_alignment(values, real_beliefs) for key, values in imagined_by_name.items()
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_WM_IMAGINED_H1_DISAMBIGUATION",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "population": {**context_stats, "val_moving_contexts": len(rows), "real_candidate_stats": real_stats},
        "selectors": selectors,
        "belief_alignment": alignment,
        "wm_inference": wm_stats,
        "artifacts": {
            "val_features": str((policy_root / "stage_d/features/val.jsonl").resolve()),
            "val_counterfactual_cache": str((policy_root / "counterfactual_cache/val.npz").resolve()),
            "old_wm": str(wm_paths["Old_WM"].resolve()),
            "ranking_aware_wm": str(wm_paths["Ranking_aware_WM"].resolve()),
            "stgcn": str((data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth").resolve()),
            "history_identity": str((data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth").resolve()),
        },
        "protocol": {
            "h1_candidates": "all legal candidate_order viewpoints from s0, excluding s0",
            "imagined_pipeline": "frozen WM-E skeleton -> frozen ST-GCN feature/logp -> frozen HistoryIdentityEncoder belief",
            "final_evaluation": "selected candidate uses real archived skeleton via frozen HistoryIdentityEncoder",
            "diagnostic_real_archive": "all legal real candidate archives are read only to build the comparison belief and final selected-candidate identity; none is a WM input",
            "history_interface_compatibility": "H=1 s0 skeleton/descriptor/RGB; current s0 belief duplicated only for frozen WM-E's fixed 28-D belief input",
            "selectors": "Frozen current, real min-entropy/max-margin, old/ranking-aware WM imagined min-entropy/max-margin, IdentityOracle",
        },
        "leakage_flags": {
            "test_used": False,
            "formal_checkpoints_modified": False,
            "models_trained": False,
            "future_candidate_observation_used_as_input": False,
            "real_candidate_archive_read_for_diagnostic_only": True,
            "ground_truth_label_used_for_selection": "IdentityOracle_H1 only",
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started, "seed": SEED},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only WM-imagined H1 disambiguation")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("WM-imagined H1 diagnostic requires CUDA; CPU fallback is disabled")
    analyze(args.data_root, device)


if __name__ == "__main__":
    main()
