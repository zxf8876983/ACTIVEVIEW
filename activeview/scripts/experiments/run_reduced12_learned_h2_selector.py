#!/usr/bin/env python3
"""Train and evaluate a minimal reduced12 learned H2 sequential selector.

The scorer is a small deployable candidate MLP.  It sees only accumulated
MeanFeature history, the current soft posterior, viewpoint IDs, decision
index and the existing candidate geometry.  Archived future observations are
used only to create Train targets and to evaluate terminal recognition/oracle
baselines.  The run reads Train/Val artifacts only and never reads Policy
Test or regenerates perception data.
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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    classification,
    lattice_distance,
)
from activeview.scripts.experiments.run_reduced12_prefix_sequential_nbv_mvp import (
    _load_all_views,
    _load_view_sequence,
)
from activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition import (
    CHUNK_SIZE,
    COORDINATE_DIM,
    FRAME_COUNT,
    HIDDEN_DIM,
    NUM_CHUNKS,
    NUM_CLASSES,
    OFFLINE_RELATIVE,
    POLICY_RELATIVE,
    RUNTIME_RELATIVE,
    ChunkEncoder,
    FeatureFusionHead,
    _candidate_options,
    _encode_chunks,
    _log_softmax,
)
from activeview.scripts.eval.audit_reduced12_sequential_horizon import (
    INFERENCE_GROUP,
    _oracle_trajectory,
    _random_trajectory,
    _truncate_trajectory,
)


SEED = 42
GEOMETRY_DIM = 11
POLICY_INPUT_DIM = HIDDEN_DIM + NUM_CLASSES + 3 + GEOMETRY_DIM
POLICY_HIDDEN_DIM = 256
EPOCHS = 12
TRAIN_BATCH_SIZE = 4096
TRAIN_SEQUENCE_BATCH = 512
EVAL_GROUP_SIZE = INFERENCE_GROUP

DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/learned_h2_selector_v1"
RUNTIME_CHECKPOINT_RELATIVE = Path(
    "checkpoints/activeview_reduced12_eight_placement_v1/learned_h2_selector_v1"
)
TRUE_LOGP_CACHE_RELATIVE = Path(
    "diagnostics/reduced12_h1_discriminative_objective_batch/train_candidate_true_logp.npz"
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_rows(policy_root: Path, stage: str, split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in load_jsonl(policy_root / stage / "features" / f"{split}.jsonl")]


def _record_path(offline_root: Path, row: Mapping[str, Any]) -> Path:
    return offline_root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"


def _load_models(data_root: Path, device: torch.device) -> tuple[ChunkEncoder, FeatureFusionHead, dict[str, str]]:
    root = data_root / RUNTIME_RELATIVE
    encoder_path = root / "chunk_encoder_best.pth"
    fusion_path = root / "feature_fusion_head_best.pth"
    if not encoder_path.exists() or not fusion_path.exists():
        raise FileNotFoundError(f"missing frozen segment checkpoints under {root}")
    encoder = ChunkEncoder().to(device)
    encoder_payload = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_payload["state_dict"])
    encoder.eval()
    fusion = FeatureFusionHead().to(device)
    fusion_payload = torch.load(fusion_path, map_location=device, weights_only=False)
    fusion.load_state_dict(fusion_payload["state_dict"])
    fusion.eval()
    for parameter in (*encoder.parameters(), *fusion.parameters()):
        parameter.requires_grad_(False)
    return encoder, fusion, {
        "chunk_encoder": str(encoder_path.resolve()),
        "feature_fusion_head": str(fusion_path.resolve()),
    }


class H2CandidateScorer(nn.Module):
    """One shared two-layer candidate scorer for both H2 decisions."""

    def __init__(self, input_dim: int = POLICY_INPUT_DIM, hidden_dim: int = POLICY_HIDDEN_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _head_arrays(fusion: FeatureFusionHead) -> tuple[np.ndarray, np.ndarray]:
    return (
        fusion.classifier.weight.detach().cpu().numpy().astype(np.float32),
        fusion.classifier.bias.detach().cpu().numpy().astype(np.float32),
    )


def _head_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ weight.T + bias


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def _candidate_geometry_map(row: Mapping[str, Any], *, second_step: bool = False) -> dict[int, np.ndarray]:
    ids_key = "remaining_candidate_ids" if second_step else "candidate_viewpoint_ids"
    geometry_key = "second_step_candidate_geometry" if second_step else "candidate_geometry"
    ids = [int(value) for value in row.get(ids_key, [])]
    values = np.asarray(row.get(geometry_key, []), dtype=np.float32)
    if values.size == 0:
        return {}
    if values.shape != (len(ids), GEOMETRY_DIM) or not np.isfinite(values).all():
        raise ValueError(f"invalid {geometry_key} shape for {row.get('episode_id')}: {values.shape}")
    return {viewpoint: values[index] for index, viewpoint in enumerate(ids)}


def _options_for_training(current: int, ids: Sequence[int]) -> list[int]:
    options = _candidate_options(int(current), ids)
    if not options:
        return [int(current)]
    return options


def _quality_targets(
    cache: Mapping[str, np.ndarray], cache_index: int, candidate_ids: Sequence[int],
    label: int, current_id: int,
) -> dict[int, float]:
    valid = np.flatnonzero(cache["candidate_mask"][cache_index])
    by_id = {
        int(cache["candidate_ids"][cache_index, index]): np.asarray(
            cache["candidate_logp"][cache_index, index], dtype=np.float32
        )
        for index in valid
    }
    if current_id not in by_id:
        by_id[current_id] = np.asarray(cache["current_logp"][cache_index], dtype=np.float32)
    current_logp = np.asarray(cache["current_logp"][cache_index], dtype=np.float32)
    output: dict[int, float] = {int(current_id): 0.0}
    for viewpoint in candidate_ids:
        value = by_id.get(int(viewpoint))
        if value is None:
            raise ValueError(f"candidate {viewpoint} missing from true-logp cache row {cache_index}")
        output[int(viewpoint)] = float(value[int(label)] - current_logp[int(label)])
    return output


def _encode_sequences(
    encoder: ChunkEncoder,
    sequences: Sequence[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(sequences).astype(np.float32)
    return _encode_chunks(encoder, values, device, sequence_batch_size=TRAIN_SEQUENCE_BATCH)


def _state_input(
    history_features: np.ndarray,
    current_view: int,
    decision: int,
    candidate_view: int,
    geometry: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
) -> np.ndarray:
    mean_feature = np.asarray(history_features, dtype=np.float32).mean(axis=0)
    current_logits = _head_logits(mean_feature[None], head_weight, head_bias)[0]
    posterior = _softmax(current_logits[None])[0]
    values = np.concatenate(
        [
            mean_feature,
            posterior,
            np.asarray(
                [float(current_view) / 31.0, float(decision), float(candidate_view) / 31.0],
                dtype=np.float32,
            ),
            np.asarray(geometry, dtype=np.float32),
        ]
    ).astype(np.float32)
    if values.shape != (POLICY_INPUT_DIM,) or not np.isfinite(values).all():
        raise ValueError(f"invalid policy input shape/value: {values.shape}")
    return values


def _build_training_examples(
    policy_root: Path,
    offline_root: Path,
    encoder: ChunkEncoder,
    fusion: FeatureFusionHead,
    device: torch.device,
    true_cache: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    stage_c_rows = _read_rows(policy_root, "stage_c", "train")
    stage_d_rows = _read_rows(policy_root, "stage_d", "train")
    stage_c_index = {str(row["episode_id"]): index for index, row in enumerate(stage_c_rows)}
    if len(stage_c_rows) != int(true_cache["labels"].shape[0]):
        raise ValueError("Stage-C Train and true-logp cache counts differ")
    head_weight, head_bias = _head_arrays(fusion)
    d0_counts = [len(_options_for_training(int(row["current_viewpoint_id"]), row["candidate_viewpoint_ids"])) for row in stage_c_rows]
    d1_counts: list[int] = []
    for row in stage_d_rows:
        remaining = [int(value) for value in row.get("remaining_candidate_ids", [])]
        legal = [value for value in remaining if lattice_distance(int(row["s1_viewpoint_id"]), value) == 1]
        d1_counts.append(1 + len(set(legal)))
    total = int(sum(d0_counts) + sum(d1_counts))
    inputs = np.empty((total, POLICY_INPUT_DIM), dtype=np.float32)
    targets = np.empty(total, dtype=np.float32)
    cursor = 0

    for start in range(0, len(stage_c_rows), TRAIN_SEQUENCE_BATCH):
        batch_rows = stage_c_rows[start : start + TRAIN_SEQUENCE_BATCH]
        sequences = [
            _load_view_sequence(_record_path(offline_root, row), int(row["current_viewpoint_id"]))
            for row in batch_rows
        ]
        _unused_logits, features = _encode_sequences(encoder, sequences, device)
        for local, row in enumerate(batch_rows):
            cache_index = start + local
            current = int(row["current_viewpoint_id"])
            options = _options_for_training(current, row["candidate_viewpoint_ids"])
            geometry = _candidate_geometry_map(row)
            quality = _quality_targets(true_cache, cache_index, options, int(row["label_id"]), current)
            for candidate in options:
                inputs[cursor] = _state_input(
                    features[local, 0:1], current, 0, candidate,
                    geometry.get(candidate, np.zeros(GEOMETRY_DIM, dtype=np.float32)),
                    head_weight, head_bias,
                )
                targets[cursor] = quality[candidate]
                cursor += 1
        print(f"[learned-h2] built decision-0 examples {min(start + len(batch_rows), len(stage_c_rows))}/{len(stage_c_rows)}", flush=True)

    for start in range(0, len(stage_d_rows), TRAIN_SEQUENCE_BATCH):
        batch_rows = stage_d_rows[start : start + TRAIN_SEQUENCE_BATCH]
        sequences: list[np.ndarray] = []
        for row in batch_rows:
            stage_c = stage_c_rows[stage_c_index[str(row["episode_id"])] ]
            path = _record_path(offline_root, stage_c)
            sequences.extend([
                _load_view_sequence(path, int(row["s0_viewpoint_id"])),
                _load_view_sequence(path, int(row["s1_viewpoint_id"])),
            ])
        _pair_logits, pair_features = _encode_sequences(encoder, sequences, device)
        pair_features = pair_features.reshape(len(batch_rows), 2, NUM_CHUNKS, HIDDEN_DIM)
        for local, row in enumerate(batch_rows):
            stage_index = stage_c_index[str(row["episode_id"])]
            current = int(row["s1_viewpoint_id"])
            remaining = [
                int(value) for value in row.get("remaining_candidate_ids", [])
                if lattice_distance(current, int(value)) == 1
            ]
            options = [current, *sorted(set(remaining))]
            geometry = _candidate_geometry_map(row, second_step=True)
            stage_c = stage_c_rows[stage_index]
            quality = _quality_targets(true_cache, stage_index, options, int(row["label_id"]), current)
            history_features = np.stack([
                pair_features[local, 0, 0], pair_features[local, 1, 1],
            ])
            for candidate in options:
                fallback_geometry = _candidate_geometry_map(stage_c).get(candidate, np.zeros(GEOMETRY_DIM, dtype=np.float32))
                inputs[cursor] = _state_input(
                    history_features, current, 1, candidate,
                    geometry.get(candidate, fallback_geometry), head_weight, head_bias,
                )
                targets[cursor] = quality[candidate]
                cursor += 1
        print(f"[learned-h2] built decision-1 examples {min(start + len(batch_rows), len(stage_d_rows))}/{len(stage_d_rows)}", flush=True)
    if cursor != total:
        raise RuntimeError(f"training example count mismatch: {cursor} != {total}")
    summary = {
        "stage_c_train_contexts": len(stage_c_rows),
        "stage_d_train_contexts": len(stage_d_rows),
        "decision_0_examples": int(sum(d0_counts)),
        "decision_1_examples": int(sum(d1_counts)),
        "total_examples": total,
    }
    return inputs, targets, summary


def _train_scorer(
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    checkpoint: Path,
    output_summary: Path,
) -> tuple[H2CandidateScorer, dict[str, Any], np.ndarray, np.ndarray]:
    _seed()
    mean = raw_inputs.mean(axis=0).astype(np.float32)
    std = raw_inputs.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = ((raw_inputs - mean) / std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(normalized), torch.from_numpy(targets.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), pin_memory=True)
    model = H2CandidateScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for values, target in loader:
            prediction = model(values.to(device, non_blocking=True))
            loss = nn.functional.smooth_l1_loss(prediction, target.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(f"[learned-h2] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_mean": mean,
        "input_std": std,
        "input_dim": POLICY_INPUT_DIM,
        "hidden_dim": POLICY_HIDDEN_DIM,
        "target": "archived frozen ST-GCN candidate true-class log-probability gain over current view",
        "epochs": EPOCHS,
        "seed": SEED,
    }, checkpoint)
    summary = {
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "input_dim": POLICY_INPUT_DIM,
        "hidden_dim": POLICY_HIDDEN_DIM,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "loss": "SmoothL1(predicted candidate quality, archived true-class log-probability gain)",
        "checkpoint": str(checkpoint.resolve()),
        "seed": SEED,
        "test_used": False,
        "history": history,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return model, summary, mean, std


def _load_scorer(checkpoint: Path, device: torch.device) -> tuple[H2CandidateScorer, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = H2CandidateScorer(input_dim=int(payload["input_dim"]), hidden_dim=int(payload["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["input_mean"], dtype=np.float32), np.asarray(payload["input_std"], dtype=np.float32)


def _learned_trajectory(
    contexts: Sequence[Mapping[str, Any]],
    view_features: np.ndarray,
    stage_c_rows: Mapping[str, Mapping[str, Any]],
    scorer: H2CandidateScorer,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    trajectory = np.zeros((len(contexts), NUM_CHUNKS), dtype=np.int64)
    trajectory[:, 0] = np.asarray([int(item["start_view"]) for item in contexts], dtype=np.int64)
    for decision in (0, 1):
        for index, context in enumerate(contexts):
            current = int(trajectory[index, decision])
            options = _candidate_options(current, context["candidate_ids"])
            stage_c = stage_c_rows[str(context["episode_id"])]
            geometry = _candidate_geometry_map(stage_c)
            history_features = np.stack([
                view_features[index, int(trajectory[index, chunk]), chunk]
                for chunk in range(decision + 1)
            ])
            raw_inputs = np.stack([
                _state_input(
                    history_features, current, decision, candidate,
                    geometry.get(candidate, np.zeros(GEOMETRY_DIM, dtype=np.float32)), head_weight, head_bias,
                ) for candidate in options
            ]).astype(np.float32)
            normalized = torch.from_numpy((raw_inputs - input_mean) / input_std).to(device)
            with torch.inference_mode():
                scores = scorer(normalized).detach().cpu().numpy()
            choice = max(range(len(options)), key=lambda item: (float(scores[item]), -int(options[item])))
            trajectory[index, decision + 1] = int(options[choice])
    trajectory[:, 3:] = trajectory[:, 2, None]
    return trajectory


def _terminal_predictions(
    view_features: np.ndarray,
    trajectory: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
) -> np.ndarray:
    rows = np.arange(trajectory.shape[0], dtype=np.int64)[:, None]
    chunks = np.arange(NUM_CHUNKS, dtype=np.int64)[None, :]
    selected = view_features[rows, trajectory, chunks]
    logits = _head_logits(selected.mean(axis=1), head_weight, head_bias)
    return np.argmax(logits, axis=1).astype(np.int64)


def _evaluate_with_trajectories(
    contexts: Sequence[Mapping[str, Any]],
    policy_root: Path,
    offline_root: Path,
    encoder: ChunkEncoder,
    fusion: FeatureFusionHead,
    scorer: H2CandidateScorer,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[list[int]]]]:
    """Evaluate and aggregate metrics without retaining large view tensors."""
    stage_c = {str(row["episode_id"]): row for row in _read_rows(policy_root, "stage_c", "val")}
    head_weight, head_bias = _head_arrays(fusion)
    names = ("Stay", "Random-H2", "Learned-H2", "Privileged-Oracle-H2", "Privileged-Oracle-Full")
    labels_all = np.asarray([int(item["label"]) for item in contexts], dtype=np.int64)
    prediction_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    entropy_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    gt_prob_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    move_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    trajectories_out: dict[str, list[list[int]]] = {name: [] for name in names}
    rng = np.random.default_rng(SEED)
    for start in range(0, len(contexts), EVAL_GROUP_SIZE):
        group = contexts[start : start + EVAL_GROUP_SIZE]
        labels = np.asarray([int(item["label"]) for item in group], dtype=np.int64)
        all_views = np.stack([_load_all_views(_record_path(offline_root, item["row"])) for item in group]).astype(np.float32)
        size = len(group)
        raw_logits, raw_features = _encode_chunks(encoder, all_views.reshape(size * 32, FRAME_COUNT, COORDINATE_DIM), device)
        view_logits = raw_logits.reshape(size, 32, NUM_CHUNKS, NUM_CLASSES)
        view_features = raw_features.reshape(size, 32, NUM_CHUNKS, HIDDEN_DIM)
        view_logp = _log_softmax(view_logits)
        starts = np.asarray([int(item["start_view"]) for item in group], dtype=np.int64)
        oracle_full = _oracle_trajectory(group, view_logp, view_features, "MeanFeature", head_weight, head_bias, fusion, device)
        trajectories = {
            "Stay": np.repeat(starts[:, None], NUM_CHUNKS, axis=1),
            "Random-H2": _truncate_trajectory(_random_trajectory(group, rng), 2),
            "Learned-H2": _learned_trajectory(group, view_features, stage_c, scorer, input_mean, input_std, head_weight, head_bias, device),
            "Privileged-Oracle-H2": _truncate_trajectory(oracle_full, 2),
            "Privileged-Oracle-Full": oracle_full,
        }
        for name, trajectory in trajectories.items():
            rows = np.arange(size, dtype=np.int64)[:, None]
            chunks = np.arange(NUM_CHUNKS, dtype=np.int64)[None, :]
            selected = view_features[rows, trajectory, chunks]
            logits = _head_logits(selected.mean(axis=1), head_weight, head_bias)
            probabilities = _softmax(logits)
            prediction_parts[name].append(np.argmax(logits, axis=1).astype(np.int64))
            entropy_parts[name].append(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1))
            gt_prob_parts[name].append(probabilities[np.arange(size), labels])
            decision_count = 2 if name != "Privileged-Oracle-Full" else NUM_CHUNKS - 1
            moves = np.sum(trajectory[:, 1 : decision_count + 1] != trajectory[:, :decision_count], axis=1)
            move_parts[name].append(moves.astype(np.float32) / float(decision_count))
            if len(trajectories_out[name]) < 16:
                trajectories_out[name].extend(trajectory[: 16 - len(trajectories_out[name])].tolist())
        print(f"[learned-h2] evaluated Val Moving {min(start + size, len(contexts))}/{len(contexts)}", flush=True)
    metrics: dict[str, Any] = {}
    for name in names:
        predictions = np.concatenate(prediction_parts[name])
        item = classification(labels_all.tolist(), predictions.tolist())
        item.update({
            "mean_entropy": float(np.mean(np.concatenate(entropy_parts[name]))),
            "mean_gt_probability": float(np.mean(np.concatenate(gt_prob_parts[name]))),
            "move_rate": float(np.mean(np.concatenate(move_parts[name]))),
            "stay_rate": float(1.0 - np.mean(np.concatenate(move_parts[name]))),
            "horizon_decisions": 2 if name != "Privileged-Oracle-Full" else NUM_CHUNKS - 1,
            "observed_frames": FRAME_COUNT,
            "discrete_time_view_switch_approximation": True,
            "raw_cross_view_skeleton_stitching_used": False,
            "terminal_recognition": "frozen chunk encoder MeanFeature + FeatureFusionHead",
        })
        metrics[name] = item
    return metrics, trajectories_out


def _analysis(result: Mapping[str, Any]) -> str:
    metrics = result["metrics_moving"]
    learned = metrics["Learned-H2"]
    random = metrics["Random-H2"]
    oracle_h2 = metrics["Privileged-Oracle-H2"]
    oracle_full = metrics["Privileged-Oracle-Full"]
    lines = [
        "# Reduced12 Learned H2 Sequential Selector",
        "",
        f"Moving Val contexts: {result['data_summary']['moving_val_contexts']}. The policy makes decisions only after chunks 0 and 1 (5 and 10 observed frames), then holds the final viewpoint through t30. This is a discrete-time view-switch approximation, not continuous robot motion.",
        "",
        "## Policy",
        "",
        f"Input dimension is {result['policy']['input_dim']}: accumulated MeanFeature (256), current soft posterior (12), current-view/decision/candidate IDs (3 normalized scalars), and existing candidate geometry (11). The scorer is a shared 256-hidden GELU MLP with a scalar output.",
        f"Training target is archived frozen-ST-GCN candidate true-class log-probability gain over the current view; future observations are targets only. Train examples: {result['training']['total_examples']}.",
        "",
        "## Moving Val t30",
        "",
        "| Method | Accuracy | Macro-F1 | Move rate | Contexts |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} | {item['n']} |")
    lines += [
        "",
        f"Learned-H2 − Random-H2 = {(learned['accuracy'] - random['accuracy']) * 100.0:+.3f}pp Accuracy / {(learned['macro_f1'] - random['macro_f1']) * 100.0:+.3f}pp Macro-F1.",
        f"Privileged Oracle-H2 − Learned-H2 = {(oracle_h2['accuracy'] - learned['accuracy']) * 100.0:+.3f}pp Accuracy / {(oracle_h2['macro_f1'] - learned['macro_f1']) * 100.0:+.3f}pp Macro-F1.",
        f"Privileged Oracle-Full is {oracle_full['accuracy']:.6f}/{oracle_full['macro_f1']:.6f}; the Full oracle is retained only as a ceiling reference.",
        "",
        "## Leakage sanity",
        "",
        "The scorer inference tensor is assembled only from history features/posterior, viewpoint IDs, decision index and candidate geometry. It never receives GT labels, future candidate skeletons, future features/logits, or hard predicted actions. Policy Test was not read.",
        "",
        "## Answers",
        "",
        f"1. Learned-H2 {'exceeds' if learned['accuracy'] > random['accuracy'] else 'does not exceed'} Random-H2 by {(learned['accuracy'] - random['accuracy']) * 100.0:+.3f}pp Accuracy.",
        f"2. The remaining H2 headroom to the privileged H2 oracle is {(oracle_h2['accuracy'] - learned['accuracy']) * 100.0:.3f}pp; this is the direct selector gap under the same frozen MeanFeature terminal recognizer.",
        "3. The policy is intentionally a minimal regression scorer rather than RL, MCTS, beam search or a new recognizer. Further policy work should depend on whether this Val-only gap is scientifically meaningful.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "future_candidate_observation_in_policy_input=false",
        "future_candidate_feature_or_logits_in_policy_input=false",
        "gt_action_in_policy_input=false",
        "recognizer_modified=false",
        "new_data_generated=false",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    started = time.time()
    policy_root = data_root / POLICY_RELATIVE
    offline_root = data_root / OFFLINE_RELATIVE
    contexts, context_summary = _load_val_contexts(policy_root)
    encoder, fusion, frozen_checkpoints = _load_models(data_root, device)
    true_cache_path = data_root / TRUE_LOGP_CACHE_RELATIVE
    if not true_cache_path.exists():
        raise FileNotFoundError(f"missing Train true-logp cache: {true_cache_path}")
    with np.load(true_cache_path, allow_pickle=False) as archive:
        true_cache = {key: np.asarray(archive[key]) for key in archive.files}
    raw_inputs, targets, train_summary = _build_training_examples(policy_root, offline_root, encoder, fusion, device, true_cache)
    runtime_checkpoint = data_root / RUNTIME_CHECKPOINT_RELATIVE / "h2_candidate_scorer_final.pth"
    scorer, training_summary, input_mean, input_std = _train_scorer(
        raw_inputs, targets, device, runtime_checkpoint, output_dir / "training_summary.json",
    )
    metrics_moving, trajectories = _evaluate_with_trajectories(
        contexts, policy_root, offline_root, encoder, fusion, scorer,
        input_mean, input_std, device,
    )
    data_summary = {
        **context_summary,
        "moving_val_contexts": len(contexts),
        "train_candidate_quality_cache": str(true_cache_path.resolve()),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_LEARNED_H2_SELECTOR_V1",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.time() - started,
        },
        "data_summary": data_summary,
        "labels": list(LABELS),
        "policy": {
            "input_dim": POLICY_INPUT_DIM,
            "hidden_dim": POLICY_HIDDEN_DIM,
            "architecture": "Linear(282,256) -> GELU -> Linear(256,1)",
            "shared_across_decisions": True,
            "decisions": "after chunk0 and chunk1 only; hold final viewpoint to t30",
            "legal_actions": "Stay or current-view lattice distance <= 1 legal neighbor",
            "candidate_geometry_dim": GEOMETRY_DIM,
            "candidate_quality_target": "archived frozen ST-GCN true-class log-probability gain over current view",
        },
        "training": {**train_summary, **training_summary},
        "frozen_checkpoints": frozen_checkpoints,
        "metrics_moving": metrics_moving,
        "trajectory_examples": trajectories,
        "policy_test_used": False,
        "future_candidate_observation_in_policy_input": False,
        "future_candidate_feature_or_logits_in_policy_input": False,
        "gt_action_in_policy_input": False,
        "recognizer_modified": False,
        "new_data_generated": False,
        "training_used_train_only": True,
        "gt_future_observation_used_for_target_only": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_dir / "leakage_sanity.json").write_text(json.dumps({
        "gt_label_in_policy_input": False,
        "future_candidate_observation_in_policy_input": False,
        "future_candidate_feature_or_logits_in_policy_input": False,
        "hard_predicted_action_in_policy_input": False,
        "policy_test_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trajectory_examples.json").write_text(json.dumps(trajectories, indent=2) + "\n", encoding="utf-8")
    return result


def _load_val_contexts(policy_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stage_c_val = _read_rows(policy_root, "stage_c", "val")
    stage_d_val = _read_rows(policy_root, "stage_d", "val")
    stage_c_by_episode = {str(row["episode_id"]): row for row in stage_c_val}
    contexts: list[dict[str, Any]] = []
    for row in stage_d_val:
        episode = str(row["episode_id"])
        stage_c = stage_c_by_episode.get(episode)
        if stage_c is None:
            raise ValueError(f"missing Stage-C Val row for {episode}")
        start = int(row["s0_viewpoint_id"])
        if start != int(stage_c["current_viewpoint_id"]):
            raise ValueError(f"Stage-C/Stage-D start mismatch for {episode}")
        contexts.append({
            "episode_id": episode,
            "row": row,
            "label": int(row["label_id"]),
            "start_view": start,
            "candidate_ids": sorted(int(value) for value in stage_c["candidate_viewpoint_ids"]),
        })
    return contexts, {
        "stage_c_val_contexts": len(stage_c_val),
        "stage_d_val_contexts": len(stage_d_val),
        "stage_d_val_moving_contexts": len(contexts),
        "train_split_used": "stage_c/features/train.jsonl + stage_d/features/train.jsonl",
        "val_split_used": "stage_d/features/val.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_root.resolve(), _cuda(args.device))
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["data_summary"]["moving_val_contexts"],
        "learned_accuracy": result["metrics_moving"]["Learned-H2"]["accuracy"],
        "random_accuracy": result["metrics_moving"]["Random-H2"]["accuracy"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
