#!/usr/bin/env python3
"""Audit the effect of temporal segment length on fixed-view recognition.

The frozen five-frame chunk encoder from the segment-aware pipeline is reused
for L=5, 10, and 15 segments.  Segment features are concatenated in temporal
order and classified by a small MLP.  L=30 is a positive control evaluated by
the frozen full-sequence reduced12 ST-GCN checkpoint.  Only Stage-C Train and
Stage-D Moving Val archives are read; no policy Test or perception generation
is involved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification
from activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition import (
    BATCH_SIZE,
    HIDDEN_DIM,
    NUM_CLASSES,
    ChunkEncoder,
    FeatureFusionHead,
    _encode_chunks,
    _load_contexts,
    _load_train_sequences,
    _load_val_sequences,
    _load_view_sequence,
    _record_path,
)
from activeview.recognition.stgcn.model import load_checkpoint


SEED = 42
FRAME_COUNT = 30
JOINT_COUNT = 17
COORDINATE_DIM = JOINT_COUNT * 3
SEGMENT_LENGTHS = (5, 10, 15)
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INFERENCE_BATCH_SIZE = 2048
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/chunk_length_representation_audit"
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
OFFLINE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")
STGCN_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
RUNTIME_RELATIVE = Path(
    "checkpoints/activeview_reduced12_eight_placement_v1/chunk_length_representation_audit"
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


def _load_frozen_segment_models(
    data_root: Path, device: torch.device,
) -> tuple[ChunkEncoder, FeatureFusionHead, dict[str, str]]:
    runtime_root = data_root / (
        "checkpoints/activeview_reduced12_eight_placement_v1/"
        "segment_aware_sequential_recognition"
    )
    encoder_checkpoint = runtime_root / "chunk_encoder_best.pth"
    fusion_checkpoint = runtime_root / "feature_fusion_head_best.pth"
    if not encoder_checkpoint.exists() or not fusion_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing frozen segment checkpoints: {encoder_checkpoint}, {fusion_checkpoint}"
        )
    encoder = ChunkEncoder().to(device)
    encoder_payload = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_payload["state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    fusion_head = FeatureFusionHead().to(device)
    fusion_payload = torch.load(fusion_checkpoint, map_location=device, weights_only=False)
    fusion_head.load_state_dict(fusion_payload["state_dict"])
    fusion_head.eval()
    for parameter in fusion_head.parameters():
        parameter.requires_grad_(False)
    return encoder, fusion_head, {
        "chunk_encoder": str(encoder_checkpoint.resolve()),
        "mean_feature_head": str(fusion_checkpoint.resolve()),
    }


class OrderedSegmentClassifier(nn.Module):
    """MLP over chronologically concatenated segment features."""

    def __init__(self, segment_count: int) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.input_dim = self.segment_count * HIDDEN_DIM
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _segment_features(
    encoder: ChunkEncoder,
    sequences: np.ndarray,
    segment_length: int,
    device: torch.device,
) -> np.ndarray:
    if segment_length not in SEGMENT_LENGTHS:
        raise ValueError(f"unsupported segment length {segment_length}")
    if sequences.ndim != 3 or sequences.shape[1:] != (FRAME_COUNT, COORDINATE_DIM):
        raise ValueError(f"unexpected sequence shape {sequences.shape}")
    count = FRAME_COUNT // segment_length
    feature_parts: list[np.ndarray] = []
    encoder.eval()
    for segment in range(count):
        start = segment * segment_length
        end = start + segment_length
        values = sequences[:, start:end].reshape(-1, segment_length, COORDINATE_DIM)
        parts: list[np.ndarray] = []
        for batch_start in range(0, len(values), INFERENCE_BATCH_SIZE):
            batch = torch.from_numpy(values[batch_start : batch_start + INFERENCE_BATCH_SIZE]).to(
                device, non_blocking=True
            )
            with torch.inference_mode():
                _, features = encoder(batch)
            parts.append(features.cpu().numpy())
        feature_parts.append(np.concatenate(parts, axis=0))
    return np.stack(feature_parts, axis=1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)


def _metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=1)
    probabilities = _softmax(logits)
    result = classification(labels.tolist(), predictions.tolist())
    result.update({
        "mean_entropy": float(np.mean(_entropy(probabilities))),
        "mean_gt_probability": float(np.mean(probabilities[np.arange(len(labels)), labels])),
    })
    return result


def _batched_logits(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(values), INFERENCE_BATCH_SIZE):
        batch = torch.from_numpy(values[start : start + INFERENCE_BATCH_SIZE]).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _train_length_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    segment_length: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    _seed()
    segment_count = FRAME_COUNT // segment_length
    train_values = train_features.reshape(len(train_features), -1).astype(np.float32)
    val_values = val_features.reshape(len(val_features), -1).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(train_values), torch.from_numpy(train_labels))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=True,
    )
    model = OrderedSegmentClassifier(segment_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for values, labels in loader:
            logits = model(values.to(device, non_blocking=True))
            loss = nn.functional.cross_entropy(logits, labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = _batched_logits(model, val_values, device)
        val_metrics = _metrics(val_logits, val_labels)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(record)
        print(
            f"[length-{segment_length}] epoch={epoch:02d}/{EPOCHS} "
            f"loss={record['train_loss']:.6f} "
            f"val_acc={record['val_accuracy']:.6f} val_f1={record['val_macro_f1']:.6f}",
            flush=True,
        )
        if float(val_metrics["macro_f1"]) > best_score + 1e-8:
            best_score = float(val_metrics["macro_f1"])
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "segment_length": segment_length,
                "segment_count": segment_count,
                "input_dim": segment_count * HIDDEN_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
                "seed": SEED,
                "epoch": epoch,
            }, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    final_logits = _batched_logits(model, val_values, device)
    final_metrics = _metrics(final_logits, val_labels)
    summary: dict[str, Any] = {
        "segment_length": segment_length,
        "segment_count": segment_count,
        "input_dim": segment_count * HIDDEN_DIM,
        "architecture": f"Linear({segment_count * HIDDEN_DIM},256) -> GELU -> Linear(256,12)",
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "final_selected_metrics": final_metrics,
        "train_contexts": int(len(train_features)),
        "val_moving_contexts": int(len(val_features)),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint": str(checkpoint.resolve()),
        "seed": SEED,
        "test_used": False,
        "history": history,
    }
    return summary, final_logits


def _full_stgcn_metrics(
    sequences: np.ndarray,
    labels: np.ndarray,
    data_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    checkpoint = data_root / STGCN_RELATIVE
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, loaded_device = load_checkpoint(checkpoint, NUM_CLASSES, str(device))
    if loaded_device.type != "cuda":
        raise RuntimeError("full-sequence ST-GCN unexpectedly loaded on CPU")
    skeleton = sequences.reshape(len(sequences), FRAME_COUNT, JOINT_COUNT, 3)
    skeleton = np.transpose(skeleton, (0, 3, 1, 2)).astype(np.float32)
    logits_parts: list[np.ndarray] = []
    for start in range(0, len(skeleton), INFERENCE_BATCH_SIZE):
        batch = torch.from_numpy(skeleton[start : start + INFERENCE_BATCH_SIZE]).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            logits_parts.append(model(batch).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0).astype(np.float32)
    return _metrics(logits, labels), str(checkpoint.resolve())


def _load_val_sequences_for_field(
    contexts: list[dict[str, Any]], offline_root: Path, viewpoint_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = [
        _load_view_sequence(
            _record_path(offline_root, context["row"]),
            int(context["row"][viewpoint_field]),
        )
        for context in contexts
    ]
    labels = np.asarray([int(context["label"]) for context in contexts], dtype=np.int64)
    return np.stack(sequences).astype(np.float32), labels


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _analysis(result: Mapping[str, Any]) -> str:
    lengths = result["length_metrics"]
    lines = [
        "# Reduced12 Chunk-Length / Segment-Representation Audit",
        "",
        "This is a fixed-view Stay-only audit on the same Stage-C Train and Stage-D Moving Val protocol. L=5/10/15 use the frozen shared segment encoder and ordered feature concatenation followed by a small MLP. L=30 is evaluated on fixed Stay s0, while the existing FrozenStageCv0-selected s1 is reported separately as the positive-control reference. Policy Test is not read and no perception data is regenerated.",
        "",
        "## Moving Val results",
        "",
        "| Length | Segments | Accuracy | Macro-F1 | Representation |",
        "|---:|---:|---:|---:|---|",
    ]
    for length in (5, 10, 15, 30):
        item = lengths[f"L{length}"]
        lines.append(
            f"| {length} | {item['segment_count']} | {item['metrics']['accuracy']:.6f} | "
            f"{item['metrics']['macro_f1']:.6f} | {item['representation']} |"
        )
    l5 = lengths["L5"]["metrics"]
    l10 = lengths["L10"]["metrics"]
    l15 = lengths["L15"]["metrics"]
    l30 = lengths["L30"]["metrics"]
    positive_control = result["full_sequence_positive_control"]
    selected_s1 = positive_control["frozen_stagecv0_selected_s1"]
    lines += [
        "",
        f"The existing MeanFeature L=5 reference is Acc={result['mean_feature_reference']['accuracy']:.6f}, Macro-F1={result['mean_feature_reference']['macro_f1']:.6f}; it is reproduced from the frozen MeanFeature head without retraining the recognizer.",
        f"OrderedConcat L=5→10 changes accuracy by {(l10['accuracy'] - l5['accuracy']) * 100:+.2f}pp; L=10→15 changes it by {(l15['accuracy'] - l10['accuracy']) * 100:+.2f}pp.",
        f"The L=30 frozen ST-GCN on fixed Stay s0 is Acc={l30['accuracy']:.6f}, Macro-F1={l30['macro_f1']:.6f}. The same frozen ST-GCN on the existing FrozenStageCv0-selected s1 is Acc={selected_s1['accuracy']:.6f}, Macro-F1={selected_s1['macro_f1']:.6f}; the recorded reference is Acc≈0.454266 / Macro-F1≈0.444782.",
        "",
        "## Interpretation",
        "",
    ]
    if abs(selected_s1["accuracy"] - 0.454266) <= 0.005:
        lines.append("The FrozenStageCv0-selected s1 positive control reproduces the recorded full-sequence reference within 0.5pp. Fixed Stay s0 is a different viewpoint protocol and therefore has the lower 0.254266 ceiling.")
    else:
        lines.append("Even the FrozenStageCv0-selected s1 positive control does not reproduce the recorded reference within 0.5pp; a protocol/checkpoint mismatch remains and segment-length conclusions should be treated as provisional.")
    if l10["accuracy"] >= l5["accuracy"] + 0.02:
        lines.append("L=10 is at least 2pp above L=5, indicating that a longer sensing window materially recovers recognition.")
    else:
        lines.append("L=10 is not at least 2pp above L=5, so a longer isolated segment does not yet show a large recovery.")
    if l15["accuracy"] >= l10["accuracy"] + 0.02:
        lines.append("L=15 continues to improve by at least 2pp over L=10.")
    else:
        lines.append("L=15 does not continue with a 2pp gain over L=10; isolated segment representation remains a likely limitation.")
    if l30["accuracy"] - l5["accuracy"] >= 0.10:
        lines.append("The large L=30 versus L=5 gap is confounded by the viewpoint change: L=30 Stay uses s0, while the 0.454266 reference uses the selected s1 viewpoint. It therefore cannot by itself establish segment length as the cause; the same-view L=5/10/15 trend is the valid segment comparison.")
    else:
        lines.append("The fixed-view L=30 s0 control is not comparable to the recorded 0.454266 Stage-C baseline because that reference uses the selected s1 viewpoint. Segment-length attribution should therefore rely on the same-view L=5/10/15 comparison.")
    lines += [
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "new_rgb_generated=false",
        "new_skeleton_generated=false",
        "recognizer_backbone_modified=false",
        "fixed_view_stay_only=true",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_root = data_root / POLICY_RELATIVE
    offline_root = data_root / OFFLINE_RELATIVE
    contexts, data_summary = _load_contexts(policy_root)
    print(json.dumps(data_summary, ensure_ascii=False), flush=True)
    train_sequences, train_labels = _load_train_sequences(policy_root, offline_root)
    val_sequences, val_labels = _load_val_sequences(contexts, offline_root)
    val_s1_sequences, val_s1_labels = _load_val_sequences_for_field(
        contexts, offline_root, "s1_viewpoint_id"
    )
    if not np.array_equal(val_labels, val_s1_labels):
        raise ValueError("s0/s1 validation labels are not aligned")
    encoder, fusion_head, frozen_segment_checkpoints = _load_frozen_segment_models(data_root, device)
    _, train_features_5 = _encode_chunks(encoder, train_sequences, device)
    _, val_features_5 = _encode_chunks(encoder, val_sequences, device)
    mean_feature_logits = _batched_logits(
        fusion_head, val_features_5.mean(axis=1).astype(np.float32), device
    )
    mean_feature_reference = _metrics(mean_feature_logits, val_labels)
    length_metrics: dict[str, Any] = {
        "L5": {
            "segment_count": 6,
            "metrics": {},
            "representation": "ordered shared five-frame encoder features + MLP",
        },
        "L10": {
            "segment_count": 3,
            "metrics": {},
            "representation": "ordered shared segment encoder features + MLP",
        },
        "L15": {
            "segment_count": 2,
            "metrics": {},
            "representation": "ordered shared segment encoder features + MLP",
        },
        "L30": {
            "segment_count": 1,
            "metrics": {},
            "representation": "frozen full-sequence reduced12 ST-GCN on fixed Stay s0",
        },
    }
    runtime_root = data_root / RUNTIME_RELATIVE
    training: dict[str, Any] = {}
    for length in SEGMENT_LENGTHS:
        if length == 5:
            train_features = train_features_5
            val_features = val_features_5
        else:
            train_features = _segment_features(encoder, train_sequences, length, device)
            val_features = _segment_features(encoder, val_sequences, length, device)
        summary, logits = _train_length_classifier(
            train_features,
            train_labels,
            val_features,
            val_labels,
            length,
            device,
            runtime_root / f"ordered_segment_classifier_L{length}_best.pth",
        )
        training[f"L{length}"] = summary
        length_metrics[f"L{length}"]["metrics"] = summary["final_selected_metrics"]
        del train_features, val_features, logits
    full_metrics_s0, stgcn_checkpoint = _full_stgcn_metrics(val_sequences, val_labels, data_root, device)
    full_metrics_s1, _ = _full_stgcn_metrics(val_s1_sequences, val_s1_labels, data_root, device)
    length_metrics["L30"]["metrics"] = full_metrics_s0
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_CHUNK_LENGTH_REPRESENTATION_AUDIT",
        "runtime": runtime,
        "data_summary": data_summary,
        "labels": list(LABELS),
        "frozen_segment_checkpoints": frozen_segment_checkpoints,
        "full_sequence_stgcn_checkpoint": stgcn_checkpoint,
        "protocol": {
            "fixed_view_policy": "Stay",
            "segment_lengths": list(SEGMENT_LENGTHS) + [30],
            "frame_count": FRAME_COUNT,
            "coordinate_dim": COORDINATE_DIM,
            "chunk_feature_dim": HIDDEN_DIM,
            "classifier": "Linear(segment_count*256,256) -> GELU -> Linear(256,12)",
            "macro_f1": "fixed labels 0..11, global macro average, zero_division=0",
            "random_seed": SEED,
            "full_sequence_positive_control_reference": {
                "accuracy": 0.454266,
                "macro_f1": 0.444782,
            },
        },
        "mean_feature_reference": mean_feature_reference,
        "length_metrics": length_metrics,
        "full_sequence_positive_control": {
            "fixed_stay_s0": full_metrics_s0,
            "frozen_stagecv0_selected_s1": full_metrics_s1,
            "recorded_reference": {"accuracy": 0.454266, "macro_f1": 0.444782},
            "interpretation": "The recorded 0.454266 reference is the Stage-D/FrozenStageCv0 selected s1 viewpoint, not fixed-view Stay s0.",
        },
        "training": training,
        "policy_test_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "recognizer_backbone_modified": False,
        "fixed_view_stay_only": True,
    }
    _write(output_dir / "config.json", {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "segment_lengths": list(SEGMENT_LENGTHS),
        "num_classes": NUM_CLASSES,
        "hidden_dim": HIDDEN_DIM,
        "fixed_view_policy": "Stay",
        "policy_test_used": False,
    })
    _write(output_dir / "training_summary.json", training)
    _write(output_dir / "result.json", result)
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = _cuda(args.device)
    result = run(args.output_dir.resolve(), args.data_root.resolve(), device)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["data_summary"]["stage_d_val_moving_contexts"],
        "L30_accuracy": result["length_metrics"]["L30"]["metrics"]["accuracy"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
