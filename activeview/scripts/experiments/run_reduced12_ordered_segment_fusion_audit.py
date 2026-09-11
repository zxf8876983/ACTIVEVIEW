#!/usr/bin/env python3
"""Train and evaluate an ordered segment-feature fusion audit.

The audit keeps the existing frozen five-frame chunk encoder fixed and compares
the current MeanFeature fusion with a small classifier that receives the six
chunk features in temporal order.  It only reads Stage-C Train and Stage-D
Moving Val artifacts; no policy Test or perception generation is involved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification
from activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition import (
    BATCH_SIZE,
    CHUNK_SIZE,
    DEFAULT_OUTPUT as SEGMENT_OUTPUT,
    HIDDEN_DIM,
    INFERENCE_BATCH_SIZE,
    NUM_CHUNKS,
    NUM_CLASSES,
    _encode_chunks,
    _load_contexts,
    _load_train_sequences,
    _load_val_sequences,
    _record_path,
    ChunkEncoder,
    FeatureFusionHead,
)


SEED = 42
FRAME_COUNT = 30
ORDERED_FEATURE_DIM = NUM_CHUNKS * HIDDEN_DIM
MASK_DIM = NUM_CHUNKS
ORDERED_INPUT_DIM = ORDERED_FEATURE_DIM + MASK_DIM
ORDERED_HIDDEN_DIM = 256
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PREFIXES = tuple(range(1, NUM_CHUNKS + 1))
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/ordered_segment_fusion_audit"
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
OFFLINE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")
RUNTIME_RELATIVE = Path(
    "checkpoints/activeview_reduced12_eight_placement_v1/segment_aware_sequential_recognition"
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


class OrderedConcatClassifier(nn.Module):
    """Ordered six-chunk classifier with an explicit observed-prefix mask."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ORDERED_INPUT_DIM, ORDERED_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(ORDERED_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _ordered_inputs(features: np.ndarray, prefix: int, reverse: bool = False) -> np.ndarray:
    if features.ndim != 3 or features.shape[1:] != (NUM_CHUNKS, HIDDEN_DIM):
        raise ValueError(f"unexpected chunk feature shape {features.shape}")
    if prefix < 0 or prefix > NUM_CHUNKS:
        raise ValueError(f"prefix must be in [0,{NUM_CHUNKS}], got {prefix}")
    ordered = features[:, ::-1] if reverse else features
    output = np.zeros((len(features), ORDERED_INPUT_DIM), dtype=np.float32)
    if prefix:
        output[:, : prefix * HIDDEN_DIM] = ordered[:, :prefix].reshape(len(features), -1)
    output[:, ORDERED_FEATURE_DIM : ORDERED_FEATURE_DIM + prefix] = 1.0
    return output


def _classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=1)
    probabilities = _softmax(logits)
    metrics = classification(labels.tolist(), predictions.tolist())
    metrics.update({
        "mean_entropy": float(np.mean(_entropy(probabilities))),
        "mean_gt_probability": float(np.mean(probabilities[np.arange(len(labels)), labels])),
    })
    return metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)


def _load_frozen_models(data_root: Path, device: torch.device) -> tuple[ChunkEncoder, FeatureFusionHead, dict[str, str]]:
    runtime_root = data_root / RUNTIME_RELATIVE
    encoder_checkpoint = runtime_root / "chunk_encoder_best.pth"
    fusion_checkpoint = runtime_root / "feature_fusion_head_best.pth"
    if not encoder_checkpoint.exists() or not fusion_checkpoint.exists():
        raise FileNotFoundError(
            "Missing frozen segment-aware checkpoints: "
            f"{encoder_checkpoint} and {fusion_checkpoint}"
        )
    encoder = ChunkEncoder().to(device)
    encoder_payload = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_payload["state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    fusion = FeatureFusionHead().to(device)
    fusion_payload = torch.load(fusion_checkpoint, map_location=device, weights_only=False)
    fusion.load_state_dict(fusion_payload["state_dict"])
    fusion.eval()
    for parameter in fusion.parameters():
        parameter.requires_grad_(False)
    return encoder, fusion, {
        "chunk_encoder": str(encoder_checkpoint.resolve()),
        "mean_feature_head": str(fusion_checkpoint.resolve()),
    }


def _ordered_logits(
    model: OrderedConcatClassifier,
    features: np.ndarray,
    prefix: int,
    device: torch.device,
    reverse: bool = False,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    inputs = _ordered_inputs(features, prefix, reverse=reverse)
    parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device, non_blocking=True)
        with torch.inference_mode():
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _mean_feature_logits(
    fusion_head: FeatureFusionHead,
    features: np.ndarray,
    prefix: int,
    device: torch.device,
) -> np.ndarray:
    mean_features = features[:, :prefix].mean(axis=1).astype(np.float32)
    parts: list[np.ndarray] = []
    fusion_head.eval()
    for start in range(0, len(mean_features), INFERENCE_BATCH_SIZE):
        batch = torch.from_numpy(mean_features[start : start + INFERENCE_BATCH_SIZE]).to(device, non_blocking=True)
        with torch.inference_mode():
            parts.append(fusion_head(batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _evaluate_prefixes(
    model: OrderedConcatClassifier,
    fusion_head: FeatureFusionHead,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for prefix in PREFIXES:
        ordered_metrics = _classification_metrics(
            _ordered_logits(model, val_features, prefix, device), val_labels
        )
        mean_metrics = _classification_metrics(
            _mean_feature_logits(fusion_head, val_features, prefix, device), val_labels
        )
        observed = prefix * CHUNK_SIZE
        output[f"t{observed}"] = {
            "observed_frames": observed,
            "mean_feature": mean_metrics,
            "ordered_concat": ordered_metrics,
        }
    return output


def _evaluate_reverse(
    model: OrderedConcatClassifier,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    normal = _classification_metrics(_ordered_logits(model, val_features, NUM_CHUNKS, device), val_labels)
    reversed_metrics = _classification_metrics(
        _ordered_logits(model, val_features, NUM_CHUNKS, device, reverse=True), val_labels
    )
    return {"normal_order": normal, "reversed_order": reversed_metrics}


def _val_score(model: OrderedConcatClassifier, features: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    values: list[float] = []
    for prefix in PREFIXES:
        logits = _ordered_logits(model, features, prefix, device)
        values.append(float(classification(labels.tolist(), np.argmax(logits, axis=1).tolist())["macro_f1"]))
    return float(np.mean(values))


def _train_ordered_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
) -> tuple[OrderedConcatClassifier, dict[str, Any]]:
    _seed()
    model = OrderedConcatClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    train_count = len(train_features)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        prefix_losses: list[float] = []
        for prefix_index, prefix in enumerate(PREFIXES):
            generator = torch.Generator().manual_seed(SEED + epoch * 100 + prefix_index)
            order = torch.randperm(train_count, generator=generator).numpy()
            losses: list[float] = []
            for start in range(0, train_count, BATCH_SIZE):
                batch_indices = order[start : start + BATCH_SIZE]
                inputs = torch.from_numpy(_ordered_inputs(train_features[batch_indices], prefix)).to(
                    device, non_blocking=True
                )
                labels = torch.from_numpy(train_labels[batch_indices]).to(device, non_blocking=True)
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            prefix_losses.append(float(np.mean(losses)))
        val_macro = _val_score(model, val_features, val_labels, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(prefix_losses)),
            "val_mean_prefix_macro_f1": val_macro,
        }
        history.append(record)
        print(
            f"[ordered-concat] epoch={epoch:02d}/{EPOCHS} "
            f"loss={record['train_loss']:.6f} val_macro_f1={val_macro:.6f}",
            flush=True,
        )
        if val_macro > best_score + 1e-8:
            best_score = val_macro
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": ORDERED_INPUT_DIM,
                    "hidden_dim": ORDERED_HIDDEN_DIM,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "epoch": epoch,
                },
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary: dict[str, Any] = {
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "best_val_mean_prefix_macro_f1": best_score,
        "train_contexts": int(len(train_features)),
        "val_contexts": int(len(val_features)),
        "prefix_frames": [int(prefix * CHUNK_SIZE) for prefix in PREFIXES],
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_dim": ORDERED_INPUT_DIM,
        "hidden_dim": ORDERED_HIDDEN_DIM,
        "architecture": "Linear(1542,256) -> GELU -> Linear(256,12)",
        "checkpoint": str(checkpoint.resolve()),
        "seed": SEED,
        "test_used": False,
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model, summary


def _analysis(result: Mapping[str, Any]) -> str:
    rows = result["prefix_metrics"]
    lines = [
        "# Reduced12 Ordered Segment Fusion Audit",
        "",
        "This is a fixed-view Stay-only audit. The frozen five-frame chunk encoder is reused; only the ordered fusion classifier is trained on Stage-C Train. Stage-D Moving Val is used for evaluation and Policy Test is not read.",
        "",
        "## Classifier",
        "",
        "Input is the chronological concatenation `[z0; z1; z2; z3; z4; z5]` (1536-D) plus a six-dimensional observed-chunk mask (1542-D). The classifier is `Linear(1542, 256) -> GELU -> Linear(256, 12)`. Unobserved chunks are zero-filled and their mask entries are zero.",
        "",
        "## Moving Val prefix results",
        "",
        "| Prefix | MeanFeature Acc | MeanFeature F1 | OrderedConcat Acc | OrderedConcat F1 | ΔAcc | ΔF1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for observed in (5, 10, 15, 20, 25, 30):
        item = rows[f"t{observed}"]
        mean_item = item["mean_feature"]
        ordered_item = item["ordered_concat"]
        lines.append(
            f"| {observed} | {mean_item['accuracy']:.6f} | {mean_item['macro_f1']:.6f} | "
            f"{ordered_item['accuracy']:.6f} | {ordered_item['macro_f1']:.6f} | "
            f"{ordered_item['accuracy'] - mean_item['accuracy']:+.6f} | "
            f"{ordered_item['macro_f1'] - mean_item['macro_f1']:+.6f} |"
        )
    normal = result["t30_order_sanity"]["normal_order"]
    reverse = result["t30_order_sanity"]["reversed_order"]
    t30 = rows["t30"]
    delta_acc = result["ordered_concat_minus_mean_feature_t30"]["accuracy"]
    delta_f1 = result["ordered_concat_minus_mean_feature_t30"]["macro_f1"]
    lines += [
        "",
        "## Temporal-order sanity",
        "",
        f"Normal order t30: Acc={normal['accuracy']:.6f}, Macro-F1={normal['macro_f1']:.6f}. Reversed order t30: Acc={reverse['accuracy']:.6f}, Macro-F1={reverse['macro_f1']:.6f}.",
        f"OrderedConcat minus MeanFeature at t30: ΔAcc={delta_acc * 100:+.2f}pp, ΔF1={delta_f1 * 100:+.2f}pp. The full-sequence frozen ST-GCN reference (~0.454266 Acc) is retained only as an external reference; it is not recomputed or modified here.",
        "",
        "## Interpretation",
        "",
        f"At t30, OrderedConcat reaches {t30['ordered_concat']['accuracy']:.6f} accuracy versus MeanFeature {t30['mean_feature']['accuracy']:.6f}.",
    ]
    if delta_acc >= 0.05:
        lines.append("The gain is at least 5pp, so temporal averaging is a strong candidate bottleneck and ordered fusion merits further study.")
    elif delta_acc >= 0.02:
        lines.append("The gain is material but below 5pp; both ordering and the information content of five-frame chunks remain plausible contributors.")
    elif t30["ordered_concat"]["accuracy"] < 0.35:
        lines.append("OrderedConcat remains below 35%, indicating that temporal order alone is not the main bottleneck; the five-frame chunk representation is likely too weak.")
    else:
        lines.append("The gain is limited; temporal averaging is not established as the sole bottleneck by this audit.")
    reverse_drop = normal["accuracy"] - reverse["accuracy"]
    if reverse_drop > 0.01:
        lines.append(f"Reversing chunk order changes accuracy by {reverse_drop * 100:.2f}pp, providing evidence that the classifier uses temporal order.")
    else:
        lines.append(f"Reversing chunk order changes accuracy by only {reverse_drop * 100:.2f}pp, so dimensionality rather than order may explain most of the classifier behavior.")
    lines += [
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "training_used_for_ordered_classifier=true (Stage-C Train only)",
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
    checkpoint_root = data_root / RUNTIME_RELATIVE
    contexts, data_summary = _load_contexts(policy_root)
    print(json.dumps(data_summary, ensure_ascii=False), flush=True)
    train_sequences, train_labels = _load_train_sequences(policy_root, offline_root)
    val_sequences, val_labels = _load_val_sequences(contexts, offline_root)
    encoder, fusion_head, frozen_checkpoints = _load_frozen_models(data_root, device)
    print("[ordered-data] encoding Train/Moving Val chunks with frozen encoder", flush=True)
    _, train_features = _encode_chunks(encoder, train_sequences, device)
    _, val_features = _encode_chunks(encoder, val_sequences, device)
    del train_sequences, val_sequences
    ordered_checkpoint = data_root / "checkpoints/activeview_reduced12_eight_placement_v1/ordered_segment_fusion_audit/ordered_concat_classifier_best.pth"
    model, training_summary = _train_ordered_classifier(
        train_features, train_labels, val_features, val_labels, device,
        ordered_checkpoint, output_dir / "training_summary.json",
    )
    prefix_metrics = _evaluate_prefixes(model, fusion_head, val_features, val_labels, device)
    order_sanity = _evaluate_reverse(model, val_features, val_labels, device)
    t30_ordered = prefix_metrics["t30"]["ordered_concat"]
    t30_mean = prefix_metrics["t30"]["mean_feature"]
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_ORDERED_SEGMENT_FUSION_AUDIT",
        "runtime": runtime,
        "data_summary": data_summary,
        "labels": list(LABELS),
        "frozen_checkpoints": frozen_checkpoints,
        "protocol": {
            "fixed_view_policy": "Stay",
            "chunks": [[i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE] for i in range(NUM_CHUNKS)],
            "chunk_feature_dim": HIDDEN_DIM,
            "ordered_feature_dim": ORDERED_FEATURE_DIM,
            "observed_chunk_mask_dim": MASK_DIM,
            "ordered_input_dim": ORDERED_INPUT_DIM,
            "classifier": "Linear(1542,256) -> GELU -> Linear(256,12)",
            "unobserved_chunk_representation": "zero-filled with explicit observed mask",
            "random_seed": SEED,
            "full_sequence_frozen_reference_accuracy": 0.454266,
        },
        "training": training_summary,
        "prefix_metrics": prefix_metrics,
        "t30_order_sanity": order_sanity,
        "ordered_concat_minus_mean_feature_t30": {
            "accuracy": float(t30_ordered["accuracy"] - t30_mean["accuracy"]),
            "macro_f1": float(t30_ordered["macro_f1"] - t30_mean["macro_f1"]),
        },
        "policy_test_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "recognizer_backbone_modified": False,
        "fixed_view_stay_only": True,
    }
    (output_dir / "config.json").write_text(
        json.dumps({
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "input_dim": ORDERED_INPUT_DIM,
            "ordered_feature_dim": ORDERED_FEATURE_DIM,
            "mask_dim": MASK_DIM,
            "num_classes": NUM_CLASSES,
            "prefix_frames": [prefix * CHUNK_SIZE for prefix in PREFIXES],
            "fixed_view_policy": "Stay",
            "policy_test_used": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
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
    print(
        json.dumps({
            "output_dir": str(args.output_dir.resolve()),
            "contexts": result["data_summary"]["stage_d_val_moving_contexts"],
            "t30_accuracy": result["prefix_metrics"]["t30"]["ordered_concat"]["accuracy"],
        }, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
