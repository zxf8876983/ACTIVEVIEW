#!/usr/bin/env python3
"""Audit fixed-view versus Stay MeanLogP consistency for segment-aware MVP.

This is a read-only Val Moving audit.  It reuses the existing chunk-encoder
checkpoint and compares the direct s0 sequence path with the all-view path
used by Stay, tensor by tensor.  No model is trained and no MeanFeature or
formal ST-GCN result is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import classification
from activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition import (
    CHUNK_SIZE,
    DEFAULT_OUTPUT,
    FRAME_COUNT,
    INFERENCE_BATCH_SIZE,
    NUM_CHUNKS,
    OFFLINE_RELATIVE,
    POLICY_RELATIVE,
    RUNTIME_RELATIVE,
    _encode_chunks,
    _load_all_views,
    _load_contexts,
    _load_val_sequences,
    _log_softmax,
    _record_path,
    ChunkEncoder,
)


DEFAULT_AUDIT_OUTPUT = DEFAULT_OUTPUT / "meanlogp_consistency_audit"
BATCH_CONTEXTS = 128
PRE_FIX_FIXED_MEANLOGP_ACCURACY = 0.09295634920634921


def _load_encoder(checkpoint: Path, device: torch.device) -> ChunkEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ChunkEncoder(hidden_dim=int(payload.get("hidden_dim", 256))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _metrics_from_logp(logp: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logp, axis=1)
    return classification(labels.tolist(), predictions.tolist())


def _run_audit(
    data_root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_contexts: int = BATCH_CONTEXTS,
) -> dict[str, Any]:
    policy_root = data_root / POLICY_RELATIVE
    offline_root = data_root / OFFLINE_RELATIVE
    contexts, summary = _load_contexts(policy_root)
    val_sequences, labels = _load_val_sequences(contexts, offline_root)
    model = _load_encoder(checkpoint, device)

    # Match the production fixed-view and all-view paths (2048 sequences per
    # encoder call) so the audit isolates the MeanLogP dimension bug.
    fixed_logits, fixed_features = _encode_chunks(
        model, val_sequences, device, sequence_batch_size=INFERENCE_BATCH_SIZE
    )
    fixed_logp = _log_softmax(fixed_logits)
    selected_sequence_parts: list[np.ndarray] = []
    skeleton_max_abs = 0.0
    chunk_input_max_abs = 0.0
    first_mismatch: dict[str, Any] | None = None

    for group_start in range(0, len(contexts), batch_contexts):
        group = contexts[group_start : group_start + batch_contexts]
        group_sequences = np.stack([
            _load_all_views(_record_path(offline_root, context["row"]))
            for context in group
        ]).astype(np.float32)
        starts = np.asarray([int(context["start_view"]) for context in group], dtype=np.int64)
        indices = np.arange(len(group), dtype=np.int64)
        selected = group_sequences[indices, starts]
        direct = val_sequences[group_start : group_start + len(group)]
        skeleton_error = _max_abs(direct, selected)
        skeleton_max_abs = max(skeleton_max_abs, skeleton_error)
        direct_chunks = direct.reshape(-1, NUM_CHUNKS, CHUNK_SIZE, direct.shape[-1])
        selected_chunks = selected.reshape(-1, NUM_CHUNKS, CHUNK_SIZE, selected.shape[-1])
        chunk_error = _max_abs(direct_chunks, selected_chunks)
        chunk_input_max_abs = max(chunk_input_max_abs, chunk_error)

        selected_sequence_parts.append(selected)
        if first_mismatch is None and (skeleton_error > 0.0 or chunk_error > 0.0):
            first_mismatch = {
                "stage": "skeleton/chunk input",
                "context_index": group_start,
                "episode_id": str(group[0]["episode_id"]),
                "skeleton_max_abs": skeleton_error,
                "chunk_input_max_abs": chunk_error,
            }

    # Encode the selected Stay sequences as one Val stream with the same
    # sequence batch size as the fixed-view path, so the tensor comparison is
    # not confounded by CUDA GRU batch-shape-dependent arithmetic.
    selected_sequences = np.concatenate(selected_sequence_parts, axis=0)
    stay_logits, stay_features = _encode_chunks(
        model, selected_sequences, device, sequence_batch_size=INFERENCE_BATCH_SIZE
    )
    stay_logp = _log_softmax(stay_logits)
    fixed_aggregate = np.mean(fixed_logp, axis=1)
    stay_aggregate = np.mean(stay_logp, axis=1)
    feature_error = _max_abs(fixed_features, stay_features)
    logits_error = _max_abs(fixed_logits, stay_logits)
    logp_error = _max_abs(fixed_logp, stay_logp)
    aggregate_error = _max_abs(fixed_aggregate, stay_aggregate)
    prediction_mismatch = int(np.sum(np.argmax(fixed_aggregate, axis=1) != np.argmax(stay_aggregate, axis=1)))

    if first_mismatch is None:
        checks = [
            ("feature", feature_error),
            ("logits", logits_error),
            ("log_softmax", logp_error),
            ("mean_logp_aggregation", aggregate_error),
            ("prediction", float(prediction_mismatch)),
        ]
        for stage, value in checks:
            if value > 0.0:
                first_mismatch = {"stage": stage, "max_abs_or_count": value}
                break

    previous_accuracy = PRE_FIX_FIXED_MEANLOGP_ACCURACY
    fixed_metrics = _metrics_from_logp(fixed_aggregate, labels)
    stay_metrics = _metrics_from_logp(stay_aggregate, labels)
    return {
        "audit": {
            "context_count": len(contexts),
            "same_start_view_alignment": True,
            "chunk_ranges": [[i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE] for i in range(NUM_CHUNKS)],
            "encoder_checkpoint": str(checkpoint.resolve()),
            "device": str(device),
            "first_difference": first_mismatch,
        },
        "tensor_comparison": {
            "skeleton_max_abs_error": skeleton_max_abs,
            "chunk_input_max_abs_error": chunk_input_max_abs,
            "feature_max_abs_error": feature_error,
            "logits_max_abs_error": logits_error,
            "logp_max_abs_error": logp_error,
            "aggregated_meanlogp_max_abs_error": aggregate_error,
            "prediction_mismatch_count": prediction_mismatch,
        },
        "metrics": {
            "pre_fix_fixed_view_meanlogp_t30_accuracy": previous_accuracy,
            "pre_fix_first_difference": {
                "stage": "log_softmax",
                "detail": "fixed-view passed a (N,6,12) tensor to axis=1 normalization, so six chunks—not twelve classes—were normalized",
            },
            "recomputed_fixed_view_meanlogp_t30": fixed_metrics,
            "recomputed_sequential_stay_meanlogp_t30": stay_metrics,
            "fixed_view_accuracy": fixed_metrics["accuracy"],
            "stay_accuracy": stay_metrics["accuracy"],
            "final_consistent": bool(prediction_mismatch == 0 and aggregate_error <= 1e-7),
        },
        "data_summary": summary,
        "flags": {
            "policy_test_used": False,
            "model_retrained": False,
            "meanfeature_modified": False,
            "stgcn_modified": False,
            "train_val_split_modified": False,
        },
    }


def _analysis(result: dict[str, Any]) -> str:
    comparison = result["tensor_comparison"]
    metrics = result["metrics"]
    first = result["audit"]["first_difference"]
    root_cause = (
        "Before the fix, fixed-view applied axis=1 log-softmax to the 3-D "
        "(N,6,12) logits tensor, normalizing across chunk/time instead of the "
        "12-class dimension. Stay receives a 2-D (N,12) tensor, so axis=1 "
        "was the class dimension there."
    )
    if first is not None:
        root_cause += f" The first residual numerical difference occurs at `{first['stage']}`."
    else:
        root_cause += " After the fix, no residual difference remains in the aligned path."
    consistent = metrics["final_consistent"]
    conclusion = "fixed-view and Stay MeanLogP are now numerically consistent." if consistent else "fixed-view and Stay MeanLogP remain inconsistent and require the first-difference stage above to be corrected before horizon audits."
    return "\n".join([
        "# MeanLogP Fixed-view / Stay Consistency Audit",
        "",
        f"Val Moving contexts compared: {result['audit']['context_count']}. Both paths use the same Stage-D s0 viewpoint and six chunks {result['audit']['chunk_ranges']}; no Test data was read.",
        "",
        "## Tensor comparison",
        "",
        "| Quantity | Max absolute error |",
        "|---|---:|",
        f"| Skeleton | {comparison['skeleton_max_abs_error']:.9g} |",
        f"| Chunk input | {comparison['chunk_input_max_abs_error']:.9g} |",
        f"| Encoder feature | {comparison['feature_max_abs_error']:.9g} |",
        f"| Encoder logits | {comparison['logits_max_abs_error']:.9g} |",
        f"| log-softmax | {comparison['logp_max_abs_error']:.9g} |",
        f"| MeanLogP aggregation | {comparison['aggregated_meanlogp_max_abs_error']:.9g} |",
        f"| Prediction mismatches | {comparison['prediction_mismatch_count']} |",
        "",
        f"Root-cause diagnosis: {root_cause}",
        "",
        "## Accuracy comparison",
        "",
        f"- Previous reported fixed-view MeanLogP t30 Accuracy: {metrics['pre_fix_fixed_view_meanlogp_t30_accuracy']:.9f}",
        f"- Recomputed fixed-view MeanLogP t30 Accuracy: {metrics['fixed_view_accuracy']:.9f}",
        f"- Recomputed sequential Stay MeanLogP t30 Accuracy: {metrics['stay_accuracy']:.9f}",
        f"- Final verdict: {conclusion}",
        "",
        "## Root cause / scope",
        "",
        "The audit compares the direct fixed-view sequence path and the all-view Stay path at skeleton, chunk, feature, logits, log-softmax and aggregation levels. It does not alter MeanFeature, the frozen ST-GCN, Train/Val splits or runtime artifacts.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "model_retrained=false",
        "meanfeature_modified=false",
        "stgcn_modified=false",
        "train_val_split_modified=false",
        "```",
        "",
    ])


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    checkpoint = args.checkpoint or (args.data_root / RUNTIME_RELATIVE / "chunk_encoder_best.pth")
    result = _run_audit(args.data_root.resolve(), checkpoint.resolve(), device)
    _write(args.output_dir.resolve() / "result.json", result)
    (args.output_dir.resolve() / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["audit"]["context_count"],
        "first_difference": result["audit"]["first_difference"],
        "final_consistent": result["metrics"]["final_consistent"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
