"""Frozen ST-GCN intermediate features and Stage C-v2 cache helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


JOINT_COUNT = 17
JOINT_TOKEN_DIM = 256
SEMANTIC_CONTEXT_DIM = 19  # 16 log-probabilities + entropy/margin/confidence
SKELETON_SHAPE = (3, 30, JOINT_COUNT)


def semantic_context_from_current_feature(current_feature: Sequence[float]) -> np.ndarray:
    """Return only the 19-D observable semantic context from a v0 row."""
    values = np.asarray(current_feature, dtype=np.float32)
    if values.shape != (275,) or not np.isfinite(values).all():
        raise ValueError("current_feature must be a finite 275-D vector")
    return values[256:].copy()


def extract_frozen_joint_tokens(
    model: torch.nn.Module,
    skeleton_batch: torch.Tensor,
) -> torch.Tensor:
    """Extract final-block joint tokens without changing the frozen ST-GCN.

    ``STGCN.forward_features`` performs the normal forward pass and a hook on
    its final ``STGCNBlock`` captures the tensor immediately before the model's
    global temporal/spatial average pooling.  The resulting temporal mean has
    shape ``(B, 17, 256)`` for the accepted H36M-17, 30-frame input.
    """
    if skeleton_batch.ndim == 4:
        skeleton_batch = skeleton_batch.unsqueeze(-1)
    if skeleton_batch.ndim != 5:
        raise ValueError("skeleton_batch must have shape (B,3,T,17[,1])")
    if skeleton_batch.shape[1] != 3 or skeleton_batch.shape[3] != JOINT_COUNT:
        raise ValueError(f"Unexpected skeleton shape: {tuple(skeleton_batch.shape)}")
    if not torch.isfinite(skeleton_batch).all():
        raise ValueError("skeleton_batch contains non-finite values")
    blocks = getattr(model, "st_gcn_networks", None)
    if blocks is None or len(blocks) == 0:
        raise TypeError("Frozen model does not expose ST-GCN blocks")
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["output"] = output

    handle = blocks[-1].register_forward_hook(_capture)
    try:
        with torch.inference_mode():
            model.forward_features(skeleton_batch)
    finally:
        handle.remove()
    output = captured.get("output")
    if output is None or output.ndim != 4:
        raise RuntimeError("Failed to capture final ST-GCN joint-temporal feature")
    if output.shape[0] != skeleton_batch.shape[0] or output.shape[1] != JOINT_TOKEN_DIM or output.shape[3] != JOINT_COUNT:
        raise ValueError(f"Unexpected final ST-GCN feature shape: {tuple(output.shape)}")
    tokens = output.mean(dim=2).permute(0, 2, 1).contiguous()
    if tokens.shape[1:] != (JOINT_COUNT, JOINT_TOKEN_DIM):
        raise ValueError(f"Unexpected joint-token shape: {tuple(tokens.shape)}")
    return tokens


def v2_schema(*, candidate_geometry_dim: int = 11) -> dict[str, Any]:
    """Describe the leakage-safe Stage C-v2 cache contract."""
    return {
        "version": "stage-c-v2-current-observation-cache",
        "current_joint_tokens_shape": [JOINT_COUNT, JOINT_TOKEN_DIM],
        "current_skeleton_shape": list(SKELETON_SHAPE),
        "semantic_context_dim": SEMANTIC_CONTEXT_DIM,
        "candidate_geometry_dim": int(candidate_geometry_dim),
        "joint_token_source": "frozen ST-GCN final block before global average pooling, temporal mean only",
        "input_whitelist": [
            "current_joint_tokens",
            "current_skeleton",
            "current_log_probabilities",
            "current_entropy",
            "current_margin",
            "current_pose_confidence",
            "candidate_geometry",
        ],
        "forbidden_input_fields": [
            "label_id",
            "action_label",
            "candidate_skeleton",
            "candidate_confidence",
            "candidate_log_probs",
            "candidate_entropy",
            "candidate_utility",
            "candidate_prediction",
            "gt_correctness",
            "viewpoint_id",
            "safe_oracle_choice",
            "future_perception",
        ],
        "future_candidate_perception_used_as_input": False,
        "body_yaw_used": False,
        "movement_cost_penalty_used": False,
    }
