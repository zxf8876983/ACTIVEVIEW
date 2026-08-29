"""Current-conditioned Utility predictors for Stage C."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from activeview.active_view.stage_c_features import CANDIDATE_GEOMETRY_DIM, CURRENT_FEATURE_DIM
from activeview.active_view.stage_c_v2_features import JOINT_TOKEN_DIM, SEMANTIC_CONTEXT_DIM


class CurrentContextEncoder(nn.Module):
    def __init__(self, input_dim: int = CURRENT_FEATURE_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
        )

    def forward(self, current_feature: torch.Tensor) -> torch.Tensor:
        return self.network(current_feature)


class CandidateGeometryEncoder(nn.Module):
    def __init__(self, input_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.network(geometry)


class _UtilityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(tokens).squeeze(-1)


class PairwiseUtilityMLP(nn.Module):
    """Pairwise baseline without candidate-set interaction."""

    model_type = "pairwise_mlp"

    def __init__(self, current_dim: int = CURRENT_FEATURE_DIM, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__()
        self.current_encoder = CurrentContextEncoder(current_dim)
        self.geometry_encoder = CandidateGeometryEncoder(geometry_dim)
        self.token_projection = nn.Sequential(nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU())
        self.utility_head = _UtilityHead()

    def forward(self, current_feature: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = self.current_encoder(current_feature).unsqueeze(1)
        geometry = self.geometry_encoder(candidate_geometry)
        tokens = self.token_projection(torch.cat([current.expand(-1, geometry.size(1), -1), geometry], dim=-1))
        return self.utility_head(tokens)


class SetUtilityRanker(nn.Module):
    """Permutation-equivariant candidate-set Utility ranker."""

    model_type = "set_ranker"

    def __init__(self, current_dim: int = CURRENT_FEATURE_DIM, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__()
        self.current_encoder = CurrentContextEncoder(current_dim)
        self.geometry_encoder = CandidateGeometryEncoder(geometry_dim)
        self.token_projection = nn.Sequential(nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.utility_head = _UtilityHead()

    def forward(self, current_feature: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = self.current_encoder(current_feature).unsqueeze(1)
        geometry = self.geometry_encoder(candidate_geometry)
        tokens = self.token_projection(torch.cat([current.expand(-1, geometry.size(1), -1), geometry], dim=-1))
        padding_mask = None if candidate_mask is None else ~candidate_mask
        tokens = self.interaction(tokens, src_key_padding_mask=padding_mask)
        return self.utility_head(tokens)


class MoveStaySetUtilityRanker(SetUtilityRanker):
    """Set ranker with a current-only binary Move/Stay decision head."""

    model_type = "move_stay_set_ranker"

    def __init__(self, current_dim: int = CURRENT_FEATURE_DIM, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__(current_dim, geometry_dim)
        self.move_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def move_logits(self, current_feature: torch.Tensor) -> torch.Tensor:
        """Return one Move logit per Episode using current context only."""
        return self.move_head(self.current_encoder(current_feature)).squeeze(-1)


class _SemanticContextEncoder(nn.Module):
    """Encode the observable ST-GCN classification context only."""

    def __init__(self, input_dim: int = SEMANTIC_CONTEXT_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(),
        )

    def forward(self, semantic_context: torch.Tensor) -> torch.Tensor:
        return self.network(semantic_context)


class _JointTokenEncoder(nn.Module):
    """Shared projection for final-block frozen ST-GCN joint tokens."""

    def __init__(self, input_dim: int = JOINT_TOKEN_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
        )

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        if joint_tokens.ndim != 3 or joint_tokens.size(-1) != JOINT_TOKEN_DIM:
            raise ValueError("joint_tokens must have shape (batch, joints, 256)")
        return self.network(joint_tokens)


class JointAwareCurrentEncoder(nn.Module):
    """Mean-pool joint-aware frozen features with current semantic context."""

    def __init__(self, semantic_dim: int = SEMANTIC_CONTEXT_DIM) -> None:
        super().__init__()
        self.joint_encoder = _JointTokenEncoder()
        self.semantic_encoder = _SemanticContextEncoder(semantic_dim)
        self.context = nn.Sequential(
            nn.Linear(128 + 32, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
        )

    def forward(self, joint_tokens: torch.Tensor, semantic_context: torch.Tensor) -> torch.Tensor:
        pooled = self.joint_encoder(joint_tokens).mean(dim=1)
        semantic = self.semantic_encoder(semantic_context)
        return self.context(torch.cat([pooled, semantic], dim=-1))


class _SetInteractionMixin:
    """Shared candidate-set interaction implementation for v2 rankers."""

    def _interact(self, tokens: torch.Tensor, candidate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        padding_mask = None if candidate_mask is None else ~candidate_mask
        return self.interaction(tokens, src_key_padding_mask=padding_mask)


class JointAwareSetUtilityRanker(_SetInteractionMixin, nn.Module):
    """EXP008: Set Ranker using mean-pooled frozen joint-aware tokens."""

    model_type = "joint_aware_set_ranker"

    def __init__(self, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__()
        self.current_encoder = JointAwareCurrentEncoder()
        self.geometry_encoder = CandidateGeometryEncoder(geometry_dim)
        self.token_projection = nn.Sequential(nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.utility_head = _UtilityHead()

    def forward(self, joint_tokens: torch.Tensor, semantic_context: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = self.current_encoder(joint_tokens, semantic_context).unsqueeze(1)
        geometry = self.geometry_encoder(candidate_geometry)
        tokens = self.token_projection(torch.cat([current.expand(-1, geometry.size(1), -1), geometry], dim=-1))
        return self.utility_head(self._interact(tokens, candidate_mask))

class CandidateConditionedAttentionRanker(nn.Module):
    """EXP009: each candidate geometry queries the current joint tokens."""

    model_type = "candidate_conditioned_attention"

    def __init__(self, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> None:
        super().__init__()
        self.joint_encoder = _JointTokenEncoder()
        self.semantic_encoder = _SemanticContextEncoder()
        self.query_encoder = nn.Sequential(nn.Linear(geometry_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.geometry_encoder = CandidateGeometryEncoder(geometry_dim)
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.token_projection = nn.Sequential(nn.Linear(128 + 64 + 32, 128), nn.LayerNorm(128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.utility_head = _UtilityHead()

    def forward(self, joint_tokens: torch.Tensor, semantic_context: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        joint = self.joint_encoder(joint_tokens)
        batch, candidates, _ = candidate_geometry.shape
        queries = self.query_encoder(candidate_geometry)
        keys = joint.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(batch * candidates, joint.size(1), 128)
        query = queries.reshape(batch * candidates, 1, 128)
        attended, _ = self.cross_attention(query, keys, keys)
        attended = attended.reshape(batch, candidates, 128)
        semantic = self.semantic_encoder(semantic_context).unsqueeze(1).expand(-1, candidates, -1)
        geometry = self.geometry_encoder(candidate_geometry)
        tokens = self.token_projection(torch.cat([attended, geometry, semantic], dim=-1))
        padding_mask = None if candidate_mask is None else ~candidate_mask
        return self.utility_head(self.interaction(tokens, src_key_padding_mask=padding_mask))


class SkeletonPolicyTransformer(nn.Module):
    """EXP010: directly encode the current 30x17 skeleton sequence."""

    model_type = "skeleton_policy_transformer"

    def __init__(self, geometry_dim: int = CANDIDATE_GEOMETRY_DIM, frames: int = 30, joints: int = 17) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.skeleton_projection = nn.Linear(3, 128)
        self.joint_embedding = nn.Embedding(joints, 128)
        self.temporal_embedding = nn.Embedding(frames, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.skeleton_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.semantic_encoder = _SemanticContextEncoder()
        self.query_encoder = nn.Sequential(nn.Linear(geometry_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.geometry_encoder = CandidateGeometryEncoder(geometry_dim)
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.token_projection = nn.Sequential(nn.Linear(128 + 64 + 32, 128), nn.LayerNorm(128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=1)
        self.utility_head = _UtilityHead()

    def forward(self, skeleton: torch.Tensor, semantic_context: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skeleton.ndim != 4 or skeleton.shape[1:] != (3, self.frames, self.joints):
            raise ValueError(f"skeleton must have shape (batch,3,{self.frames},{self.joints})")
        sequence = skeleton.permute(0, 2, 3, 1).reshape(skeleton.size(0), self.frames * self.joints, 3)
        joint_ids = torch.arange(self.joints, device=skeleton.device).repeat(self.frames)
        time_ids = torch.arange(self.frames, device=skeleton.device).repeat_interleave(self.joints)
        tokens = self.skeleton_projection(sequence) + self.joint_embedding(joint_ids) + self.temporal_embedding(time_ids)
        encoded = self.skeleton_encoder(tokens)
        batch, candidates, _ = candidate_geometry.shape
        queries = self.query_encoder(candidate_geometry)
        keys = encoded.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(batch * candidates, encoded.size(1), 128)
        query = queries.reshape(batch * candidates, 1, 128)
        attended, _ = self.cross_attention(query, keys, keys)
        attended = attended.reshape(batch, candidates, 128)
        semantic = self.semantic_encoder(semantic_context).unsqueeze(1).expand(-1, candidates, -1)
        geometry = self.geometry_encoder(candidate_geometry)
        candidate_tokens = self.token_projection(torch.cat([attended, geometry, semantic], dim=-1))
        padding_mask = None if candidate_mask is None else ~candidate_mask
        return self.utility_head(self.interaction(candidate_tokens, src_key_padding_mask=padding_mask))


def build_utility_predictor(model_type: str, *, current_dim: int = CURRENT_FEATURE_DIM, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> nn.Module:
    if model_type == "pairwise_mlp":
        return PairwiseUtilityMLP(current_dim, geometry_dim)
    if model_type == "set_ranker":
        return SetUtilityRanker(current_dim, geometry_dim)
    if model_type == "move_stay_set_ranker":
        return MoveStaySetUtilityRanker(current_dim, geometry_dim)
    if model_type == "joint_aware_set_ranker":
        return JointAwareSetUtilityRanker(geometry_dim)
    if model_type == "candidate_conditioned_attention":
        return CandidateConditionedAttentionRanker(geometry_dim)
    if model_type == "skeleton_policy_transformer":
        return SkeletonPolicyTransformer(geometry_dim)
    raise ValueError(f"Unknown Stage C model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
