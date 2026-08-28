"""Current-conditioned Utility predictors for Stage C."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from activeview.active_view.stage_c_features import CANDIDATE_GEOMETRY_DIM, CURRENT_FEATURE_DIM


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


def build_utility_predictor(model_type: str, *, current_dim: int = CURRENT_FEATURE_DIM, geometry_dim: int = CANDIDATE_GEOMETRY_DIM) -> nn.Module:
    if model_type == "pairwise_mlp":
        return PairwiseUtilityMLP(current_dim, geometry_dim)
    if model_type == "set_ranker":
        return SetUtilityRanker(current_dim, geometry_dim)
    raise ValueError(f"Unknown Stage C model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
