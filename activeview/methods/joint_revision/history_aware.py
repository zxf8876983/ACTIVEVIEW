"""History-aware extension of the frozen Multi-positive Joint Revision."""

from __future__ import annotations

import torch
from torch import nn

from activeview.methods.joint_revision.model import JointRevision


STGCN_FEATURE_DIM = 256
HISTORY_LATENT_DIM = 128


class HistoryIdentityEncoder(nn.Module):
    """Encode observed s0/s1 ST-GCN features and posteriors."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_dim = 2 * STGCN_FEATURE_DIM + 2 * self.num_classes
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, HISTORY_LATENT_DIM),
            nn.GELU(),
        )
        self.classifier = nn.Linear(HISTORY_LATENT_DIM, self.num_classes)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(history)
        return latent, self.classifier(latent)


class HistoryAwareJointRevision(JointRevision):
    """Multi-positive JR with a Train-supervised history identity branch."""

    def __init__(self, num_classes: int = 14) -> None:
        super().__init__(num_classes=num_classes)
        self.history_identity = HistoryIdentityEncoder(num_classes)
        self.current_projector = nn.Sequential(
            nn.Linear(self.current_dim + HISTORY_LATENT_DIM, 128),
            nn.GELU(),
        )

    def forward(
        self,
        current: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, refined_logits = self.history_identity(history)
        current_token = self.current_projector(torch.cat([current, latent], dim=-1)).unsqueeze(1)
        candidate_tokens = self.candidate_projector(candidates)
        tokens = torch.cat([current_token, candidate_tokens], dim=1)
        full_mask = torch.cat(
            [torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.score(encoded[:, 1:]).squeeze(-1), self.posterior(encoded[:, 1:]), refined_logits


__all__ = ["HISTORY_LATENT_DIM", "HistoryAwareJointRevision", "HistoryIdentityEncoder"]
