"""Pretrained history identity encoder paired with the Multi-positive JR."""

from __future__ import annotations

import torch
from torch import nn

from activeview.methods.joint_revision.history_aware import (
    HISTORY_LATENT_DIM,
    HistoryIdentityEncoder,
)
from activeview.methods.joint_revision.model import JointRevision


class PretrainedHistoryAwareJointRevision(JointRevision):
    """Joint Revision that consumes a pretrained history identity branch.

    The base Multi-positive JR architecture is retained.  The only change is
    that the current token receives both the 128-D history latent and the
    refined 14-way identity logits produced by ``HistoryIdentityEncoder``.
    """

    def __init__(self, num_classes: int = 14) -> None:
        super().__init__(num_classes=num_classes)
        self.history_identity = HistoryIdentityEncoder(num_classes)
        self.current_projector = nn.Sequential(
            nn.Linear(self.current_dim + HISTORY_LATENT_DIM + num_classes, 128),
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
        enriched_current = torch.cat([current, latent, refined_logits], dim=-1)
        current_token = self.current_projector(enriched_current).unsqueeze(1)
        candidate_tokens = self.candidate_projector(candidates)
        tokens = torch.cat([current_token, candidate_tokens], dim=1)
        full_mask = torch.cat(
            [torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return (
            self.score(encoded[:, 1:]).squeeze(-1),
            self.posterior(encoded[:, 1:]),
            refined_logits,
        )


__all__ = ["PretrainedHistoryAwareJointRevision"]
