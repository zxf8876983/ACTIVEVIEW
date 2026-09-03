"""文件用途：
    实现候选集合 Joint Revision 策略。

主要输入：
    - 历史状态与候选表示。
主要输出：
    - 候选分数和动作选择。
项目角色：
    - 属于 methods.joint_revision 方法模块。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

N_CLASSES = 16


class JointRevision(nn.Module):
    """The frozen EXP055 multi-positive Joint Revision architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.current_projector = nn.Sequential(nn.Linear(38, 128), nn.GELU())
        self.candidate_projector = nn.Sequential(nn.Linear(26, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Linear(128, 1)
        self.posterior = nn.Linear(128, N_CLASSES)

    def forward(self, current: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_token = self.current_projector(current).unsqueeze(1)
        candidate_tokens = self.candidate_projector(candidates)
        tokens = torch.cat([current_token, candidate_tokens], dim=1)
        full_mask = torch.cat([torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.score(encoded[:, 1:]).squeeze(-1), self.posterior(encoded[:, 1:])


def tie_argmax(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array)
    return int(np.flatnonzero(np.isclose(array, maximum, rtol=0.0, atol=1e-12))[0])


def _features(cache: Mapping[str, np.ndarray], index: int, candidates: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    current = np.exp(cache["current_logp_s1"][index])
    probs0 = np.exp(cache["current_logp_s0"][index])
    stats = np.asarray([*probs0, *current, -np.sum(probs0 * cache["current_logp_s0"][index]), -np.sum(current * cache["current_logp_s1"][index]), np.max(probs0), np.max(current), np.sort(current)[-1] - np.sort(current)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)[None]
    values = [np.concatenate([cache["current_logp_s1"][index], np.zeros(9, dtype=np.float32), [1.0]])]
    values.extend(np.concatenate([cache["imagined_logp"][index, candidate], cache["candidate_descriptor"][index, candidate], [0.0]]) for candidate in candidates)
    return stats, np.asarray(values, dtype=np.float32)


def select_actions(
    model: JointRevision, cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]], budget: int | str = "ALL_LEGAL",
    device: torch.device | None = None, posterior_mode: bool = False,
) -> list[int | None]:
    """Select Stay or one legal candidate using model scores only."""
    if device is None:
        device = torch.device("cpu")
    indices = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    selected: list[int | None] = []
    for row in rows:
        order = list(orders[str(row["episode_id"])])
        if not order:
            selected.append(None)
            continue
        index = indices[str(row["episode_id"])]
        stats, values = _features(cache, index, order)
        tensor = torch.zeros((1, max(31, len(values)), 26), dtype=torch.float32)
        tensor[0, : len(values)] = torch.from_numpy(values)
        mask = torch.zeros((1, tensor.shape[1]), dtype=torch.bool)
        mask[0, : len(values)] = True
        with torch.inference_mode():
            scores, _ = model(torch.from_numpy(stats).to(device), tensor.to(device), mask.to(device))
        choice = tie_argmax(scores[0, : len(values)].cpu().numpy())
        selected.append(None if choice == 0 else order[choice - 1])
    return selected


def choose_next(
    model: JointRevision, current_logs: tuple[np.ndarray, np.ndarray], imagined: np.ndarray,
    descriptors: np.ndarray, device: torch.device,
) -> int | None:
    """Choose a second-step index; index 0 is the Stay action."""
    previous, current = current_logs
    probs0, probs1 = np.exp(previous), np.exp(current)
    stats = np.asarray([*probs0, *probs1, -np.sum(probs0 * previous), -np.sum(probs1 * current), np.max(probs0), np.max(probs1), np.sort(probs1)[-1] - np.sort(probs1)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)[None]
    rows = [np.concatenate([current, np.zeros(9, dtype=np.float32), [1.0]])]
    rows.extend(np.concatenate([imagined[index], descriptors[index], [0.0]]) for index in range(len(descriptors)))
    values = torch.zeros((1, max(31, len(rows)), 26), dtype=torch.float32)
    values[0, : len(rows)] = torch.from_numpy(np.asarray(rows, dtype=np.float32))
    mask = torch.zeros((1, values.shape[1]), dtype=torch.bool)
    mask[0, : len(rows)] = True
    with torch.inference_mode():
        scores, _ = model(torch.from_numpy(stats).to(device), values.to(device), mask.to(device))
    choice = tie_argmax(scores[0, : len(rows)].cpu().numpy())
    return None if choice == 0 else int(choice - 1)


__all__ = ["JointRevision", "N_CLASSES", "choose_next", "select_actions", "tie_argmax"]
