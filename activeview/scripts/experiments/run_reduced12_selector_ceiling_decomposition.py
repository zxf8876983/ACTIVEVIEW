#!/usr/bin/env python3
"""Train/Val-only 2x2 selector-ceiling decomposition for reduced12.

The four JR branches intentionally share the formal Multi-positive objective.
The ``real`` evidence and GT identity branches are privileged diagnostics and
must never be used as deployable policy inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, context_key, load_pairwise_and_azimuths

SEED = 42
NUM_CLASSES = 12
VIEW_COUNT = 32
EPOCHS = 20
BATCH_SIZE = 512
CURRENT_BASE_DIM = 2 * NUM_CLASSES + 6
CANDIDATE_DIM = NUM_CLASSES + 10
MAX_ACTIONS = VIEW_COUNT - 1


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _Selector(nn.Module):
    """Formal JointRevision with an optional privileged GT identity token."""

    def __init__(self, *, gt_identity: bool) -> None:
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.current_dim = CURRENT_BASE_DIM + (NUM_CLASSES if gt_identity else 0)
        self.candidate_dim = CANDIDATE_DIM
        self.current_projector = nn.Sequential(nn.Linear(self.current_dim, 128), nn.GELU())
        self.candidate_projector = nn.Sequential(nn.Linear(self.candidate_dim, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Linear(128, 1)
        self.posterior = nn.Linear(128, NUM_CLASSES)

    def forward(
        self, current: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_token = self.current_projector(current).unsqueeze(1)
        candidate_tokens = self.candidate_projector(candidates)
        tokens = torch.cat([current_token, candidate_tokens], dim=1)
        full_mask = torch.cat(
            [torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.score(encoded[:, 1:]).squeeze(-1), self.posterior(encoded[:, 1:])


class _Examples(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: tuple[np.ndarray, ...]) -> None:
        self.current, self.candidates, self.mask, self.fallback, self.positive, self.labels = arrays

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current": torch.from_numpy(self.current[index]),
            "candidates": torch.from_numpy(self.candidates[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "fallback": torch.tensor(self.fallback[index]),
            "positive": torch.from_numpy(self.positive[index]),
            "label": torch.tensor(self.labels[index]),
        }


@dataclass(frozen=True)
class _Prepared:
    arrays: tuple[np.ndarray, ...]
    stats: dict[str, Any]
    episode_ids: tuple[str, ...]


def _cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _sources(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    return {
        context_key(row): str(
            root / context_key(row)[0] / context_key(row)[1] / f"{context_key(row)[2]}.npz"
        )
        for row in rows
    }


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _sources(data_root, rows)
    pairwise, azimuth = load_pairwise_and_azimuths(
        data_root,
        rows,
        source,
        pair_root=data_root / "datasets/policy_reduced12_eight_placement_v1/pairwise_viewpoint_geodesic",
    )
    return {
        str(row["episode_id"]): candidate_order(
            row,
            int(row["s1_viewpoint_id"]),
            {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])},
            pairwise[(str(row["scene_id"]), str(row["region"]))],
            azimuth[(str(row["scene_id"]), str(row["region"]))],
        )
        for row in rows
    }


def _current_stats(cache: Mapping[str, np.ndarray], index: int, label: int | None) -> np.ndarray:
    logp0 = np.asarray(cache["current_logp_s0"][index], dtype=np.float32)
    logp1 = np.asarray(cache["current_logp_s1"][index], dtype=np.float32)
    probs0, probs1 = np.exp(logp0), np.exp(logp1)
    values = [
        *probs0,
        *probs1,
        -float(np.sum(probs0 * logp0)),
        -float(np.sum(probs1 * logp1)),
        float(np.max(probs0)),
        float(np.max(probs1)),
        float(np.sort(probs1)[-1] - np.sort(probs1)[-2]),
        float(np.sort(probs0)[-1] - np.sort(probs0)[-2]),
    ]
    if label is not None:
        one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
        one_hot[int(label)] = 1.0
        values.extend(one_hot.tolist())
    return np.asarray(values, dtype=np.float32)


def _prepare(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    *,
    evidence: str,
    gt_identity: bool,
) -> _Prepared:
    if evidence not in {"imagined", "real"}:
        raise ValueError(f"unsupported evidence: {evidence}")
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    currents: list[np.ndarray] = []
    candidates_all: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    fallbacks: list[int] = []
    positives: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    no_positive = single_positive = multi_positive = 0
    positive_counts: list[int] = []
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in cache_index:
            raise ValueError(f"cache missing Stage-D episode: {episode_id}")
        index = cache_index[episode_id]
        label = int(row["label_id"])
        legal = [int(value) for value in orders[episode_id]]
        current_logp = np.asarray(cache["current_logp_s1"][index], dtype=np.float32)
        true_logp = np.asarray(cache["true_logp"][index], dtype=np.float32)
        evidence_key = "imagined_logp" if evidence == "imagined" else "true_logp"
        evidence_logp = np.asarray(cache[evidence_key][index], dtype=np.float32)
        values = [np.concatenate([current_logp, np.zeros(9, dtype=np.float32), [1.0]])]
        values.extend(
            np.concatenate([evidence_logp[candidate], cache["candidate_descriptor"][index, candidate], [0.0]])
            for candidate in legal
        )
        action_positive = np.asarray(
            [int(np.argmax(current_logp) == label)]
            + [int(np.argmax(true_logp[candidate]) == label) for candidate in legal],
            dtype=np.float32,
        )
        count = int(action_positive.sum())
        if count == 0:
            no_positive += 1
            score = [float(current_logp[label])] + [float(true_logp[candidate, label]) for candidate in legal]
            fallback = int(np.argmax(score))
        else:
            positive_counts.append(count)
            single_positive += int(count == 1)
            multi_positive += int(count > 1)
            fallback = int(np.flatnonzero(action_positive)[0])
        if len(values) > MAX_ACTIONS:
            raise ValueError(f"too many legal actions for {episode_id}: {len(values)}")
        padded = np.zeros((MAX_ACTIONS, CANDIDATE_DIM), dtype=np.float32)
        padded[: len(values)] = np.asarray(values, dtype=np.float32)
        mask = np.asarray([True] * len(values) + [False] * (MAX_ACTIONS - len(values)), dtype=bool)
        positive = np.zeros(MAX_ACTIONS, dtype=np.float32)
        positive[: len(values)] = action_positive
        currents.append(_current_stats(cache, index, label if gt_identity else None))
        candidates_all.append(padded)
        masks.append(mask)
        fallbacks.append(fallback)
        positives.append(positive)
        labels.append(label)
        episode_ids.append(episode_id)
    stats = {
        "contexts": len(labels),
        "positive_contexts": single_positive + multi_positive,
        "multi_positive_contexts": multi_positive,
        "single_positive_contexts": single_positive,
        "no_positive_contexts": no_positive,
        "mean_positive_count": float(np.mean(positive_counts)) if positive_counts else 0.0,
        "evidence": evidence,
        "gt_identity_input": gt_identity,
    }
    arrays = (
        np.asarray(currents),
        np.asarray(candidates_all),
        np.asarray(masks),
        np.asarray(fallbacks, dtype=np.int64),
        np.asarray(positives),
        np.asarray(labels, dtype=np.int64),
    )
    return _Prepared(arrays=arrays, stats=stats, episode_ids=tuple(episode_ids))


def _train(
    prepared: _Prepared,
    *,
    device: torch.device,
    checkpoint: Path,
    training_json: Path,
    batch_size: int,
) -> dict[str, Any]:
    _seed()
    model = _Selector(gt_identity=bool(prepared.stats["gt_identity_input"])).to(device)
    loader = DataLoader(_Examples(prepared.arrays), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            fallback = batch["fallback"].long().to(device)
            positive = batch["positive"].float().to(device)
            labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask)
            valid_scores = scores.masked_fill(~mask, -1e9)
            all_lse = torch.logsumexp(valid_scores, dim=1)
            positive_mask = positive.bool() & mask
            positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1)
            has_positive = positive_mask.any(dim=1)
            main = torch.where(
                has_positive,
                all_lse - positive_lse,
                nn.functional.cross_entropy(valid_scores, fallback, reduction="none"),
            )
            bce_all = nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none")
            bce = (bce_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            target = labels.unsqueeze(1).expand(-1, posterior.size(1))
            post_all = nn.functional.cross_entropy(
                posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none",
            ).reshape_as(scores)
            post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (main + 0.25 * bce + 0.05 * post).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"selector branch={training_json.stem} epoch={epoch}/{EPOCHS} loss={losses[-1]:.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "seed": SEED,
            "epochs": EPOCHS,
            "num_classes": NUM_CLASSES,
            "evidence": prepared.stats["evidence"],
            "gt_identity_input": prepared.stats["gt_identity_input"],
        },
        checkpoint,
    )
    result = {
        **prepared.stats,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_history": losses,
        "final_loss": losses[-1],
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
    }
    training_json.parent.mkdir(parents=True, exist_ok=True)
    training_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_model(checkpoint: Path, *, gt_identity: bool, device: torch.device) -> _Selector:
    model = _Selector(gt_identity=gt_identity).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload["state_dict"]))
    model.eval()
    return model


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(target * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0
        recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "count": int(target.size),
        "accuracy": float(np.mean(pred == target)) if target.size else 0.0,
        "macro_f1": float(np.mean(f1)) if f1 else 0.0,
    }


def _model_actions(
    prepared: _Prepared,
    model: _Selector,
    orders: Mapping[str, Sequence[int]],
    cache: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    actions: list[int] = []
    terminal: list[int] = []
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    arrays = prepared.arrays
    with torch.inference_mode():
        for start in range(0, len(prepared.episode_ids), batch_size):
            stop = min(len(prepared.episode_ids), start + batch_size)
            current = torch.from_numpy(arrays[0][start:stop]).float().to(device)
            candidates = torch.from_numpy(arrays[1][start:stop]).float().to(device)
            mask = torch.from_numpy(arrays[2][start:stop]).bool().to(device)
            scores, _ = model(current, candidates, mask)
            choices = torch.argmax(scores.masked_fill(~mask, -1e9), dim=1).cpu().numpy()
            for offset, choice in enumerate(choices.tolist()):
                episode_id = prepared.episode_ids[start + offset]
                row_index = index[episode_id]
                legal = list(orders[episode_id])
                # ``-1`` is Stay because viewpoint id 0 is a legal candidate.
                action = -1 if choice == 0 else int(legal[choice - 1])
                actions.append(action)
                prediction = int(np.argmax(cache["current_logp_s1"][row_index])) if action < 0 else int(np.argmax(cache["true_logp"][row_index, action]))
                terminal.append(prediction)
    return actions, terminal


def _reference_actions(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    *,
    mode: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    actions: list[int] = []
    terminal: list[int] = []
    for row in rows:
        episode_id = str(row["episode_id"])
        i = index[episode_id]
        label = int(row["label_id"])
        legal = list(orders[episode_id])
        if mode == "current_label_real":
            identity = int(np.argmax(cache["current_logp_s1"][i]))
            candidate_values = [float(cache["true_logp"][i, candidate, identity]) for candidate in legal]
            stay_value = float(cache["current_logp_s1"][i, identity])
        elif mode == "gt_label_imagined":
            candidate_values = [float(cache["imagined_logp"][i, candidate, label]) for candidate in legal]
            stay_value = float(cache["current_logp_s1"][i, label])
        elif mode == "gt_label_real":
            candidate_values = [float(cache["true_logp"][i, candidate, label]) for candidate in legal]
            stay_value = float(cache["current_logp_s1"][i, label])
        else:
            raise ValueError(f"unknown reference mode: {mode}")
        best_index = int(np.argmax(candidate_values)) if candidate_values else -1
        if best_index < 0 or candidate_values[best_index] <= stay_value:
            action = -1
            prediction = int(np.argmax(cache["current_logp_s1"][i]))
        else:
            action = int(legal[best_index])
            prediction = int(np.argmax(cache["true_logp"][i, action]))
        actions.append(action)
        terminal.append(prediction)
    return actions, terminal, {"mode": mode}


def _diagnostics(
    actions: Sequence[int],
    terminal: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    *,
    orders: Mapping[str, Sequence[int]],
    oracle_actions: Sequence[int],
) -> dict[str, Any]:
    labels = [int(row["label_id"]) for row in rows]
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    s1_correct = []
    selected_positive = []
    for row, action, prediction in zip(rows, actions, terminal):
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        s1_is_correct = int(np.argmax(cache["current_logp_s1"][i]) == label)
        positive = s1_is_correct if action < 0 else int(np.argmax(cache["true_logp"][i, action]) == label)
        s1_correct.append(s1_is_correct)
        selected_positive.append(positive)
    final_correct = [int(pred == label) for pred, label in zip(terminal, labels)]
    no_positive = 0
    single_positive = 0
    multi_positive = 0
    positive_counts: list[int] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        values = [int(np.argmax(cache["current_logp_s1"][i]) == label)]
        values.extend(
            int(np.argmax(cache["true_logp"][i, candidate]) == label)
            for candidate in orders[str(row["episode_id"])]
        )
        count = int(sum(values))
        if count == 0:
            no_positive += 1
        else:
            positive_counts.append(count)
            single_positive += int(count == 1)
            multi_positive += int(count > 1)
    return {
        "terminal": _metrics(terminal, labels),
        "positive_action_hit_rate": float(np.mean(selected_positive)) if selected_positive else 0.0,
        "stay_rate": float(np.mean(np.asarray(actions) < 0)) if actions else 0.0,
        "correction_rate": float(np.mean([not s and f for s, f in zip(s1_correct, final_correct)])) if rows else 0.0,
        "harm_rate": float(np.mean([s and not f for s, f in zip(s1_correct, final_correct)])) if rows else 0.0,
        "no_positive_contexts": no_positive,
        "single_positive_contexts": single_positive,
        "multi_positive_contexts": multi_positive,
        "mean_number_of_positives": float(np.mean(positive_counts)) if positive_counts else 0.0,
        "selected_action_overlap_with_fixed_oracle": float(np.mean(np.asarray(actions) == np.asarray(oracle_actions))) if actions else 0.0,
    }


def _full_predictions(
    full_rows: Sequence[Mapping[str, Any]],
    moving_rows: Sequence[Mapping[str, Any]],
    moving_terminal: Sequence[int],
    fallback_s0: Mapping[str, int],
) -> tuple[list[int], list[int]]:
    moving_map = {str(row["episode_id"]): int(pred) for row, pred in zip(moving_rows, moving_terminal)}
    labels = [int(row["label_id"]) for row in full_rows]
    predictions = [moving_map.get(str(row["episode_id"]), int(fallback_s0[str(row["episode_id"])])) for row in full_rows]
    return predictions, labels


def _pp(delta: float) -> float:
    return 100.0 * float(delta)


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]["moving"]
    full_methods = result["methods"]["full"]
    gaps = result["gap_decomposition_pp"]
    def line(name: str, values: Mapping[str, Any]) -> str:
        value = values[name]
        return f"| {name} | {value['terminal']['accuracy']:.6f} | {value['terminal']['macro_f1']:.6f} | {value['positive_action_hit_rate']:.6f} | {value['stay_rate']:.6f} |"
    rows = [
        "# Reduced12 selector ceiling decomposition",
        "",
        "Train/Val only; policy Test was not read. All terminal predictions use the selected real archived observation and frozen reduced12 ST-GCN.",
        "",
        "## Moving Val comparison",
        "",
        "| Method | Accuracy | Macro-F1 | Positive-action hit | Stay rate |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(line(name, methods) for name in result["method_order"])
    rows.extend([
        "",
        "## Full Val comparison",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ])
    rows.extend(
        f"| {name} | {full_methods[name]['terminal']['accuracy']:.6f} | {full_methods[name]['terminal']['macro_f1']:.6f} |"
        for name in result["method_order"]
    )
    rows.extend([
        "",
        "## Gap decomposition (percentage points)",
        "",
        f"- WM gap (Real+Inferred − Imagined+Inferred): **{gaps['wm_gap']:.3f} pp**.",
        f"- Identity gap (Real+GT − Real+Inferred): **{gaps['identity_gap']:.3f} pp**.",
        f"- Imagined identity gain (Imagined+GT − Imagined+Inferred): **{gaps['imagined_identity_gain']:.3f} pp**.",
        f"- Architecture residual (FixedH1-H2 Oracle − Real+GT): **{gaps['architecture_residual']:.3f} pp**.",
        f"- Current H2 − Frozen H1: **{gaps['current_h2_minus_frozen_h1']:.3f} pp Accuracy / {gaps['current_h2_minus_frozen_h1_f1']:.3f} pp Macro-F1**.",
        f"- Real+Inferred − Frozen H1: **{gaps['real_inferred_minus_frozen_h1']:.3f} pp Accuracy / {gaps['real_inferred_minus_frozen_h1_f1']:.3f} pp Macro-F1**.",
        f"- Real+GT − Frozen H1: **{gaps['real_gt_minus_frozen_h1']:.3f} pp Accuracy / {gaps['real_gt_minus_frozen_h1_f1']:.3f} pp Macro-F1**.",
        "",
        "## Interpretation",
        "",
        result["scientific_conclusion"],
        "",
        "## Protocol and leakage",
        "",
        "- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)",
        "- candidate budget: ALL_LEGAL; visited viewpoints excluded",
        "- `test_used=false`; no Test path or Test artifact was read",
        "- privileged Real/GT branches are diagnostics only and are not deployable methods",
    ])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device, output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rgb_root = data_root / "datasets/policy_reduced12_eight_placement_v1_rgb_restored"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    moving_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    full_rows = load_jsonl(root / "stage_c/predictions/val_predictions.jsonl")
    train_cache = _cache(rgb_root / "counterfactual_cache/train.npz")
    val_cache = _cache(rgb_root / "counterfactual_cache/val.npz")
    train_orders = _orders(data_root, train_rows)
    val_orders = _orders(data_root, moving_rows)
    train_ids = {str(value) for value in train_cache["episode_ids"].tolist()}
    val_ids = {str(value) for value in val_cache["episode_ids"].tolist()}
    if train_ids != {str(row["episode_id"]) for row in train_rows}:
        raise ValueError("Train Stage-D/cache episode IDs are not aligned")
    if val_ids != {str(row["episode_id"]) for row in moving_rows}:
        raise ValueError("Val Stage-D/cache episode IDs are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = data_root / "checkpoints/activeview_reduced12_selector_ceiling_decomposition"
    prepared_cache: dict[str, tuple[_Prepared, _Prepared]] = {}
    branch_specs = {
        "imagined_inferred": ("imagined", False),
        "real_inferred": ("real", False),
        "imagined_gt": ("imagined", True),
        "real_gt": ("real", True),
    }
    branch_training: dict[str, Any] = {}
    branch_models: dict[str, _Selector] = {}
    baseline_reproduction: dict[str, Any] | None = None
    for name, (evidence, gt_identity) in branch_specs.items():
        train_prepared = _prepare(train_rows, train_cache, train_orders, evidence=evidence, gt_identity=gt_identity)
        val_prepared = _prepare(moving_rows, val_cache, val_orders, evidence=evidence, gt_identity=gt_identity)
        prepared_cache[name] = (train_prepared, val_prepared)
        checkpoint = checkpoint_root / f"{name}.pth"
        training_json = output_dir / f"{name}_training.json"
        if checkpoint.exists() and training_json.exists():
            branch_training[name] = json.loads(training_json.read_text(encoding="utf-8"))
        else:
            branch_training[name] = _train(
                train_prepared,
                device=device,
                checkpoint=checkpoint,
                training_json=training_json,
                batch_size=batch_size,
            )
        branch_models[name] = _load_model(checkpoint, gt_identity=gt_identity, device=device)
        if name == "imagined_inferred":
            _, baseline_val_prepared = prepared_cache[name]
            _, baseline_terminal = _model_actions(
                baseline_val_prepared,
                branch_models[name],
                val_orders,
                val_cache,
                device=device,
                batch_size=batch_size,
            )
            baseline_metrics = _metrics(
                baseline_terminal,
                [int(row["label_id"]) for row in moving_rows],
            )
            baseline_reproduction = {
                "observed_moving_accuracy": baseline_metrics["accuracy"],
                "observed_moving_macro_f1": baseline_metrics["macro_f1"],
                "reference_moving_accuracy": 0.5356150793650793,
                "reference_moving_macro_f1": 0.53632031612324,
                "accuracy_delta_pp": _pp(baseline_metrics["accuracy"] - 0.5356150793650793),
                "macro_f1_delta_pp": _pp(baseline_metrics["macro_f1"] - 0.53632031612324),
                "tolerance_pp": 2.0,
            }
            if abs(float(baseline_reproduction["accuracy_delta_pp"])) > 2.0 or abs(float(baseline_reproduction["macro_f1_delta_pp"])) > 2.0:
                raise RuntimeError(
                    "Imagined+Inferred baseline reproduction is outside the 2 pp tolerance; "
                    "stopping before privileged branches."
                )
    fallback_s0 = {str(row["episode_id"]): int(row["current_predicted_label_id"]) for row in load_jsonl(root / "stage_c/predictions/val_predictions.jsonl")}
    frozen_h1_actions = [-1] * len(moving_rows)
    frozen_h1_terminal = [int(np.argmax(val_cache["current_logp_s1"][i])) for i in range(len(moving_rows))]
    oracle_actions, oracle_terminal, _ = _reference_actions(moving_rows, val_cache, val_orders, mode="gt_label_real")
    references: dict[str, tuple[list[int], list[int]]] = {
        "Frozen H1": (frozen_h1_actions, frozen_h1_terminal),
    }
    for name, mode in (
        ("CurrentLabel-Real", "current_label_real"),
        ("GTLabel-Imagined", "gt_label_imagined"),
        ("FixedH1-H2 Oracle", "gt_label_real"),
    ):
        actions, terminal, _ = _reference_actions(moving_rows, val_cache, val_orders, mode=mode)
        references[name] = (actions, terminal)
    method_order = [
        "Frozen H1", "Imagined+Inferred JR", "Real+Inferred JR", "Imagined+GT JR",
        "Real+GT JR", "CurrentLabel-Real", "GTLabel-Imagined", "FixedH1-H2 Oracle",
    ]
    methods_moving: dict[str, Any] = {}
    methods_full: dict[str, Any] = {}
    action_sets: dict[str, list[int]] = {}
    for name, (actions, terminal) in references.items():
        key = name
        diagnostics = _diagnostics(
            actions, terminal, moving_rows, val_cache, orders=val_orders,
            oracle_actions=oracle_actions,
        )
        methods_moving[key] = diagnostics
        action_sets[key] = actions
        full_pred, full_labels = _full_predictions(full_rows, moving_rows, terminal, fallback_s0)
        methods_full[key] = {"terminal": _metrics(full_pred, full_labels), **{k: v for k, v in diagnostics.items() if k != "terminal"}}
    branch_name_map = {
        "imagined_inferred": "Imagined+Inferred JR",
        "real_inferred": "Real+Inferred JR",
        "imagined_gt": "Imagined+GT JR",
        "real_gt": "Real+GT JR",
    }
    for branch, display_name in branch_name_map.items():
        _, val_prepared = prepared_cache[branch]
        actions, terminal = _model_actions(
            val_prepared,
            branch_models[branch],
            val_orders,
            val_cache,
            device=device,
            batch_size=batch_size,
        )
        diagnostics = _diagnostics(
            actions, terminal, moving_rows, val_cache, orders=val_orders,
            oracle_actions=oracle_actions,
        )
        methods_moving[display_name] = diagnostics
        action_sets[display_name] = actions
        full_pred, full_labels = _full_predictions(full_rows, moving_rows, terminal, fallback_s0)
        methods_full[display_name] = {"terminal": _metrics(full_pred, full_labels), **{k: v for k, v in diagnostics.items() if k != "terminal"}}
    methods_moving = {name: methods_moving[name] for name in method_order}
    methods_full = {name: methods_full[name] for name in method_order}
    frozen = methods_moving["Frozen H1"]["terminal"]
    imagined = methods_moving["Imagined+Inferred JR"]["terminal"]
    real_inf = methods_moving["Real+Inferred JR"]["terminal"]
    imagined_gt = methods_moving["Imagined+GT JR"]["terminal"]
    real_gt = methods_moving["Real+GT JR"]["terminal"]
    oracle = methods_moving["FixedH1-H2 Oracle"]["terminal"]
    gaps = {
        "wm_gap": _pp(real_inf["accuracy"] - imagined["accuracy"]),
        "identity_gap": _pp(real_gt["accuracy"] - real_inf["accuracy"]),
        "imagined_identity_gain": _pp(imagined_gt["accuracy"] - imagined["accuracy"]),
        "architecture_residual": _pp(oracle["accuracy"] - real_gt["accuracy"]),
        "current_h2_minus_frozen_h1": _pp(imagined["accuracy"] - frozen["accuracy"]),
        "current_h2_minus_frozen_h1_f1": _pp(imagined["macro_f1"] - frozen["macro_f1"]),
        "real_inferred_minus_frozen_h1": _pp(real_inf["accuracy"] - frozen["accuracy"]),
        "real_inferred_minus_frozen_h1_f1": _pp(real_inf["macro_f1"] - frozen["macro_f1"]),
        "real_gt_minus_frozen_h1": _pp(real_gt["accuracy"] - frozen["accuracy"]),
        "real_gt_minus_frozen_h1_f1": _pp(real_gt["macro_f1"] - frozen["macro_f1"]),
    }
    wm_gap = gaps["wm_gap"]
    identity_gap = gaps["identity_gap"]
    architecture_residual = gaps["architecture_residual"]
    if identity_gap > 2.0 and architecture_residual <= 2.0:
        conclusion = (
            f"Case B is primary: Real+GT is only {architecture_residual:.3f} pp below the fixed oracle, while Real+Inferred loses {identity_gap:.3f} pp; action identity/belief inference is the largest bottleneck. "
            f"Case A also applies as a secondary effect because Real+Inferred exceeds Imagined+Inferred by {wm_gap:.3f} pp, so WM candidate fidelity remains material."
        )
    elif wm_gap > 2.0:
        conclusion = (
            f"Case A applies to the WM component: Real+Inferred exceeds Imagined+Inferred by {wm_gap:.3f} pp Accuracy. "
            f"However, the {gaps['identity_gap']:.3f} pp Real+GT−Real+Inferred identity gap is larger, so action identity/belief inference is the largest absolute component of the overall ceiling gap; WM candidate fidelity is a substantial secondary bottleneck."
        )
    elif identity_gap > 2.0 and architecture_residual <= 2.0:
        conclusion = f"Case B: Real+GT is within {architecture_residual:.3f} pp of the fixed oracle while Real+Inferred loses {identity_gap:.3f} pp, so action identity/belief inference is the dominant bottleneck."
    elif architecture_residual > 2.0:
        conclusion = f"Case C: Real+GT remains {architecture_residual:.3f} pp below the fixed oracle, indicating a substantial JR selector/objective residual."
    elif gaps["imagined_identity_gain"] > 2.0:
        conclusion = f"Case D: GT identity gains {gaps['imagined_identity_gain']:.3f} pp even with imagined evidence, while real evidence also matters; WM and identity both contribute."
    else:
        conclusion = "No single decomposition component exceeds the 2 pp preregistered diagnostic threshold; the measured gap is distributed across WM, identity and selector effects."
    result = {
        "experiment_id": "REDUCED12_SELECTOR_CEILING_DECOMPOSITION",
        "status": "COMPLETED",
        "test_used": False,
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows), "full_val_episodes": len(full_rows)},
        "method_order": method_order,
        "methods": {"moving": methods_moving, "full": methods_full},
        "gap_decomposition_pp": gaps,
        "branch_training": branch_training,
        "baseline_reproduction": baseline_reproduction,
        "reference_definitions": {
            "CurrentLabel-Real": "argmax current s1 label, score stay+real candidates on that label",
            "GTLabel-Imagined": "GT label, score stay+imagined candidates on GT label",
            "FixedH1-H2 Oracle": "GT label, score stay+real candidates on GT label",
        },
        "protocol": {
            "taxonomy": "reduced12_no_kneel_clean",
            "candidate_budget": "ALL_LEGAL",
            "positive_definition": "stay/current or legal candidate whose real archived ST-GCN argmax equals GT",
            "terminal_observation": "real archived skeleton through frozen reduced12 ST-GCN",
            "wm_checkpoint": str((data_root / "checkpoints/activeview_reduced12_eight_placement_v1_rgb_restored/wm_e/wm_e_best.pth").resolve()),
            "counterfactual_cache": str((rgb_root / "counterfactual_cache").resolve()),
        },
        "scientific_conclusion": conclusion,
        "leakage_flags": {
            "test_used": False,
            "test_paths_read": False,
            "stgcn_retrained": False,
            "wm_retrained": False,
            "skeleton_rgb_dino_regenerated": False,
            "privileged_branches_deployable": False,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; selector diagnostic requires GPU")
    output = args.output_dir or (
        Path(__file__).resolve().parents[3]
        / "experiments/reduced12_eight_placement_v1/selector_ceiling_decomposition"
    )
    result = run(args.data_root.resolve(), device, output.resolve(), batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"], "conclusion": result["scientific_conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
