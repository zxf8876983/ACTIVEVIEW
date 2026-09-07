#!/usr/bin/env python3
"""Train and evaluate a deployable reduced14 H1 disambiguation ranker.

The ranker learns a candidate score from the observation available before H1:
the frozen s0 ST-GCN feature/posterior and the candidate's relative geometry.
Training targets are margins produced by the separately frozen history-identity
classifier on real archived candidate observations.  Those targets are used
only for supervision.  This script intentionally has no Test data path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import relative_view_descriptor
from activeview.scripts.eval.analyze_reduced14_h1_disambiguation import (
    _candidate_beliefs,
    _classification_metrics,
    _h1_orders,
    _label_names,
    _load_identity,
    _source_map,
)
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _validate_rows_cache,
)


SEED = 42
NUM_CLASSES = 14
STGCN_FEATURE_DIM = 256
LOGP_DIM = NUM_CLASSES
BASE_INPUT_DIM = STGCN_FEATURE_DIM + LOGP_DIM
DESCRIPTOR_DIM = 9
INPUT_DIM = BASE_INPUT_DIM + DESCRIPTOR_DIM
HIDDEN_DIM = 256
SECOND_HIDDEN_DIM = 128
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PAIR_COUNT = 16
DATASET_NAME = "policy_reduced14_kneel_eight_placement_v1"
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/h1_disambiguation_ranker"
CHECKPOINT_DIR = "activeview_reduced14_eight_placement_v1"
CHECKPOINT_NAME = "h1_disambiguation_ranker_best.pth"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RankerDataset(Dataset[dict[str, torch.Tensor]]):
    """One context per item, padded to the largest legal H1 candidate count."""

    def __init__(
        self,
        base: np.ndarray,
        descriptors: np.ndarray,
        utilities: np.ndarray,
        candidate_mask: np.ndarray,
        pair_indices: np.ndarray,
        pair_mask: np.ndarray,
    ) -> None:
        self.base = torch.from_numpy(np.asarray(base, dtype=np.float32))
        self.descriptors = torch.from_numpy(np.asarray(descriptors, dtype=np.float32))
        self.utilities = torch.from_numpy(np.asarray(utilities, dtype=np.float32))
        self.candidate_mask = torch.from_numpy(np.asarray(candidate_mask, dtype=bool))
        self.pair_indices = torch.from_numpy(np.asarray(pair_indices, dtype=np.int64))
        self.pair_mask = torch.from_numpy(np.asarray(pair_mask, dtype=bool))

    def __len__(self) -> int:
        return int(self.base.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "base": self.base[index],
            "descriptors": self.descriptors[index],
            "utilities": self.utilities[index],
            "candidate_mask": self.candidate_mask[index],
            "pair_indices": self.pair_indices[index],
            "pair_mask": self.pair_mask[index],
        }


class H1DisambiguationRanker(nn.Module):
    """Small candidate-conditioned MLP returning one score per candidate."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, SECOND_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(SECOND_HIDDEN_DIM, 1),
        )

    def forward(self, base: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        if base.ndim != 2 or descriptors.ndim != 3:
            raise ValueError("ranker expects base [B,270] and descriptors [B,K,9]")
        if base.size(0) != descriptors.size(0) or base.size(1) != BASE_INPUT_DIM:
            raise ValueError("ranker base shape mismatch")
        if descriptors.size(-1) != DESCRIPTOR_DIM:
            raise ValueError("ranker descriptor shape mismatch")
        repeated = base.unsqueeze(1).expand(-1, descriptors.size(1), -1)
        values = torch.cat([repeated, descriptors], dim=-1)
        return self.network(values).squeeze(-1)


def _load_split(data_root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if split not in {"train", "val"}:
        raise ValueError("Test is locked for H1 disambiguation ranker")
    root = data_root / "datasets" / DATASET_NAME
    rows = load_jsonl(root / "stage_d" / "features" / f"{split}.jsonl")
    cache = _load_npz(root / "counterfactual_cache" / f"{split}.npz")
    _validate_rows_cache(rows, cache, split)
    if any(str(row.get("policy_split", "")).lower() != split for row in rows):
        raise ValueError(f"{split} rows must explicitly carry policy_split={split}")
    expected = (len(rows), LOGP_DIM)
    for name in ("current_logp_s0",):
        if cache[name].shape != expected or not np.isfinite(cache[name]).all():
            raise ValueError(f"invalid {name} shape/value: {cache[name].shape}")
    if "label_id" not in cache or not np.array_equal(
        np.asarray(cache["label_id"], dtype=np.int64),
        np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64),
    ):
        raise ValueError(f"{split} cache labels are not aligned with rows")
    for row in rows:
        feature = np.asarray(row["s0_feature"], dtype=np.float32)
        if feature.shape != (STGCN_FEATURE_DIM + LOGP_DIM + 3,) or not np.isfinite(feature).all():
            raise ValueError(f"invalid s0 feature in {row['episode_id']}: {feature.shape}")
    return rows, cache


def _build_candidate_arrays(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    beliefs: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
) -> tuple[RankerDataset, dict[str, Any], list[np.ndarray]]:
    """Build padded context tensors and frozen discrimination-margin targets."""
    sources = _source_map(data_root, rows)
    max_candidates = max(len(orders[str(row["episode_id"])]) for row in rows)
    base = np.zeros((len(rows), BASE_INPUT_DIM), dtype=np.float32)
    descriptors = np.zeros((len(rows), max_candidates, DESCRIPTOR_DIM), dtype=np.float32)
    utilities = np.zeros((len(rows), max_candidates), dtype=np.float32)
    candidate_mask = np.zeros((len(rows), max_candidates), dtype=bool)
    pair_indices = np.zeros((len(rows), PAIR_COUNT, 2), dtype=np.int64)
    pair_mask = np.zeros((len(rows), PAIR_COUNT), dtype=bool)
    candidate_ids: list[np.ndarray] = []
    target_beliefs: list[np.ndarray] = []

    for index, row in enumerate(rows):
        episode_id = str(row["episode_id"])
        candidates = [int(value) for value in orders[episode_id]]
        if not candidates:
            raise ValueError(f"no legal H1 candidate for {episode_id}")
        source = Path(sources[(str(row["scene_id"]), str(row["region"]), str(row["record_id"]))])
        with np.load(source, allow_pickle=False) as archive:
            positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        by_id = {int(value): position for position, value in enumerate(viewpoint_ids.tolist())}
        s0_id = int(row["s0_viewpoint_id"])
        if s0_id not in by_id or any(candidate not in by_id for candidate in candidates):
            raise ValueError(f"viewpoint alignment failure for {episode_id}")
        current_position = positions[by_id[s0_id]]
        base[index] = np.concatenate([
            np.asarray(row["s0_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM],
            np.asarray(cache["current_logp_s0"][index], dtype=np.float32),
        ])
        belief = np.asarray(beliefs[index], dtype=np.float32)
        if belief.shape != (len(candidates), NUM_CLASSES) or not np.isfinite(belief).all():
            raise ValueError(f"invalid frozen history belief for {episode_id}: {belief.shape}")
        target_beliefs.append(belief)
        label = int(row["label_id"])
        log_belief = np.log(np.clip(belief, 1e-8, 1.0))
        other = log_belief.copy()
        other[:, label] = -np.inf
        target_utility = log_belief[:, label] - np.max(other, axis=1)
        if not np.isfinite(target_utility).all():
            raise ValueError(f"non-finite target utility for {episode_id}")
        utilities[index, : len(candidates)] = target_utility.astype(np.float32)
        candidate_mask[index, : len(candidates)] = True
        candidate_ids.append(np.asarray(candidates, dtype=np.int64))
        for candidate_index, candidate in enumerate(candidates):
            descriptors[index, candidate_index] = relative_view_descriptor(
                positions, current_position, candidate,
            )
        valid_pairs: list[tuple[int, int]] = []
        attempts = 0
        while len(valid_pairs) < PAIR_COUNT and attempts < PAIR_COUNT * 30:
            attempts += 1
            left, right = rng.choice(len(candidates), size=2, replace=False).tolist()
            if target_utility[left] == target_utility[right]:
                continue
            pair = (int(left), int(right))
            if pair not in valid_pairs:
                valid_pairs.append(pair)
        if valid_pairs:
            pair_indices[index, : len(valid_pairs)] = np.asarray(valid_pairs, dtype=np.int64)
            pair_mask[index, : len(valid_pairs)] = True
    dataset = RankerDataset(base, descriptors, utilities, candidate_mask, pair_indices, pair_mask)
    stats = {
        "contexts": len(rows),
        "candidate_hypothesis_samples": int(candidate_mask.sum()),
        "mean_candidates_per_context": float(candidate_mask.sum() / len(rows)),
        "pair_samples_per_context": int(pair_mask.sum() / len(rows)),
        "target_utility_mean": float(utilities[candidate_mask].mean()),
        "target_utility_std": float(utilities[candidate_mask].std()),
    }
    return dataset, stats, candidate_ids


def _train_epoch(
    model: H1DisambiguationRanker,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        base = batch["base"].to(device=device, dtype=torch.float32)
        descriptors = batch["descriptors"].to(device=device, dtype=torch.float32)
        pairs = batch["pair_indices"].to(device=device, dtype=torch.long)
        pair_valid = batch["pair_mask"].to(device=device, dtype=torch.bool)
        utility = batch["utilities"].to(device=device, dtype=torch.float32)
        scores = model(base, descriptors)
        left = scores.gather(1, pairs[:, :, 0])
        right = scores.gather(1, pairs[:, :, 1])
        target = (utility.gather(1, pairs[:, :, 0]) > utility.gather(1, pairs[:, :, 1])).float()
        pair_loss = nn.functional.binary_cross_entropy_with_logits(
            left - right, target, reduction="none",
        )
        loss = (pair_loss * pair_valid.float()).sum() / pair_valid.float().sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("ranker produced no training batches")
    return float(np.mean(losses))


def _rank_scores(
    model: H1DisambiguationRanker,
    dataset: RankerDataset,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            scores = model(
                batch["base"].to(device=device, dtype=torch.float32),
                batch["descriptors"].to(device=device, dtype=torch.float32),
            )
            output.append(scores.cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1]) if np.std(left_rank) and np.std(right_rank) else 0.0


def _selector_metrics(
    name: str,
    selected: Sequence[int],
    beliefs: Sequence[np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    s0_predictions: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    selected_beliefs = np.stack([beliefs[index][choice] for index, choice in enumerate(selected)])
    predictions = selected_beliefs.argmax(axis=1).astype(np.int64)
    entropy = -np.sum(selected_beliefs * np.log(np.clip(selected_beliefs, 1e-12, None)), axis=1)
    s0_wrong = s0_predictions != labels
    metrics = _classification_metrics(predictions, labels, label_names)
    return {
        "selector": name,
        "selected_count": len(selected),
        "stay_rate": 0.0,
        "mean_entropy": float(entropy.mean()),
        "s0_error_count": int(s0_wrong.sum()),
        "s0_error_correction_rate": float(np.mean(predictions[s0_wrong] == labels[s0_wrong])) if np.any(s0_wrong) else 0.0,
        "identity": metrics,
    }


def _select_privileged(
    beliefs: Sequence[np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    rng: np.random.Generator,
) -> list[int]:
    choices: list[int] = []
    for row, belief in zip(rows, beliefs):
        if mode == "random":
            choices.append(int(rng.integers(len(belief))))
        elif mode == "min_entropy":
            entropy = -np.sum(belief * np.log(np.clip(belief, 1e-12, None)), axis=1)
            choices.append(int(np.argmin(entropy)))
        elif mode == "max_margin":
            ordered = np.sort(belief, axis=1)
            choices.append(int(np.argmax(ordered[:, -1] - ordered[:, -2])))
        elif mode == "identity_oracle":
            label = int(row["label_id"])
            choices.append(int(np.argmax(belief[:, label])))
        else:
            raise ValueError(f"unknown selector mode: {mode}")
    return choices


def _frozen_indices(rows: Sequence[Mapping[str, Any]], candidate_ids: Sequence[np.ndarray]) -> list[int]:
    values: list[int] = []
    for row, candidates in zip(rows, candidate_ids):
        recorded = int(row["s1_viewpoint_id"])
        matches = np.flatnonzero(candidates == recorded)
        if matches.size != 1:
            raise ValueError(f"recorded s1 is not a unique legal H1 candidate: {row['episode_id']}")
        values.append(int(matches[0]))
    return values


def _ranker_selection(scores: np.ndarray, candidate_ids: Sequence[np.ndarray]) -> list[int]:
    return [int(np.argmax(scores[index, : len(ids)])) for index, ids in enumerate(candidate_ids)]


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    selectors = result["selectors"]
    lines = [
        "# Reduced14 H1 Disambiguation Ranker (Train/Val)",
        "",
        "The ranker sees only the frozen s0 ST-GCN feature/posterior and each candidate's 9-D relative geometry. Candidate utility targets are frozen history-identity discrimination margins computed from real archived observations and are used only during Train supervision.",
        "",
        f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}. Test was not read.",
        "",
        "## Val history identity after H1 selection",
        "",
        "| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("Frozen_current_H1", "Random_H1", "Min_entropy_H1", "Max_margin_H1", "Deployable_Disambiguation_Ranker", "IdentityOracle_H1"):
        item = selectors[name]
        lines.append(
            f"| {name} | {item['identity']['accuracy']:.6f} | {item['identity']['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |",
        )
    ranker = result["ranker_diagnostics"]
    lines.extend([
        "",
        "## Ranker diagnostics",
        "",
        f"Utility Pearson: {ranker['utility_pearson']:.6f}; utility Spearman: {ranker['utility_spearman']:.6f}; Top-1 oracle-positive hit: {ranker['top1_oracle_positive_hit']:.6f}.",
        f"Best checkpoint epoch: {result['training']['best_epoch']} (selected by Val history-identity Macro-F1).",
        "",
        "## Interpretation",
        "",
        f"Deployable ranker versus Frozen H1: Accuracy {selectors['Deployable_Disambiguation_Ranker']['identity']['accuracy'] - selectors['Frozen_current_H1']['identity']['accuracy']:+.6f}; Macro-F1 {selectors['Deployable_Disambiguation_Ranker']['identity']['macro_f1'] - selectors['Frozen_current_H1']['identity']['macro_f1']:+.6f}.",
        f"Privileged Min-entropy and Max-margin provide upper-bound candidate-choice references ({selectors['Min_entropy_H1']['identity']['accuracy']:.6f} and {selectors['Max_margin_H1']['identity']['accuracy']:.6f} Accuracy). IdentityOracle reaches {selectors['IdentityOracle_H1']['identity']['accuracy']:.6f}, so the remaining gap measures candidate selection difficulty under the frozen history representation.",
        "A positive ranker correlation and improvement over Frozen H1 would support predicting disambiguation value from currently observable state and geometry; a weak or negative result would indicate that this deployable input is insufficient for information-seeking H1 selection.",
        "",
        "Leakage audit: `test_used=false`; no Test path was loaded; ground-truth labels and frozen history-identity margins were used only to construct Train targets and offline Val references; no formal WM-E, JR or ST-GCN checkpoint was modified.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    started = time.perf_counter()
    data_root = data_root.resolve()
    train_rows, train_cache = _load_split(data_root, "train")
    val_rows, val_cache = _load_split(data_root, "val")
    train_orders = _h1_orders(data_root, train_rows)
    val_orders = _h1_orders(data_root, val_rows)
    print("building frozen Train candidate beliefs", flush=True)
    train_beliefs, train_inference = _candidate_beliefs(data_root, train_rows, train_cache, train_orders, device)
    print("building frozen Val candidate beliefs", flush=True)
    val_beliefs, val_inference = _candidate_beliefs(data_root, val_rows, val_cache, val_orders, device)
    train_dataset, train_stats, train_candidate_ids = _build_candidate_arrays(
        data_root, train_rows, train_cache, train_orders, train_beliefs,
        rng=np.random.default_rng(SEED),
    )
    val_dataset, val_stats, val_candidate_ids = _build_candidate_arrays(
        data_root, val_rows, val_cache, val_orders, val_beliefs,
        rng=np.random.default_rng(SEED),
    )
    label_names = _label_names(data_root)
    s0_predictions = np.asarray(val_cache["current_logp_s0"], dtype=np.float32).argmax(axis=1)
    frozen = _frozen_indices(val_rows, val_candidate_ids)
    random_selection = _select_privileged(val_beliefs, val_rows, "random", np.random.default_rng(SEED))
    min_entropy = _select_privileged(val_beliefs, val_rows, "min_entropy", np.random.default_rng(SEED))
    max_margin = _select_privileged(val_beliefs, val_rows, "max_margin", np.random.default_rng(SEED))
    identity_oracle = _select_privileged(val_beliefs, val_rows, "identity_oracle", np.random.default_rng(SEED))

    model = H1DisambiguationRanker().to(device)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_f1 = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_val_scores: np.ndarray | None = None
    for epoch in range(1, EPOCHS + 1):
        loss = _train_epoch(model, loader, optimizer, device)
        val_scores = _rank_scores(model, val_dataset, device)
        ranker_selection = _ranker_selection(val_scores, val_candidate_ids)
        ranker_metrics = _selector_metrics(
            "Deployable_Disambiguation_Ranker", ranker_selection, val_beliefs,
            val_rows, s0_predictions, label_names,
        )
        val_f1 = float(ranker_metrics["identity"]["macro_f1"])
        history.append({"epoch": epoch, "train_pairwise_loss": loss, "val_macro_f1": val_f1, "val_accuracy": float(ranker_metrics["identity"]["accuracy"])})
        print(f"h1 ranker epoch {epoch}/{EPOCHS} loss={loss:.6f} val_f1={val_f1:.6f}", flush=True)
        if val_f1 > best_val_f1:
            best_epoch = epoch
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_scores = val_scores.copy()
    if best_state is None or best_val_scores is None:
        raise RuntimeError("no best ranker checkpoint selected")
    checkpoint_path = data_root / "checkpoints" / CHECKPOINT_DIR / CHECKPOINT_NAME
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "experiment_id": "REDUCED14_H1_DISAMBIGUATION_RANKER",
        "epoch": best_epoch,
        "seed": SEED,
        "input_dim": INPUT_DIM,
    }, checkpoint_path)

    ranker_selection = _ranker_selection(best_val_scores, val_candidate_ids)
    selectors = {
        "Frozen_current_H1": _selector_metrics("Frozen_current_H1", frozen, val_beliefs, val_rows, s0_predictions, label_names),
        "Random_H1": _selector_metrics("Random_H1", random_selection, val_beliefs, val_rows, s0_predictions, label_names),
        "Min_entropy_H1": _selector_metrics("Min_entropy_H1", min_entropy, val_beliefs, val_rows, s0_predictions, label_names),
        "Max_margin_H1": _selector_metrics("Max_margin_H1", max_margin, val_beliefs, val_rows, s0_predictions, label_names),
        "Deployable_Disambiguation_Ranker": _selector_metrics("Deployable_Disambiguation_Ranker", ranker_selection, val_beliefs, val_rows, s0_predictions, label_names),
        "IdentityOracle_H1": _selector_metrics("IdentityOracle_H1", identity_oracle, val_beliefs, val_rows, s0_predictions, label_names),
    }
    target_values = np.concatenate([
        np.asarray(belief, dtype=np.float64)[:, int(row["label_id"])]
        - np.max(np.delete(np.asarray(belief, dtype=np.float64), int(row["label_id"]), axis=1), axis=1)
        for row, belief in zip(val_rows, val_beliefs)
    ])
    score_values = np.concatenate([
        best_val_scores[index, : len(candidate_ids)].astype(np.float64)
        for index, candidate_ids in enumerate(val_candidate_ids)
    ])
    oracle_matches = np.asarray([
        int(choice == int(np.argmax(target_values[offset: offset + len(candidate_ids)])))
        for offset, (choice, candidate_ids) in zip(
            np.cumsum([0] + [len(ids) for ids in val_candidate_ids[:-1]]),
            zip(ranker_selection, val_candidate_ids),
        )
    ], dtype=np.float32)
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_H1_DISAMBIGUATION_RANKER",
        "status": "COMPLETED",
        "split": "train_val",
        "test_used": False,
        "population": {
            "train_contexts": len(train_rows),
            "val_moving_contexts": len(val_rows),
            "train_candidate_hypothesis_samples": train_stats["candidate_hypothesis_samples"],
            "val_candidate_hypothesis_samples": val_stats["candidate_hypothesis_samples"],
            "train_mean_candidates_per_context": train_stats["mean_candidates_per_context"],
            "val_mean_candidates_per_context": val_stats["mean_candidates_per_context"],
        },
        "selectors": selectors,
        "ranker_diagnostics": {
            "utility_pearson": float(np.corrcoef(score_values, target_values)[0, 1]) if np.std(score_values) and np.std(target_values) else 0.0,
            "utility_spearman": _spearman(score_values, target_values),
            "top1_oracle_positive_hit": float(oracle_matches.mean()),
            "top1_oracle_positive_hit_count": int(oracle_matches.sum()),
            "candidate_identity_matches": int(sum(len(ids) == len(set(ids.tolist())) for ids in val_candidate_ids)),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "pair_count": PAIR_COUNT,
            "seed": SEED,
            "architecture": f"Linear({INPUT_DIM},256) -> GELU -> Linear(256,128) -> GELU -> Linear(128,1)",
            "loss": "pairwise logistic BCE on within-context target utility margins",
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "history": history,
            "train_target_stats": train_stats,
            "val_target_stats": val_stats,
        },
        "inference": {
            "train_frozen_history_inference": train_inference,
            "val_frozen_history_inference": val_inference,
        },
        "artifacts": {
            "ranker_checkpoint": str(checkpoint_path.resolve()),
            "ranker_checkpoint_sha256": _sha256(checkpoint_path),
            "identity_checkpoint": str((data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth").resolve()),
            "stgcn_checkpoint": str((data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth").resolve()),
            "train_features": str((data_root / "datasets" / DATASET_NAME / "stage_d/features/train.jsonl").resolve()),
            "val_features": str((data_root / "datasets" / DATASET_NAME / "stage_d/features/val.jsonl").resolve()),
        },
        "protocol": {
            "deployable_input": "frozen s0 ST-GCN feature 256-D + s0 logp 14-D + candidate relative descriptor 9-D",
            "target": "log p_history(y_gt|s0,v) - max_{c!=y_gt} log p_history(c|s0,v)",
            "candidate_source": "all legal candidate_order viewpoints from s0, excluding s0",
            "selector": "argmax predicted disambiguation utility over legal H1 candidates",
            "checkpoint_selection": "Val moving history-identity Macro-F1",
        },
        "leakage_flags": {
            "test_used": False,
            "future_observation_used_as_input": False,
            "true_label_used_in_deployable_input": False,
            "true_label_used_only_for_training_target_and_identity_oracle": True,
            "formal_checkpoint_modified": False,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reduced14 H1 disambiguation ranker on Train and evaluate Val")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H1 disambiguation ranker requires CUDA; CPU fallback is disabled")
    analyze(args.data_root, device)


if __name__ == "__main__":
    main()
