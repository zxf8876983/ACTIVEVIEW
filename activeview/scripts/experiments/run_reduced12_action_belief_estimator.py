#!/usr/bin/env python3
"""Train a visited-history action belief and evaluate two JR branches.

Only reduced12 Train/Val moving contexts are read. Existing frozen caches and
checkpoints are reused; no Test, perception, RGB or DINO regeneration occurs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.action_belief import ActionBeliefEstimator
from activeview.scripts.experiments.run_reduced12_selector_ceiling_decomposition import (
    _Examples,
    _Prepared,
    _cache,
    _current_stats,
    _diagnostics,
    _full_predictions,
    _metrics,
    _model_actions,
    _orders,
    _prepare,
    _reference_actions,
)

SEED = 42
NUM_CLASSES = 12
HISTORY_FEATURE_DIM = 256
DINO_DIM = 768
HISTORY_INPUT_DIM = HISTORY_FEATURE_DIM + NUM_CLASSES + DINO_DIM
JR_CURRENT_DIM = 2 * NUM_CLASSES + 6 + NUM_CLASSES
JR_CANDIDATE_DIM = NUM_CLASSES + 10
EPOCHS = 20
BATCH_SIZE = 512


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class _BeliefJointRevision(nn.Module):
    """Formal JR architecture with a 12-D learned visited-history belief."""

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.current_dim = JR_CURRENT_DIM
        self.candidate_dim = JR_CANDIDATE_DIM
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


def _load_dino_store(data_root: Path) -> tuple[dict[tuple[str, str, str, int], int], np.ndarray, dict[str, Any]]:
    cache_dir = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history"
    manifest_path = cache_dir / "manifest.jsonl"
    embeddings = np.asarray(np.load(cache_dir / "embeddings.npy", mmap_mode="r"))
    lookup: dict[tuple[str, str, str, int], int] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
            if key in lookup:
                raise ValueError(f"duplicate DINO observation key: {key}")
            lookup[key] = index
    if embeddings.ndim != 3 or embeddings.shape[1:] != (16, DINO_DIM):
        raise ValueError(f"unexpected DINO cache shape: {embeddings.shape}")
    if len(lookup) != len(embeddings):
        raise ValueError("DINO manifest and embedding count mismatch")
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    if bool(summary.get("future_candidate_rgb_used", True)):
        raise ValueError("DINO cache provenance indicates future candidate RGB use")
    return lookup, embeddings, summary


def _history_inputs(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    dino_lookup: Mapping[tuple[str, str, str, int], int],
    dino_embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    values: list[np.ndarray] = []
    missing = 0
    unique_keys: set[tuple[str, str, str, int]] = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in cache_index:
            raise ValueError(f"cache missing history episode: {episode_id}")
        i = cache_index[episode_id]
        observations: list[np.ndarray] = []
        for feature_name, viewpoint_name in (("s0_feature", "s0_viewpoint_id"), ("s1_feature", "s1_viewpoint_id")):
            key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row[viewpoint_name]))
            unique_keys.add(key)
            if key not in dino_lookup:
                missing += 1
                raise ValueError(f"DINO cache missing visited observation: {key}")
            feature = np.asarray(row[feature_name], dtype=np.float32)
            if feature.shape != (271,):
                raise ValueError(f"expected 271-D Stage-D feature, got {feature.shape}")
            dino_mean = np.asarray(dino_embeddings[dino_lookup[key]], dtype=np.float32).mean(axis=0)
            observations.append(np.concatenate([
                feature[:HISTORY_FEATURE_DIM],
                np.asarray(cache[f"current_logp_{feature_name[:2]}"][i], dtype=np.float32),
                dino_mean,
            ]))
        values.append(np.concatenate([observations[0], observations[1], observations[1] - observations[0]]))
    if missing:
        raise ValueError(f"missing visited DINO observations: {missing}")
    return np.asarray(values, dtype=np.float32), {"contexts": len(values), "unique_visited_observations": len(unique_keys)}


def _belief_logits(
    model: ActionBeliefEstimator,
    inputs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    normalized = ((inputs - mean) / std).astype(np.float32)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_belief(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    val_inputs: np.ndarray,
    val_labels: np.ndarray,
    *,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
    batch_size: int,
) -> tuple[ActionBeliefEstimator, np.ndarray, np.ndarray, dict[str, Any]]:
    _seed()
    mean = train_inputs.mean(axis=0).astype(np.float32)
    std = train_inputs.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized_train = ((train_inputs - mean) / std).astype(np.float32)
    normalized_val = ((val_inputs - mean) / std).astype(np.float32)
    model = ActionBeliefEstimator(input_dim=train_inputs.shape[1], hidden_dim=256, num_classes=NUM_CLASSES).to(device)
    loader = DataLoader(_ArrayDataset(normalized_train, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1 = -1.0
    best_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            features = features.float().to(device)
            labels = labels.long().to(device)
            loss = nn.functional.cross_entropy(model(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.inference_mode():
            val_logits = model(torch.from_numpy(normalized_val).float().to(device)).cpu().numpy()
        val_pred = np.argmax(val_logits, axis=1)
        val_metrics = _metrics(val_pred.tolist(), val_labels.tolist())
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(record)
        print(
            f"belief epoch={epoch}/{EPOCHS} loss={record['train_loss']:.6f} "
            f"val_acc={record['val_accuracy']:.6f} val_f1={record['val_macro_f1']:.6f}",
            flush=True,
        )
        if (record["val_macro_f1"], record["val_accuracy"]) > (best_f1, best_accuracy):
            best_f1, best_accuracy, best_epoch = record["val_macro_f1"], record["val_accuracy"], epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_mean": mean,
                    "input_std": std,
                    "input_dim": train_inputs.shape[1],
                    "hidden_dim": 256,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "best_epoch": epoch,
                },
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_macro_f1": best_f1,
        "final_train_loss": history[-1]["train_loss"],
        "elapsed_seconds": time.time() - started,
        "input_dim": int(train_inputs.shape[1]),
        "feature_schema": "[stgcn256, logp12, dino_mean768] for s0/s1 plus h1-h0",
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model, mean, std, summary


def _prepare_belief_jr(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    beliefs: np.ndarray,
    *,
    evidence: str,
) -> _Prepared:
    base = _prepare(rows, cache, orders, evidence=evidence, gt_identity=False)
    current = np.concatenate([base.arrays[0], beliefs.astype(np.float32)], axis=1)
    arrays = (current, *base.arrays[1:])
    return _Prepared(arrays=arrays, stats={**base.stats, "learned_belief_input": True}, episode_ids=base.episode_ids)


def _train_jr(
    prepared: _Prepared,
    *,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    _seed()
    model = _BeliefJointRevision().to(device)
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
        print(f"belief-jr evidence={prepared.stats['evidence']} epoch={epoch}/{EPOCHS} loss={losses[-1]:.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS, "num_classes": NUM_CLASSES, "learned_belief_input": True, "evidence": prepared.stats["evidence"]}, checkpoint)
    summary = {**prepared.stats, "seed": SEED, "epochs": EPOCHS, "batch_size": batch_size, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "loss_history": losses, "final_loss": losses[-1], "elapsed_seconds": time.time() - started, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_jr(checkpoint: Path, device: torch.device) -> _BeliefJointRevision:
    model = _BeliefJointRevision().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload["state_dict"]))
    model.eval()
    return model


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(np.mean(-np.sum(safe * np.log(safe), axis=1)))


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
    dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root)
    train_inputs, train_input_info = _history_inputs(train_rows, train_cache, dino_lookup, dino_embeddings)
    val_inputs, val_input_info = _history_inputs(moving_rows, val_cache, dino_lookup, dino_embeddings)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    if train_inputs.shape[0] != len(train_rows) or val_inputs.shape[0] != len(moving_rows):
        raise ValueError("history input and row counts are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = data_root / "checkpoints/activeview_reduced12_action_belief_estimator_v1"
    belief_checkpoint = checkpoint_root / "action_belief_best.pth"
    belief_training_path = output_dir / "belief_training.json"
    if belief_checkpoint.exists() and belief_training_path.exists():
        belief_payload = torch.load(belief_checkpoint, map_location=device, weights_only=False)
        belief_model = ActionBeliefEstimator(input_dim=int(belief_payload["input_dim"]), hidden_dim=int(belief_payload["hidden_dim"]), num_classes=NUM_CLASSES).to(device)
        belief_model.load_state_dict(belief_payload["state_dict"])
        belief_mean = np.asarray(belief_payload["input_mean"], dtype=np.float32)
        belief_std = np.asarray(belief_payload["input_std"], dtype=np.float32)
        belief_summary = json.loads(belief_training_path.read_text(encoding="utf-8"))
    else:
        belief_model, belief_mean, belief_std, belief_summary = _train_belief(
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            device=device,
            checkpoint=belief_checkpoint,
            summary_path=belief_training_path,
            batch_size=batch_size,
        )
    train_logits = _belief_logits(belief_model, train_inputs, belief_mean, belief_std, device=device, batch_size=batch_size)
    val_logits = _belief_logits(belief_model, val_inputs, belief_mean, belief_std, device=device, batch_size=batch_size)
    train_belief = torch.softmax(torch.from_numpy(train_logits), dim=1).numpy()
    val_belief = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    s1_pred = np.argmax(val_cache["current_logp_s1"], axis=1)
    learned_pred = np.argmax(val_belief, axis=1)
    belief_metrics = {
        "s1_only": _metrics(s1_pred.tolist(), val_labels.tolist()),
        "learned_belief": _metrics(learned_pred.tolist(), val_labels.tolist()),
        "s1_mean_entropy": _entropy(np.exp(val_cache["current_logp_s1"])),
        "learned_mean_entropy": _entropy(val_belief),
    }
    jr_prepared: dict[str, tuple[_Prepared, _Prepared]] = {}
    jr_summaries: dict[str, Any] = {}
    jr_models: dict[str, _BeliefJointRevision] = {}
    for evidence in ("imagined", "real"):
        train_prepared = _prepare_belief_jr(train_rows, train_cache, train_orders, train_belief, evidence=evidence)
        val_prepared = _prepare_belief_jr(moving_rows, val_cache, val_orders, val_belief, evidence=evidence)
        jr_prepared[evidence] = (train_prepared, val_prepared)
        checkpoint = checkpoint_root / f"joint_revision_{evidence}_learned_belief.pth"
        summary_path = output_dir / f"{evidence}_learned_belief_training.json"
        if checkpoint.exists() and summary_path.exists():
            jr_summaries[evidence] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            jr_summaries[evidence] = _train_jr(train_prepared, device=device, checkpoint=checkpoint, summary_path=summary_path, batch_size=batch_size)
        jr_models[evidence] = _load_jr(checkpoint, device)
    oracle_actions, _, _ = _reference_actions(moving_rows, val_cache, val_orders, mode="gt_label_real")
    fallback_s0 = {str(row["episode_id"]): int(row["current_predicted_label_id"]) for row in full_rows}
    new_methods_moving: dict[str, Any] = {}
    new_methods_full: dict[str, Any] = {}
    for evidence, display_name in (("imagined", "Imagined + LearnedBelief JR"), ("real", "Real + LearnedBelief JR")):
        _, val_prepared = jr_prepared[evidence]
        actions, terminal = _model_actions(val_prepared, jr_models[evidence], val_orders, val_cache, device=device, batch_size=batch_size)
        diagnostics = _diagnostics(actions, terminal, moving_rows, val_cache, orders=val_orders, oracle_actions=oracle_actions)
        new_methods_moving[display_name] = diagnostics
        full_pred, full_labels = _full_predictions(full_rows, moving_rows, terminal, fallback_s0)
        new_methods_full[display_name] = {"terminal": _metrics(full_pred, full_labels), **{k: v for k, v in diagnostics.items() if k != "terminal"}}
    previous_path = Path(__file__).resolve().parents[3] / "experiments/reduced12_eight_placement_v1/selector_ceiling_decomposition/result.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    method_order = [
        "Frozen H1", "Imagined + Inferred JR", "Real + Inferred JR",
        "Imagined + LearnedBelief JR", "Real + LearnedBelief JR",
        "Imagined + GT JR", "Real + GT JR", "FixedH1-H2 Oracle",
    ]
    methods_moving: dict[str, Any] = {}
    methods_full: dict[str, Any] = {}
    previous_names = {
        "Frozen H1": "Frozen H1",
        "Imagined + Inferred JR": "Imagined+Inferred JR",
        "Real + Inferred JR": "Real+Inferred JR",
        "Imagined + GT JR": "Imagined+GT JR",
        "Real + GT JR": "Real+GT JR",
        "FixedH1-H2 Oracle": "FixedH1-H2 Oracle",
    }
    for name, old_name in previous_names.items():
        methods_moving[name] = previous["methods"]["moving"][old_name]
        methods_full[name] = previous["methods"]["full"][old_name]
    methods_moving.update(new_methods_moving)
    methods_full.update(new_methods_full)
    imagined_inferred = methods_moving["Imagined + Inferred JR"]["terminal"]
    imagined_learned = methods_moving["Imagined + LearnedBelief JR"]["terminal"]
    imagined_gt = methods_moving["Imagined + GT JR"]["terminal"]
    real_learned = methods_moving["Real + LearnedBelief JR"]["terminal"]
    real_gt = methods_moving["Real + GT JR"]["terminal"]
    gain_accuracy = 100.0 * (imagined_learned["accuracy"] - imagined_inferred["accuracy"])
    gain_f1 = 100.0 * (imagined_learned["macro_f1"] - imagined_inferred["macro_f1"])
    remaining_accuracy = 100.0 * (imagined_gt["accuracy"] - imagined_learned["accuracy"])
    remaining_f1 = 100.0 * (imagined_gt["macro_f1"] - imagined_learned["macro_f1"])
    real_gap_to_gt = 100.0 * (real_gt["accuracy"] - real_learned["accuracy"])
    if gain_accuracy >= 2.0:
        belief_conclusion = f"Learned visited-history belief recovers meaningful identity information (+{gain_accuracy:.3f} pp Moving Accuracy over current JR)."
    else:
        belief_conclusion = f"Learned visited-history belief does not produce a large identity recovery (+{gain_accuracy:.3f} pp Moving Accuracy)."
    if real_gap_to_gt <= 5.0:
        selector_conclusion = f"Real+LearnedBelief is close to Real+GT (remaining {real_gap_to_gt:.3f} pp), so learned belief substantially closes the identity gap when WM evidence is accurate."
    else:
        selector_conclusion = f"Real+LearnedBelief remains {real_gap_to_gt:.3f} pp below Real+GT, so identity/selector limitations remain even with real candidate evidence."
    result = {
        "experiment_id": "REDUCED12_ACTION_BELIEF_ESTIMATOR_V1",
        "status": "COMPLETED",
        "test_used": False,
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows), "full_val_episodes": len(full_rows)},
        "method_order": method_order,
        "methods": {"moving": methods_moving, "full": methods_full},
        "belief_metrics": belief_metrics,
        "belief_gain": {"accuracy_pp": gain_accuracy, "macro_f1_pp": gain_f1},
        "remaining_identity_gap": {"accuracy_pp": remaining_accuracy, "macro_f1_pp": remaining_f1},
        "real_learned_to_real_gt_gap": {"accuracy_pp": real_gap_to_gt},
        "belief_training": belief_summary,
        "jr_training": jr_summaries,
        "dino_cache": dino_summary,
        "input_alignment": {"train": train_input_info, "val": val_input_info},
        "scientific_conclusion": {
            "belief": belief_conclusion,
            "real_candidate": selector_conclusion,
            "overall": f"{belief_conclusion} {selector_conclusion} The remaining limitation should be interpreted against the separate WM candidate-evidence gap, not as evidence to change WM-E in this experiment.",
        },
        "protocol": {"taxonomy": "reduced12_no_kneel_clean", "candidate_budget": "ALL_LEGAL", "terminal_observation": "real archived skeleton through frozen reduced12 ST-GCN", "candidate_evidence": "imagined or true frozen cache", "belief_inputs": "visited s0/s1 ST-GCN feature, logp, DINO mean and h1-h0", "test_used": False},
        "leakage_flags": {"test_used": False, "future_candidate_rgb_used": False, "future_candidate_dino_used": False, "gt_label_as_belief_input": False, "stgcn_retrained": False, "wm_retrained": False},
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    moving = result["methods"]["moving"]
    full = result["methods"]["full"]
    lines = [
        "# Reduced12 visited-history action belief estimator",
        "",
        "Train/Val only. No policy Test data was read; terminal predictions use the selected real archived observation and frozen reduced12 ST-GCN.",
        "",
        "## Belief-only Val Moving diagnostic",
        "",
        f"- S1-only Accuracy/F1: **{result['belief_metrics']['s1_only']['accuracy']:.6f} / {result['belief_metrics']['s1_only']['macro_f1']:.6f}**.",
        f"- Learned belief Accuracy/F1: **{result['belief_metrics']['learned_belief']['accuracy']:.6f} / {result['belief_metrics']['learned_belief']['macro_f1']:.6f}**.",
        f"- Mean entropy S1/learned: **{result['belief_metrics']['s1_mean_entropy']:.6f} / {result['belief_metrics']['learned_mean_entropy']:.6f}**.",
        "",
        "## Moving Val strategy comparison",
        "",
        "| Method | Accuracy | Macro-F1 | Positive-action hit | Stay rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in result["method_order"]:
        value = moving[name]
        lines.append(f"| {name} | {value['terminal']['accuracy']:.6f} | {value['terminal']['macro_f1']:.6f} | {value['positive_action_hit_rate']:.6f} | {value['stay_rate']:.6f} |")
    lines.extend(["", "## Full Val strategy comparison", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name in result["method_order"]:
        value = full[name]["terminal"]
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} |")
    lines.extend([
        "",
        "## Requested gaps",
        "",
        f"- Learned belief gain (Imagined+LearnedBelief − Imagined+Inferred): **{result['belief_gain']['accuracy_pp']:.3f} pp Accuracy / {result['belief_gain']['macro_f1_pp']:.3f} pp Macro-F1**.",
        f"- Remaining identity gap (Imagined+GT − Imagined+LearnedBelief): **{result['remaining_identity_gap']['accuracy_pp']:.3f} pp Accuracy / {result['remaining_identity_gap']['macro_f1_pp']:.3f} pp Macro-F1**.",
        f"- Real+LearnedBelief → Real+GT Accuracy gap: **{result['real_learned_to_real_gt_gap']['accuracy_pp']:.3f} pp**.",
        "",
        "## Scientific judgment",
        "",
        f"- {result['scientific_conclusion']['belief']}",
        f"- {result['scientific_conclusion']['real_candidate']}",
        f"- {result['scientific_conclusion']['overall']}",
        "",
        "## Leakage/protocol",
        "",
        "- reduced12 taxonomy; ALL_LEGAL candidate set; visited viewpoints excluded.",
        "- No ST-GCN/WM-E retraining; existing visited RGB/DINO and counterfactual caches reused.",
        "- `test_used=false`; no Test artifact was read.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; action-belief experiment requires GPU")
    output = args.output_dir or (Path(__file__).resolve().parents[3] / "experiments/reduced12_eight_placement_v1/action_belief_estimator_v1")
    result = run(args.data_root.resolve(), device, output.resolve(), batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"], "belief_gain": result["belief_gain"]}, indent=2))


if __name__ == "__main__":
    main()
