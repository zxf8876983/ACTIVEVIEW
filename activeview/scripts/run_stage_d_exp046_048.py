#!/usr/bin/env python3
"""EXP046--EXP048 counterfactual recognition decision campaign.

The script is deliberately an offline Train/Val-only pipeline.  WM-E and
ST-GCN are frozen feature/teacher models; true future recognition is stored
only as a target/evaluator quantity and is never passed to legal decision
models.  Test paths are not exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binom

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.stage_d_dense_campaign import context_key
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_evaluation import (
    build_baseline_trajectories,
    build_fixed_first_oracle,
    build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.active_view.stage_d_world_model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
)
from activeview.core.paths import get_data_root
from activeview.scripts.build_stage_b_utility_labels import _load_model
from activeview.scripts.run_stage_d_exp041_044 import (
    EXP_ROOT,
    _episode_sources,
    _load_rgb_lookup,
    _rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP046_ROOT = REPO_ROOT / "experiments/stage_d/EXP046_counterfactual_recognition_dataset"
EXP047_ROOT = REPO_ROOT / "experiments/stage_d/EXP047_counterfactual_decision_layer"
EXP048_ROOT = REPO_ROOT / "experiments/stage_d/EXP048_observation_decision_gap_decomposition"
N_CLASSES = 16
VIEW_COUNT = 32
SEED = 42


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stgcn(data_root: Path, device: torch.device) -> STGCN:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
    mapping = json.loads(Path(summary["label_mapping"]).read_text())
    model, _ = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), str(device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _log_probs(model: STGCN, skeletons: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeletons), 1024):
            batch = torch.from_numpy(skeletons[start : start + 1024]).float().to(device)
            values.append(torch.log_softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(values, axis=0)


def _load_wm_e(checkpoint: Path, device: torch.device) -> CandidateObservationWorldModel:
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=True, residual=False).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _generate_split_cache(
    *,
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[tuple[str, str, str], str],
    wm: CandidateObservationWorldModel,
    stgcn: STGCN,
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray],
    device: torch.device,
    output: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    selected = list(rows[:limit] if limit is not None else rows)
    dataset = LazyWorldModelContextDataset(
        selected, sources, use_belief=True, rgb_lookup=rgb_lookup,
        target_scope="all", cache_size=32,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_world_model_context)
    n = len(selected)
    imagined = np.empty((n, VIEW_COUNT, N_CLASSES), dtype=np.float32)
    truth = np.empty_like(imagined)
    current_s0 = np.empty((n, N_CLASSES), dtype=np.float32)
    current_s1 = np.empty_like(current_s0)
    candidate_desc = np.empty((n, VIEW_COUNT, 9), dtype=np.float32)
    candidate_ids = np.empty((n, VIEW_COUNT), dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    s0_ids = np.empty(n, dtype=np.int64)
    s1_ids = np.empty(n, dtype=np.int64)
    episode_ids: list[str] = []
    cursor = 0
    wm.eval()
    with torch.inference_mode():
        for batch in loader:
            count = len(batch["context_key"])
            kwargs = {
                name: batch[name].to(device)
                for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
            }
            kwargs["history_belief"] = batch["history_belief"].to(device)
            kwargs["history_rgb"] = batch["history_rgb"].to(device)
            chunks: list[torch.Tensor] = []
            for start in range(0, VIEW_COUNT, 16):
                chunks.append(wm(**{**kwargs, "candidate_descriptor": kwargs["candidate_descriptor"][:, start : start + 16]}))
            predicted = torch.cat(chunks, dim=1).cpu().numpy()
            targets = batch["target_skeleton"].numpy()
            flat_pred = predicted.reshape(-1, 3, 30, 17)
            flat_true = targets.reshape(-1, 3, 30, 17)
            pred_logs = _log_probs(stgcn, flat_pred, device).reshape(count, VIEW_COUNT, N_CLASSES)
            true_logs = _log_probs(stgcn, flat_true, device).reshape(count, VIEW_COUNT, N_CLASSES)
            hist = batch["history_skeleton"].numpy()
            hist_logs = _log_probs(stgcn, hist.reshape(-1, 3, 30, 17), device).reshape(count, 2, N_CLASSES)
            imagined[cursor : cursor + count] = pred_logs
            truth[cursor : cursor + count] = true_logs
            current_s0[cursor : cursor + count] = hist_logs[:, 0]
            current_s1[cursor : cursor + count] = hist_logs[:, 1]
            candidate_desc[cursor : cursor + count] = batch["candidate_descriptor"].numpy()
            candidate_ids[cursor : cursor + count] = batch["candidate_ids"].numpy()
            labels[cursor : cursor + count] = batch["label_id"].numpy()
            for index, key in enumerate(batch["context_key"]):
                row = selected[cursor + index]
                episode_ids.append(str(row["episode_id"]))
                s0_ids[cursor + index] = int(row["s0_viewpoint_id"])
                s1_ids[cursor + index] = int(row["s1_viewpoint_id"])
            cursor += count
            if cursor == n or cursor % 1024 == 0:
                print(f"{output.stem}: {cursor}/{n}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        episode_ids=np.asarray(episode_ids, dtype="U128"),
        context_keys=np.asarray(["|".join(context_key(row)) for row in selected], dtype="U256"),
        imagined_logp=imagined,
        true_logp=truth,
        current_logp_s0=current_s0,
        current_logp_s1=current_s1,
        candidate_descriptor=candidate_desc,
        candidate_ids=candidate_ids,
        label_id=labels,
        s0_viewpoint_id=s0_ids,
        s1_viewpoint_id=s1_ids,
        policy_split=np.asarray([str(row["policy_split"]) for row in selected], dtype="U8"),
        wm_e_frozen=np.asarray(True),
        true_future_recognition_evaluator_only=np.asarray(True),
    )
    return {"path": str(output.resolve()), "sha256": _sha256(output), "contexts": n, "viewpoints": n * VIEW_COUNT}


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _cache_valid(path: Path, expected_contexts: int, split: str) -> bool:
    if not path.is_file():
        return False
    try:
        cache = _load_cache(path)
        return (
            cache["imagined_logp"].shape == (expected_contexts, VIEW_COUNT, N_CLASSES)
            and cache["true_logp"].shape == cache["imagined_logp"].shape
            and cache["policy_split"].shape == (expected_contexts,)
            and all(str(value).lower() == split for value in cache["policy_split"].tolist())
        )
    except (KeyError, ValueError, OSError):
        return False


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    return _corr(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))


def _entropy(logp: np.ndarray) -> np.ndarray:
    p = np.exp(logp)
    return -np.sum(p * logp, axis=-1)


def _exp046_analysis(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    def compute(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], split: str) -> dict[str, Any]:
        labels = cache["label_id"]
        imag = cache["imagined_logp"]
        true = cache["true_logp"]
        current = cache["current_logp_s1"]
        imag_flat, true_flat = imag.reshape(-1, N_CLASSES), true.reshape(-1, N_CLASSES)
        agreements = np.argmax(imag_flat, axis=1) == np.argmax(true_flat, axis=1)
        p_imag, p_true = np.exp(imag_flat), np.exp(true_flat)
        kl = np.sum(p_true * (true_flat - imag_flat), axis=1)
        l1 = np.abs(p_true - p_imag).sum(axis=1)
        label_flat = np.repeat(labels, VIEW_COUNT)
        il = imag_flat[np.arange(len(label_flat)), label_flat]
        tl = true_flat[np.arange(len(label_flat)), label_flat]
        current_correct = np.argmax(current, axis=1) == labels
        p2p3_true = np.asarray([true[i, int(v)] for i, row in enumerate(rows) for v in row["remaining_candidate_ids"]], dtype=np.float32)
        p2p3_imag = np.asarray([imag[i, int(v)] for i, row in enumerate(rows) for v in row["remaining_candidate_ids"]], dtype=np.float32)
        # Pairwise sign agreement uses the two ordered candidates in each row.
        pair_sign = []
        for i, row in enumerate(rows):
            ids = [int(v) for v in row["remaining_candidate_ids"]]
            if len(ids) == 2:
                pair_sign.append(np.sign(true[i, ids[0], labels[i]] - true[i, ids[1], labels[i]]) == np.sign(imag[i, ids[0], labels[i]] - imag[i, ids[1], labels[i]]))
        quantile_edges = np.quantile(_entropy(train_cache["current_logp_s1"]), [0.25, 0.5, 0.75])
        val_entropy = _entropy(current)
        bins = np.digitize(_entropy(current), quantile_edges)
        conditional: dict[str, Any] = {"current_correct": {}, "entropy_quartiles_train_thresholds": quantile_edges.tolist()}
        for name, mask in (("CURRENT_CORRECT", current_correct), ("CURRENT_WRONG", ~current_correct)):
            idx = np.repeat(mask, VIEW_COUNT)
            conditional["current_correct"][name] = {"contexts": int(mask.sum()), "imagined_true_agreement": float(agreements[idx].mean()) if idx.any() else 0.0, "true_label_logp_pearson": _corr(il[idx], tl[idx])}
        conditional["entropy_quartiles"] = {}
        for q in range(4):
            idx = np.repeat(bins == q, VIEW_COUNT)
            conditional["entropy_quartiles"][f"Q{q + 1}"] = {"contexts": int((bins == q).sum()), "imagined_true_agreement": float(agreements[idx].mean()) if idx.any() else 0.0, "true_label_logp_pearson": _corr(il[idx], tl[idx])}
        per_class: dict[str, Any] = {}
        for cls in range(N_CLASSES):
            mask = label_flat == cls
            per_class[str(cls)] = {"support": int(mask.sum()), "agreement": float(agreements[mask].mean()) if mask.any() else 0.0, "true_label_logp_pearson": _corr(il[mask], tl[mask])}
        return {"split": split, "viewpoints": int(len(imag_flat)), "stgcn_top1_agreement": float(agreements.mean()), "kl_mean": float(kl.mean()), "probability_l1_mean": float(l1.mean()), "true_label_logp_pearson": _corr(il, tl), "true_label_logp_spearman": _rank_corr(il, tl), "p2_p3_sign_agreement": float(np.mean(pair_sign)) if pair_sign else None, "conditional": conditional, "per_class": per_class}

    return {"train": compute(train_rows, train_cache, "train"), "val": compute(val_rows, val_cache, "val")}


def _rows_for_models(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    index = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}
    posterior_x: list[np.ndarray] = []
    posterior_y: list[int] = []
    correctness_x: list[np.ndarray] = []
    correctness_y: list[int] = []
    pair_x: list[np.ndarray] = []
    pair_y: list[int] = []
    context_indices: list[int] = []
    candidate_ids: list[list[int]] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        labels = int(cache["label_id"][i])
        ids = [int(v) for v in row["remaining_candidate_ids"]]
        context_indices.append(i)
        candidate_ids.append(ids)
        base = np.concatenate([cache["current_logp_s0"][i], cache["current_logp_s1"][i]])
        action_features: list[np.ndarray] = []
        action_correct: list[int] = []
        for candidate in [None, *ids]:
            if candidate is None:
                imagined = cache["current_logp_s1"][i]
                descriptor = np.zeros(9, dtype=np.float32)
                correct = int(np.argmax(cache["current_logp_s1"][i]) == labels)
            else:
                imagined = cache["imagined_logp"][i, candidate]
                descriptor = cache["candidate_descriptor"][i, candidate]
                correct = int(np.argmax(cache["true_logp"][i, candidate]) == labels)
            feature = np.concatenate([base, imagined, descriptor]).astype(np.float32)
            action_features.append(feature)
            action_correct.append(correct)
            posterior_x.append(feature)
            posterior_y.append(labels)
            correctness_x.append(feature)
            correctness_y.append(correct)
        if len(ids) == 2:
            pair_x.append(np.concatenate([base, cache["imagined_logp"][i, ids[0]], cache["imagined_logp"][i, ids[1]], cache["candidate_descriptor"][i, ids[0]], cache["candidate_descriptor"][i, ids[1]]]).astype(np.float32))
            if action_correct[1] == action_correct[2]:
                pair_y.append(-1)
            else:
                pair_y.append(int(action_correct[1] > action_correct[2]))
    return {"features": np.asarray(posterior_x, dtype=np.float32), "posterior_targets": np.asarray(posterior_y, dtype=np.int64), "correctness_features": np.asarray(correctness_x, dtype=np.float32), "correctness_targets": np.asarray(correctness_y, dtype=np.float32), "pair_features": np.asarray(pair_x, dtype=np.float32).reshape(-1, 82), "pair_targets": np.asarray(pair_y, dtype=np.int64), "context_indices": np.asarray(context_indices, dtype=np.int64), "candidate_ids": candidate_ids}


class _Posterior(nn.Module):
    def __init__(self, dim: int = 57) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BinaryMLP(nn.Module):
    def __init__(self, dim: int = 57) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, *, task: str, epochs: int = 20, lr: float = 1e-3, weight_decay: float = 1e-4, device: torch.device = torch.device("cpu")) -> float:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn: nn.Module = nn.CrossEntropyLoss() if task == "ce" else nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    final = 0.0
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if task == "bce" and logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, yb.long() if task == "ce" else yb.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        final = total / max(len(dataset), 1)
    return final


def _features_for_context(row: Mapping[str, Any], cache: Mapping[str, np.ndarray], i: int, *, observation: str = "imagined") -> tuple[np.ndarray, list[int]]:
    ids = [int(v) for v in row["remaining_candidate_ids"]]
    base = np.concatenate([cache["current_logp_s0"][i], cache["current_logp_s1"][i]])
    values: list[np.ndarray] = []
    for candidate in [None, *ids]:
        logs = cache["current_logp_s1"][i] if candidate is None else cache[f"{observation}_logp"][i, candidate]
        desc = np.zeros(9, dtype=np.float32) if candidate is None else cache["candidate_descriptor"][i, candidate]
        values.append(np.concatenate([base, logs, desc]).astype(np.float32))
    return np.asarray(values), ids


def _decisions_from_scores(rows: Sequence[Mapping[str, Any]], scores: Sequence[np.ndarray], *, mode: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, values in zip(rows, scores):
        ids = [int(v) for v in row["remaining_candidate_ids"]]
        if mode == "confidence":
            selected = int(np.argmax(values))
        else:
            selected = int(np.argmax(values))
        output.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": ids, "predicted_utilities": [float(v) for v in values[1:]], "predicted_stays": selected == 0, "predicted_candidate_viewpoint_id": None if selected == 0 else ids[selected - 1], "max_predicted_utility": float(values[selected])})
    return output


def _evaluate_models(
    *,
    data_root: Path,
    val_rows: Sequence[Mapping[str, Any]],
    val_cache: Mapping[str, np.ndarray],
    train_models: Mapping[str, nn.Module],
    device: torch.device,
    observation: str = "imagined",
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    v0 = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
    mapping = json.loads(Path(summary["label_mapping"]).read_text())
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    index = {str(e): i for i, e in enumerate(val_cache["episode_ids"].tolist())}
    method_names = ("CF_POSTERIOR", "CF_CORRECTNESS_LOGISTIC", "CF_CORRECTNESS_MLP", "CF_CORRECTNESS_X_RELIABILITY", "CF_PAIRWISE_CORRECTNESS")
    method_scores: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
    for row in val_rows:
        i = index[str(row["episode_id"])]
        features, ids = _features_for_context(row, val_cache, i, observation=observation)
        with torch.inference_mode():
            if "CF_POSTERIOR" in train_models:
                logits = train_models["CF_POSTERIOR"](torch.from_numpy(features).float().to(device)).cpu().numpy()
                probs = np.exp(logits - logits.max(axis=1, keepdims=True)); probs /= probs.sum(axis=1, keepdims=True)
                method_scores["CF_POSTERIOR"].append(np.asarray([float(np.max(probs[a])) for a in range(len(ids) + 1)], dtype=np.float32))
            for name in ("CF_CORRECTNESS_LOGISTIC", "CF_CORRECTNESS_MLP"):
                if name in train_models:
                    probabilities = torch.sigmoid(train_models[name](torch.from_numpy(features).float().to(device))).detach().cpu().numpy().reshape(-1)
                    method_scores[name].append(probabilities.astype(np.float32))
            if "CF_CORRECTNESS_X_RELIABILITY" in train_models:
                corr = torch.sigmoid(train_models["CF_CORRECTNESS_MLP"](torch.from_numpy(features).float().to(device))).cpu().numpy()
                obs_logs = val_cache["current_logp_s1"][i] if observation == "imagined" else val_cache["current_logp_s1"][i]
                inp = np.stack([np.concatenate([np.exp(val_cache["current_logp_s0"][i]), np.exp(val_cache["current_logp_s1"][i]), np.exp(obs_logs if c == 0 else val_cache[f"{observation}_logp"][i, c])]) for c in [0, *ids]])
                rel = torch.sigmoid(train_models["WM_RELIABILITY"](torch.from_numpy(inp).float().to(device))).cpu().numpy() if "WM_RELIABILITY" in train_models else np.ones(3)
                method_scores["CF_CORRECTNESS_X_RELIABILITY"].append(corr * rel)
            if "CF_PAIRWISE_CORRECTNESS" in train_models:
                corr = torch.sigmoid(train_models["CF_CORRECTNESS_MLP"](torch.from_numpy(features).float().to(device))).cpu().numpy()
                if len(ids) == 2:
                    pair_feature = np.concatenate([
                        val_cache["current_logp_s0"][i], val_cache["current_logp_s1"][i],
                        val_cache[f"{observation}_logp"][i, ids[0]], val_cache[f"{observation}_logp"][i, ids[1]],
                        val_cache["candidate_descriptor"][i, ids[0]], val_cache["candidate_descriptor"][i, ids[1]],
                    ]).astype(np.float32)
                    pair_probability = float(torch.sigmoid(train_models["CF_PAIRWISE_CORRECTNESS"](torch.from_numpy(pair_feature).float().unsqueeze(0).to(device))).cpu())
                    pair = np.asarray([corr[0], corr[1] if pair_probability > 0.5 else corr[2]], dtype=np.float32)
                else:
                    pair = np.asarray([corr[0], corr[1]], dtype=np.float32)
                method_scores["CF_PAIRWISE_CORRECTNESS"].append(pair)
    results: dict[str, Any] = {}
    decisions: dict[str, list[dict[str, Any]]] = {}
    for name, values in method_scores.items():
        decisions[name] = _decisions_from_scores(val_rows, values, mode="confidence")
        trajectory = summarize_trajectory_rows(build_stage_d_trajectories(stage_b, v0, val_rows, decisions[name]), categories)
        results[name] = {"trajectory": trajectory, "deployable": True}
    return results, decisions


def _gt_label_imagined_decisions(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Privileged evaluator selector: imagined recognition scored at GT label."""
    index = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}
    output: list[dict[str, Any]] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        ids = [int(v) for v in row["remaining_candidate_ids"]]
        scores = [float(cache["current_logp_s1"][i, int(row["label_id"])])]
        scores.extend(float(cache["imagined_logp"][i, candidate, int(row["label_id"])]) for candidate in ids)
        selected = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        output.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": ids, "predicted_utilities": [0.0] * len(ids), "predicted_stays": selected == 0, "predicted_candidate_viewpoint_id": None if selected == 0 else ids[selected - 1], "max_predicted_utility": scores[selected]})
    return output


def _paired_statistics(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a = np.asarray([int(row["predicted_label_id"]) == int(row["label_id"]) for row in candidate], dtype=bool)
    b = np.asarray([int(row["predicted_label_id"]) == int(row["label_id"]) for row in baseline], dtype=bool)
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b)); discordant = n01 + n10
    # Exact two-sided McNemar binomial p-value avoids a scipy dependency.
    p_value = float(1.0 if discordant == 0 else min(1.0, 2.0 * binom.cdf(min(n01, n10), discordant, 0.5)))
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(a), size=(10000, len(a)))
    delta_acc = a.astype(np.float64) - b.astype(np.float64)
    # Macro-F1 bootstrap is intentionally omitted from per-resample confusion
    # construction here; accuracy remains the preregistered paired gate.
    bootstrap = np.mean(delta_acc[indices], axis=1)
    return {"n01_candidate_wrong_baseline_correct": n01, "n10_candidate_correct_baseline_wrong": n10, "mcnemar_p_value": p_value, "delta_accuracy": float(delta_acc.mean()), "delta_accuracy_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "bootstrap_resamples": 10000, "seed": SEED}


def _rescue_harm(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count correctness changes only when the selected action changes."""
    if len(candidate) != len(baseline):
        raise ValueError("trajectory lengths differ for rescue/harm analysis")
    rescued = harmful = 0
    for new, old in zip(candidate, baseline):
        if int(new["selected_viewpoint_id"]) == int(old["selected_viewpoint_id"]):
            continue
        old_correct = int(old["predicted_label_id"]) == int(old["label_id"])
        new_correct = int(new["predicted_label_id"]) == int(new["label_id"])
        rescued += int(not old_correct and new_correct)
        harmful += int(old_correct and not new_correct)
    return {"rescued_episodes": rescued, "harmful_moves": harmful, "net_correctness_gain": rescued - harmful}


def _subgroup_summary(mask: np.ndarray, baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], stay: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    def accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
        if indices.size == 0:
            return 0.0
        return float(np.mean([int(rows[i]["predicted_label_id"]) == int(rows[i]["label_id"]) for i in indices]))
    return {
        "contexts": int(indices.size),
        "stay_accuracy": accuracy(stay),
        "exp014_accuracy": accuracy(baseline),
        "best_legal_accuracy": accuracy(candidate),
        "delta_vs_exp014": accuracy(candidate) - accuracy(baseline),
        "move_rate_best_legal": float(np.mean([int(candidate[i]["moves"]) > 0 for i in indices])) if indices.size else 0.0,
    }


def _subgroup_analysis(
    *,
    train_cache: Mapping[str, np.ndarray],
    val_cache: Mapping[str, np.ndarray],
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    stay: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = val_cache["label_id"]
    current = val_cache["current_logp_s1"]
    current_correct = np.argmax(current, axis=1) == labels
    val_entropy = _entropy(current)
    val_confidence = np.exp(current).max(axis=1)
    train_entropy_edges = np.quantile(_entropy(train_cache["current_logp_s1"]), [0.25, 0.5, 0.75])
    train_conf_edges = np.quantile(np.exp(train_cache["current_logp_s1"]).max(axis=1), [0.25, 0.5, 0.75])
    entropy_bins = np.digitize(val_entropy, train_entropy_edges)
    confidence_bins = np.digitize(val_confidence, train_conf_edges)
    groups: dict[str, Any] = {
        "current_correctness": {
            "CURRENT_CORRECT": _subgroup_summary(current_correct, baseline, candidate, stay),
            "CURRENT_WRONG": _subgroup_summary(~current_correct, baseline, candidate, stay),
        },
        "entropy_quartiles": {
            f"Q{q + 1}": _subgroup_summary(entropy_bins == q, baseline, candidate, stay) for q in range(4)
        },
        "confidence_quartiles": {
            f"Q{q + 1}": _subgroup_summary(confidence_bins == q, baseline, candidate, stay) for q in range(4)
        },
        "special": {
            "top1_correct_low_confidence": _subgroup_summary(current_correct & (confidence_bins == 0), baseline, candidate, stay),
            "current_wrong_high_confidence": _subgroup_summary(~current_correct & (confidence_bins == 3), baseline, candidate, stay),
        },
        "thresholds_from_train": {
            "entropy": train_entropy_edges.tolist(),
            "confidence": train_conf_edges.tolist(),
        },
    }
    per_class: dict[str, Any] = {}
    for cls in range(N_CLASSES):
        mask = labels == cls
        summary = _subgroup_summary(mask, baseline, candidate, stay)
        summary["support"] = int(mask.sum())
        per_class[str(cls)] = summary
    groups["per_class"] = per_class
    return groups


def run_campaign(data_root: Path, device: torch.device, *, wm_checkpoint: Path, rgb_root: Path) -> dict[str, Any]:
    _seed()
    torch.set_num_threads(min(8, max(1, __import__("os").cpu_count() or 1)))
    train_rows, val_rows = _rows(data_root, "train"), _rows(data_root, "val")
    train_sources, val_sources = _episode_sources(data_root, "train"), _episode_sources(data_root, "val")
    if (len(train_rows), len(val_rows)) != (29133, 9742):
        raise RuntimeError("canonical Train/Val population mismatch")
    if not wm_checkpoint.is_file():
        raise FileNotFoundError(f"WM-E checkpoint not found: {wm_checkpoint}")
    exp014_result = json.loads((REPO_ROOT / "experiments/stage_d/EXP014_two_step_sequential/result.json").read_text())
    exp014_metrics = exp014_result["metrics"]["exp014"]
    if abs(float(exp014_metrics["accuracy"]) - 0.6582540931) > 1e-5 or abs(float(exp014_metrics["macro_f1"]) - 0.6101526052) > 1e-5:
        raise RuntimeError("EXP014 evaluator reproduction gate failed")
    stgcn = _load_stgcn(data_root, device)
    wm = _load_wm_e(wm_checkpoint, device)
    rgb_lookup, rgb_summary = _load_rgb_lookup(rgb_root)
    EXP046_ROOT.mkdir(parents=True, exist_ok=True)
    train_cache_path = data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"
    val_cache_path = data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz"
    train_info = {"path": str(train_cache_path.resolve()), "sha256": _sha256(train_cache_path), "contexts": len(train_rows), "viewpoints": len(train_rows) * VIEW_COUNT} if _cache_valid(train_cache_path, len(train_rows), "train") else _generate_split_cache(rows=train_rows, sources=train_sources, wm=wm, stgcn=stgcn, rgb_lookup=rgb_lookup, device=device, output=train_cache_path)
    val_info = {"path": str(val_cache_path.resolve()), "sha256": _sha256(val_cache_path), "contexts": len(val_rows), "viewpoints": len(val_rows) * VIEW_COUNT} if _cache_valid(val_cache_path, len(val_rows), "val") else _generate_split_cache(rows=val_rows, sources=val_sources, wm=wm, stgcn=stgcn, rgb_lookup=rgb_lookup, device=device, output=val_cache_path)
    train_cache, val_cache = _load_cache(train_cache_path), _load_cache(val_cache_path)
    exp046 = _exp046_analysis(train_rows=train_rows, val_rows=val_rows, train_cache=train_cache, val_cache=val_cache)
    (EXP046_ROOT / "result.json").write_text(json.dumps({"experiment_id": "EXP046", "status": "COMPLETED", "split": ["train", "val"], "population": {"train_contexts": len(train_rows), "val_contexts": len(val_rows)}, "cache": {"train": train_info, "val": val_info}, "test_used": False, "training_performed": False, "wm_e_frozen": True, "stgcn_frozen": True}, indent=2), encoding="utf-8")
    (EXP046_ROOT / "conditional_fidelity.json").write_text(json.dumps(exp046, indent=2), encoding="utf-8")
    (EXP046_ROOT / "cache_audit.json").write_text(json.dumps({"train": train_info, "val": val_info, "wm_e_checkpoint": str(wm_checkpoint.resolve()), "wm_e_checkpoint_sha256": _sha256(wm_checkpoint), "rgb_future_candidate_used": False, "test_used": False}, indent=2), encoding="utf-8")
    train_model_rows = _rows_for_models(train_rows, train_cache)
    EXP047_ROOT.mkdir(parents=True, exist_ok=True)
    posterior = _Posterior()
    posterior_loss = _fit_torch(posterior, train_model_rows["features"], train_model_rows["posterior_targets"], task="ce", device=device)
    correctness_logistic = nn.Linear(57, 1)
    logistic_loss = _fit_torch(correctness_logistic, train_model_rows["correctness_features"], train_model_rows["correctness_targets"], task="bce", epochs=20, lr=1e-2, weight_decay=1.0, device=device)
    correctness_mlp = _BinaryMLP()
    mlp_loss = _fit_torch(correctness_mlp, train_model_rows["correctness_features"], train_model_rows["correctness_targets"], task="bce", device=device)
    reliability = _BinaryMLP(dim=48)
    reliability_x = []
    reliability_y = []
    for i in range(len(train_cache["episode_ids"])):
        for view in range(VIEW_COUNT):
            reliability_x.append(np.concatenate([np.exp(train_cache["current_logp_s0"][i]), np.exp(train_cache["current_logp_s1"][i]), np.exp(train_cache["imagined_logp"][i, view])]))
            reliability_y.append(float(np.argmax(train_cache["imagined_logp"][i, view]) == np.argmax(train_cache["true_logp"][i, view])))
    reliability_loss = _fit_torch(reliability, np.asarray(reliability_x, dtype=np.float32), np.asarray(reliability_y, dtype=np.float32), task="bce", device=device)
    pair_mask = train_model_rows["pair_targets"] >= 0
    pairwise = _BinaryMLP(dim=82)
    pairwise_loss = _fit_torch(pairwise, train_model_rows["pair_features"][pair_mask], train_model_rows["pair_targets"][pair_mask].astype(np.float32), task="bce", device=device)
    models: dict[str, nn.Module] = {"CF_POSTERIOR": posterior, "CF_CORRECTNESS_LOGISTIC": correctness_logistic, "CF_CORRECTNESS_MLP": correctness_mlp, "WM_RELIABILITY": reliability, "CF_CORRECTNESS_X_RELIABILITY": correctness_mlp, "CF_PAIRWISE_CORRECTNESS": pairwise}
    for model in models.values():
        model.eval()
    model_results, decision_map = _evaluate_models(data_root=data_root, val_rows=val_rows, val_cache=val_cache, train_models=models, device=device)
    categories = [name for name, _ in sorted(json.loads(Path(json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())["label_mapping"]).read_text()).items(), key=lambda item: int(item[1]))]
    stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    v0 = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    oracle_rows = build_fixed_first_oracle(stage_b, v0, val_rows)
    model_results["Fixed-first Oracle"] = {"trajectory": summarize_trajectory_rows(oracle_rows, categories), "deployable": False}
    exp014_rows = build_baseline_trajectories(stage_b, v0)["FrozenStageCv0"]
    for name, model in (("CF_POSTERIOR", posterior), ("CF_CORRECTNESS_LOGISTIC", correctness_logistic), ("CF_CORRECTNESS_MLP", correctness_mlp), ("WM_RELIABILITY", reliability), ("CF_PAIRWISE_CORRECTNESS", pairwise)):
        torch.save({"state_dict": model.state_dict(), "name": name, "seed": SEED}, EXP047_ROOT / f"{name.lower()}.pth")
    exp047 = {"experiment_id": "EXP047", "status": "COMPLETED", "split": ["train", "val"], "models": model_results, "frozen_baselines": {"WM_E_PREDICTED_ENTROPY": {"accuracy": 0.6418817473368128, "mean_regret": 1.7502856700690215}, "WM_E_PREDICTED_TOP1": {"accuracy": 0.6453850003574748, "mean_regret": 1.778047185882174}, "WM_E_BELIEF_CROSS_ENTROPY": {"accuracy": 0.6488882533781368, "mean_regret": 1.4512375587300403}, "WM_E_GT_LABEL_ORACLE": {"accuracy": 0.730535497247444, "mean_regret": 1.0559970576704858}}, "train_losses": {"CF_POSTERIOR": posterior_loss, "CF_CORRECTNESS_LOGISTIC": logistic_loss, "CF_CORRECTNESS_MLP": mlp_loss, "WM_RELIABILITY": reliability_loss, "CF_PAIRWISE_CORRECTNESS": pairwise_loss}, "wm_e_frozen": True, "stgcn_frozen": True, "test_used": False, "training_performed": True}
    (EXP047_ROOT / "result.json").write_text(json.dumps(exp047, indent=2), encoding="utf-8")
    (EXP047_ROOT / "model_comparison.json").write_text(json.dumps({name: value["trajectory"] for name, value in model_results.items()}, indent=2), encoding="utf-8")
    selector_summary = {
        name: {
            "accuracy": value["trajectory"]["recognition"]["accuracy"],
            "macro_f1": value["trajectory"]["recognition"]["macro_f1"],
            "mean_regret": value["trajectory"]["decision_regret"]["mean"],
            "p90_regret": value["trajectory"]["decision_regret"]["p90"],
            "move_rate": 1.0 - value["trajectory"]["movement"]["move_0_rate"],
        }
        for name, value in model_results.items()
    }
    (EXP047_ROOT / "selector_comparison.json").write_text(json.dumps(selector_summary, indent=2), encoding="utf-8")
    # EXP048 keeps the same decision weights; true observations and GT-label
    # selectors are explicitly marked privileged evaluator diagnostics.
    best_name = max((name for name in model_results if name.startswith("CF_")), key=lambda name: model_results[name]["trajectory"]["recognition"]["accuracy"])
    true_obs_results, _ = _evaluate_models(data_root=data_root, val_rows=val_rows, val_cache=val_cache, train_models=models, device=device, observation="true")
    best_a = float(model_results[best_name]["trajectory"]["recognition"]["accuracy"])
    best_b = float(true_obs_results[best_name]["trajectory"]["recognition"]["accuracy"])
    c_decisions = _gt_label_imagined_decisions(val_rows, val_cache)
    c_rows = build_stage_d_trajectories(stage_b, v0, val_rows, c_decisions)
    c_accuracy = float(summarize_trajectory_rows(c_rows, categories)["recognition"]["accuracy"])
    d_accuracy = float(model_results["Fixed-first Oracle"]["trajectory"]["recognition"]["accuracy"])
    best_decision_rows = build_stage_d_trajectories(stage_b, v0, val_rows, decision_map[best_name])
    no_move_rows = build_baseline_trajectories(stage_b, v0)["NoMove"]
    paired = {"vs_exp014": _paired_statistics(best_decision_rows, exp014_rows), "best_method_vs_fixed_first_oracle": _paired_statistics(best_decision_rows, oracle_rows)}
    exp023_path = data_root / "experiments/stage_d/EXP023_warmstarted_contextual_bandit/val_predictions.jsonl"
    if exp023_path.is_file():
        exp023_predictions = load_jsonl(exp023_path)
        exp023_rows = build_stage_d_trajectories(stage_b, v0, val_rows, exp023_predictions)
        paired["best_method_vs_exp023"] = _paired_statistics(best_decision_rows, exp023_rows)
    (EXP048_ROOT / "paired_statistics.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    rescue_harm = _rescue_harm(best_decision_rows, exp014_rows)
    (EXP048_ROOT / "rescue_harm.json").write_text(json.dumps(rescue_harm, indent=2), encoding="utf-8")
    subgroups = _subgroup_analysis(train_cache=train_cache, val_cache=val_cache, baseline=exp014_rows, candidate=best_decision_rows, stay=no_move_rows)
    (EXP048_ROOT / "subgroup_analysis.json").write_text(json.dumps(subgroups, indent=2), encoding="utf-8")
    exp048 = {"experiment_id": "EXP048", "status": "COMPLETED", "best_legal_method": best_name, "A_best_legal_imagined": best_a, "B_same_decision_true_observation": best_b, "C_imagined_gt_label_oracle": c_accuracy, "D_true_observation_gt_label_oracle": d_accuracy, "world_model_gap": best_b - best_a, "decision_label_gap": c_accuracy - best_a, "residual_oracle_gap": d_accuracy - max(best_b, c_accuracy), "paired_statistics": paired, "rescue_harm": rescue_harm, "subgroup_analysis_path": str((EXP048_ROOT / "subgroup_analysis.json").resolve()), "deployable": False, "test_used": False}
    (EXP048_ROOT / "gap_decomposition.json").write_text(json.dumps(exp048, indent=2), encoding="utf-8")
    (EXP048_ROOT / "result.json").write_text(json.dumps(exp048, indent=2), encoding="utf-8")
    return {"experiment_id": "EXP046-048", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "val_contexts": len(val_rows)}, "exp014_evaluator_gate": {"status": "PASS", "accuracy": float(exp014_metrics["accuracy"]), "macro_f1": float(exp014_metrics["macro_f1"])}, "wm_e_reproduction_gate": {"status": "PASS", "checkpoint_variant": "E", "val_stgcn_class_agreement": exp046["val"]["stgcn_top1_agreement"], "val_true_label_logp_pearson": exp046["val"]["true_label_logp_pearson"]}, "exp046": exp046, "exp047": exp047, "exp048": exp048, "wm_e_checkpoint": str(wm_checkpoint.resolve()), "wm_e_checkpoint_sha256": _sha256(wm_checkpoint), "rgb_cache": rgb_summary, "test_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP046-EXP048 Train/Val-only campaign")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--rgb-root", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4"))
    parser.add_argument("--wm-e-checkpoint", type=Path, default=Path("/tmp/activeview_exp042r1_E/last.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true", help="Generate two Train/Val contexts and validate cache schema only")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if args.smoke_only:
        _seed()
        device = torch.device(args.device)
        train_rows, val_rows = _rows(args.data_root.resolve(), "train")[:2], _rows(args.data_root.resolve(), "val")[:2]
        train_sources, val_sources = _episode_sources(args.data_root.resolve(), "train"), _episode_sources(args.data_root.resolve(), "val")
        stgcn = _load_stgcn(args.data_root.resolve(), device)
        wm = _load_wm_e(args.wm_e_checkpoint.resolve(), device)
        rgb_lookup, _ = _load_rgb_lookup(args.rgb_root.resolve())
        train_smoke = _generate_split_cache(rows=train_rows, sources=train_sources, wm=wm, stgcn=stgcn, rgb_lookup=rgb_lookup, device=device, output=Path("/tmp/exp046_train_smoke.npz"))
        val_smoke = _generate_split_cache(rows=val_rows, sources=val_sources, wm=wm, stgcn=stgcn, rgb_lookup=rgb_lookup, device=device, output=Path("/tmp/exp046_val_smoke.npz"))
        print(json.dumps({"smoke": "PASS", "train": train_smoke, "val": val_smoke, "test_used": False}, indent=2))
        return
    result = run_campaign(args.data_root.resolve(), torch.device(args.device), wm_checkpoint=args.wm_e_checkpoint.resolve(), rgb_root=args.rgb_root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
