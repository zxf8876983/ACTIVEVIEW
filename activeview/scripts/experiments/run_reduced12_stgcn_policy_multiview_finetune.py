#!/usr/bin/env python3
"""Fine-tune the pretrained reduced12 ST-GCN on Policy-Train views.

The experiment starts from the clean/BABEL reduced12 checkpoint, samples one
current-or-legal-candidate observation per ``record_id`` on every epoch, and
evaluates the resulting recognizer on the matched Moving Val population.  No
Policy Test artifact, new perception data, selector, or auxiliary loss is
used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced12_matched_privileged_oracle import (
    ADAPTED_HEAD_RELATIVE, NUM_CLASSES, NUM_VIEWS, STGCN_CHECKPOINT_RELATIVE,
    _load_inputs, _require_cuda, _seed, _sha256,
)
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, gt_margin
from activeview.scripts.experiments.run_reduced12_single_view_classifier_adaptation import (
    FEATURE_DIM,
    LightweightClassifier,
)


SEED = 42
MAX_EPOCHS = 15
PATIENCE = 4
TRAIN_BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 512
BACKBONE_LR = 1e-4
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/stgcn_policy_multiview_finetune"
FINE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/stgcn_policy_multiview_finetune/stgcn_finetuned_best.pth"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                rows.append(value)
    return rows


def _load_train_rows(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_a/episodes/train_episodes.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"empty Policy Train Stage-A file: {path}")
    for row in rows:
        if not row.get("candidate_pool"):
            raise ValueError(f"missing legal candidate_pool in {row['episode_id']}")
    return rows


def _record_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["record_id"])].append(row)
    return dict(groups)


def _viewpoint_skeleton(archive_path: Path, viewpoint_id: int) -> np.ndarray:
    with np.load(archive_path, allow_pickle=False) as archive:
        viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
    if viewpoint_ids.shape != (NUM_VIEWS,) or not np.array_equal(
        viewpoint_ids, np.arange(NUM_VIEWS, dtype=np.int64)
    ):
        raise ValueError(f"non-canonical viewpoint ids in {archive_path}")
    if skeletons.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeletons).all():
        raise ValueError(f"invalid skeleton archive in {archive_path}")
    return skeletons[int(viewpoint_id)]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size < 2 or right_values.size != left_values.size:
        return None
    left_ranks = _rank(left_values)
    right_ranks = _rank(right_values)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _sample_record_balanced_epoch(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(SEED + epoch)
    skeletons: list[np.ndarray] = []
    labels: list[int] = []
    sampled_contexts: list[str] = []
    for record_id in sorted(groups):
        contexts = groups[record_id]
        row = contexts[int(rng.integers(0, len(contexts)))]
        archive_path = Path(str(row["current_view"]["skeleton_source_path"]))
        current_id = int(row["current_view"]["viewpoint_id"])
        candidate_ids = [int(item["viewpoint_id"]) for item in row["candidate_pool"]]
        options = [current_id] + [value for value in candidate_ids if value != current_id]
        viewpoint_id = int(options[int(rng.integers(0, len(options)))])
        skeletons.append(_viewpoint_skeleton(archive_path, viewpoint_id))
        labels.append(int(row["label_id"]))
        sampled_contexts.append(str(row["episode_id"]))
    if not skeletons:
        raise ValueError("record-balanced sampler produced no observations")
    return (
        np.stack(skeletons).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        {
            "records": len(groups),
            "observations": len(skeletons),
            "unique_records_sampled": len(set(groups)),
            "sampled_contexts": sampled_contexts,
        },
    )


def _fixed_val_selection(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a deterministic two-view-per-record validation selection set."""
    rng = np.random.default_rng(SEED + 100_000)
    skeletons: list[np.ndarray] = []
    labels: list[int] = []
    for record_id in sorted(groups):
        row = groups[record_id][0]
        archive_path = Path(str(row["current_view"]["skeleton_source_path"]))
        current_id = int(row["current_view"]["viewpoint_id"])
        candidates = [int(item["viewpoint_id"]) for item in row["candidate_pool"]]
        candidate_id = candidates[int(rng.integers(0, len(candidates)))]
        for viewpoint_id in (current_id, candidate_id):
            skeletons.append(_viewpoint_skeleton(archive_path, viewpoint_id))
            labels.append(int(row["label_id"]))
    return (
        np.stack(skeletons).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        {"records": len(groups), "observations": len(skeletons), "views_per_record": 2},
    )


def _train_finetuned(
    model: torch.nn.Module,
    train_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    val_skeletons: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": BACKBONE_LR},
            {"params": list(model.fc.parameters()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    val_tensor = torch.from_numpy(val_skeletons).to(device, non_blocking=True)
    val_target = torch.from_numpy(val_labels).to(device, non_blocking=True)
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        skeletons, labels, sampling = _sample_record_balanced_epoch(train_groups, epoch)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(skeletons), torch.from_numpy(labels)),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        model.train()
        train_losses: list[float] = []
        for batch_skeletons, batch_labels in loader:
            batch_skeletons = batch_skeletons.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_skeletons)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_tensor)
            val_loss = float(nn.functional.cross_entropy(val_logits, val_target).item())
            val_predictions = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_metrics = classification(val_labels, val_predictions)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "sampled_records": int(sampling["records"]),
            "sampled_observations": int(sampling["observations"]),
        }
        history.append(record)
        print(f"[stgcn-finetune] epoch={epoch:02d}/{MAX_EPOCHS} train_loss={record['train_loss']:.6f} val_loss={val_loss:.6f} val_acc={record['val_accuracy']:.6f} val_f1={record['val_macro_f1']:.6f}", flush=True)
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": SEED,
                    "num_classes": NUM_CLASSES,
                    "feature_dim": FEATURE_DIM,
                    "source": "stgcn_reduced12_no_kneel_clean_best.pth",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"[stgcn-finetune] early_stop epoch={epoch}", flush=True)
                break
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    summary: dict[str, Any] = {
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "backbone_lr": BACKBONE_LR,
        "classification_head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": TRAIN_BATCH_SIZE,
        "seed": SEED,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    return summary


def _infer_all_views(
    data: Mapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    adapted_head: LightweightClassifier | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    count = len(data["stage_d"])
    model.eval()
    logits = np.empty((count, NUM_VIEWS, NUM_CLASSES), dtype=np.float32)
    adapted_logits = np.empty_like(logits) if adapted_head is not None else None
    if adapted_head is not None:
        adapted_head.eval()
    for row_index, archive_path in enumerate(data["archive_paths"]):
        with np.load(archive_path, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
        if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)):
            raise ValueError(f"non-canonical viewpoint ids in {archive_path}")
        if skeletons.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeletons).all():
            raise ValueError(f"invalid skeleton archive in {archive_path}")
        chunks: list[np.ndarray] = []
        adapted_chunks: list[np.ndarray] = []
        for start in range(0, NUM_VIEWS, batch_size):
            batch = torch.from_numpy(skeletons[start:start + batch_size]).to(device, non_blocking=True)
            with torch.inference_mode():
                features = model.forward_features(batch)
                chunks.append(model.fc(features).cpu().numpy())
                if adapted_head is not None:
                    adapted_chunks.append(adapted_head(features).cpu().numpy())
        logits[row_index] = np.concatenate(chunks, axis=0)
        if adapted_logits is not None:
            adapted_logits[row_index] = np.concatenate(adapted_chunks, axis=0)
        if (row_index + 1) % 500 == 0:
            print(f"[view-inference] archives={row_index + 1}/{count}", flush=True)
    return logits, adapted_logits


def _classification_for_view(
    logits: np.ndarray,
    labels: np.ndarray,
    viewpoint_ids: np.ndarray,
) -> dict[str, Any]:
    predictions = np.argmax(logits[np.arange(labels.size), viewpoint_ids], axis=1)
    return classification(labels, predictions)


def _population_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    current_ids = np.asarray(data["current_ids"], dtype=np.int64)
    s1_ids = np.asarray([int(row["s1_viewpoint_id"]) for row in data["stage_d"]], dtype=np.int64)
    s0 = _classification_for_view(logits, labels, current_ids)
    s1 = _classification_for_view(logits, labels, s1_ids)
    legal_predictions: list[int] = []
    legal_labels: list[int] = []
    context_ratios: list[float] = []
    for index, ids in enumerate(data["legal_ids"]):
        values = np.asarray([int(value) for value in ids], dtype=np.int64)
        predictions = np.argmax(logits[index, values], axis=1)
        legal_predictions.extend(predictions.tolist())
        legal_labels.extend([int(labels[index])] * len(values))
        context_ratios.append(float(np.mean(predictions == labels[index])))
    legal_y = np.asarray(legal_labels, dtype=np.int64)
    legal_p = np.asarray(legal_predictions, dtype=np.int64)
    legal_micro = classification(legal_y, legal_p)
    all_y = np.repeat(labels, NUM_VIEWS)
    all_p = np.argmax(logits, axis=2).reshape(-1)
    all32 = classification(all_y, all_p)
    return {
        "s0_current": s0,
        "s1": s1,
        "legal_candidates_micro": legal_micro,
        "legal_candidates_context_balanced": {
            "contexts": int(labels.size),
            "accuracy": float(np.mean(context_ratios)),
        },
        "all32": all32,
        "legal_candidate_observations": int(legal_y.size),
        "all32_observations": int(all_y.size),
    }


def _per_class_metrics(
    labels: np.ndarray,
    data: Mapping[str, Any],
    logits_by_name: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    current_ids = np.asarray(data["current_ids"], dtype=np.int64)
    s1_ids = np.asarray([int(row["s1_viewpoint_id"]) for row in data["stage_d"]], dtype=np.int64)
    output: dict[str, Any] = {}
    for population in ("s0_current", "s1", "legal_candidates_micro", "all32"):
        output[population] = {}
        for name, logits in logits_by_name.items():
            if population == "s0_current":
                metric = _classification_for_view(logits, labels, current_ids)
            elif population == "s1":
                metric = _classification_for_view(logits, labels, s1_ids)
            elif population == "all32":
                metric = classification(
                    np.repeat(labels, NUM_VIEWS), np.argmax(logits, axis=2).reshape(-1)
                )
            else:
                predictions: list[int] = []
                targets: list[int] = []
                for index, ids in enumerate(data["legal_ids"]):
                    values = np.asarray([int(value) for value in ids], dtype=np.int64)
                    predictions.extend(np.argmax(logits[index, values], axis=1).tolist())
                    targets.extend([int(labels[index])] * len(values))
                metric = classification(targets, predictions)
            output[population][name] = metric["per_class"]
    return output


def _per_view_metrics(
    labels: np.ndarray,
    frozen_logits: np.ndarray,
    adapted_logits: np.ndarray,
    finetuned_logits: np.ndarray,
    s1_ids: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for viewpoint_id in range(NUM_VIEWS):
        frozen = classification(labels, np.argmax(frozen_logits[:, viewpoint_id], axis=1))
        adapted = classification(labels, np.argmax(adapted_logits[:, viewpoint_id], axis=1))
        fine = classification(labels, np.argmax(finetuned_logits[:, viewpoint_id], axis=1))
        output[str(viewpoint_id)] = {
            "viewpoint_id": viewpoint_id,
            "viewpoint": f"r{viewpoint_id // 8 + 1}_a{viewpoint_id % 8}",
            "frozen_accuracy": frozen["accuracy"],
            "adapted_accuracy": adapted["accuracy"],
            "finetuned_accuracy": fine["accuracy"],
            "adapted_minus_frozen_accuracy": adapted["accuracy"] - frozen["accuracy"],
            "finetuned_minus_frozen_accuracy": fine["accuracy"] - frozen["accuracy"],
            "s1_selection_frequency": float(np.mean(s1_ids == viewpoint_id)),
        }
    return output


def _privileged_headroom(
    labels: np.ndarray,
    logits: np.ndarray,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    current_ids = np.asarray(data["current_ids"], dtype=np.int64)
    current = logits[np.arange(labels.size), current_ids]
    any_correct = np.zeros(labels.size, dtype=bool)
    selected_true: list[int] = []
    selected_margin: list[int] = []
    true_predictions: list[int] = []
    margin_predictions: list[int] = []
    for index, label in enumerate(labels):
        ids = [int(value) for value in data["legal_ids"][index]]
        options = np.concatenate((current[index][None, :], logits[index, ids]), axis=0)
        argmaxes = np.argmax(options, axis=1)
        any_correct[index] = bool(np.any(argmaxes == int(label)))
        true_scores = options[:, int(label)]
        margins = np.asarray([gt_margin(value, int(label)) for value in options])
        true_choice = int(np.argmax(true_scores))
        margin_choice = int(np.argmax(margins))
        selected_true.append(true_choice)
        selected_margin.append(margin_choice)
        true_predictions.append(int(argmaxes[true_choice]))
        margin_predictions.append(int(argmaxes[margin_choice]))
    true_metric = classification(labels, true_predictions)
    margin_metric = classification(labels, margin_predictions)
    return {
        "GT-TrueLogP Oracle": true_metric,
        "GT-Margin Oracle": margin_metric,
        "Legal AnyCorrect Coverage": {
            "contexts": int(labels.size),
            "count": int(np.sum(any_correct)),
            "rate": float(np.mean(any_correct)),
        },
    }


def _write_analysis(output_dir: Path, result: Mapping[str, Any]) -> None:
    table = result["summary_table"]
    lines = [
        "# Reduced12 Policy-Multiview ST-GCN Fine-tuning Audit",
        "",
        "The pretrained clean/BABEL reduced12 ST-GCN was initialized from the existing checkpoint and fine-tuned end-to-end with 12-class cross entropy on Policy Train observations. Every epoch samples exactly one current-or-legal-candidate observation per unique record_id. Moving Val is evaluation/model-selection only.",
        "",
        "## Data and sampling",
        "",
        f"Policy Train contexts={result['training']['train_contexts']}, unique records={result['training']['train_unique_records']}, available current+legal observations={result['training']['train_available_observations']}. Record-balanced sampler: {result['training']['sampler']}; each epoch samples {result['training']['samples_per_epoch']} observations ({result['training']['samples_per_record_min']}–{result['training']['samples_per_record_max']} per record). Moving Val contexts={result['population']['moving_val_contexts']}, legal observations={result['population']['moving_val_legal_observations']}, all-32 observations={result['population']['moving_val_all32_observations']}.",
        "",
        "## Matched Moving-Val results",
        "",
        "| Population | Frozen Acc/F1 | Existing adapted-head Acc/F1 | Fine-tuned Acc/F1 | Fine-tuned ΔAcc/ΔF1 vs Frozen |",
        "|---|---:|---:|---:|---:|",
    ]
    for population in ("s0_current", "s1", "legal_candidates_micro", "all32"):
        row = table[population]
        lines.append(
            f"| {population} | {row['Frozen']['accuracy']:.6f}/{row['Frozen']['macro_f1']:.6f} | {row['Adapted']['accuracy']:.6f}/{row['Adapted']['macro_f1']:.6f} | {row['FineTuned']['accuracy']:.6f}/{row['FineTuned']['macro_f1']:.6f} | {row['FineTuned']['delta_accuracy_pp']:+.3f}pp/{row['FineTuned']['delta_macro_f1_pp']:+.3f}pp |"
        )
    headroom = result["headroom"]
    lines.extend([
        "",
        "## Fine-tuned privileged headroom",
        "",
        f"GT-TrueLogP Oracle: {headroom['GT-TrueLogP Oracle']['accuracy']:.6f}/{headroom['GT-TrueLogP Oracle']['macro_f1']:.6f}; GT-Margin Oracle: {headroom['GT-Margin Oracle']['accuracy']:.6f}/{headroom['GT-Margin Oracle']['macro_f1']:.6f}; Legal AnyCorrect Coverage: {headroom['Legal AnyCorrect Coverage']['rate']:.6f} ({headroom['Legal AnyCorrect Coverage']['count']}/{headroom['Legal AnyCorrect Coverage']['contexts']}).",
        "AnyCorrect is coverage only and is not reported as Oracle Accuracy.",
        "",
        "## Viewpoint specialization",
        "",
        f"Spearman(s1 selection frequency, FineTuned−Frozen per-view gain)={result['viewpoint_analysis']['selection_frequency_gain_spearman']!s}; best gain views={', '.join(result['viewpoint_analysis']['best_gain_viewpoints'])}; worst gain views={', '.join(result['viewpoint_analysis']['worst_gain_viewpoints'])}.",
        f"Fixed-view mean/best/worst accuracy: Frozen={result['viewpoint_analysis']['fixed_view_accuracy']['Frozen']['mean_accuracy']:.6f}/{result['viewpoint_analysis']['fixed_view_accuracy']['Frozen']['best_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['Frozen']['best_accuracy']:.6f})/{result['viewpoint_analysis']['fixed_view_accuracy']['Frozen']['worst_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['Frozen']['worst_accuracy']:.6f}); Adapted={result['viewpoint_analysis']['fixed_view_accuracy']['Adapted']['mean_accuracy']:.6f}/{result['viewpoint_analysis']['fixed_view_accuracy']['Adapted']['best_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['Adapted']['best_accuracy']:.6f})/{result['viewpoint_analysis']['fixed_view_accuracy']['Adapted']['worst_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['Adapted']['worst_accuracy']:.6f}); FineTuned={result['viewpoint_analysis']['fixed_view_accuracy']['FineTuned']['mean_accuracy']:.6f}/{result['viewpoint_analysis']['fixed_view_accuracy']['FineTuned']['best_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['FineTuned']['best_accuracy']:.6f})/{result['viewpoint_analysis']['fixed_view_accuracy']['FineTuned']['worst_viewpoint']} ({result['viewpoint_analysis']['fixed_view_accuracy']['FineTuned']['worst_accuracy']:.6f}).",
        f"FineTuned−Frozen Accuracy gains: s0={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['s0']:+.3f}pp, s1={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['s1']:+.3f}pp, legal={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['legal']:+.3f}pp, all32={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['all32']:+.3f}pp; s1−legal={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['s1_minus_legal']:+.3f}pp; s1−all32={result['viewpoint_analysis']['fine_tuned_gain_accuracy_pp']['s1_minus_all32']:+.3f}pp.",
        "",
        "## Training and decision",
        "",
        f"Best epoch={result['training']['best_epoch']}, train loss={result['training']['best_epoch_train_loss']:.6f}, Val loss={result['training']['best_val_loss']:.6f}; overfitting flag={result['training']['overfitting_flag']}.",
    ])
    legal_gain = table["legal_candidates_micro"]["FineTuned"]["delta_accuracy_pp"]
    all_gain = table["all32"]["FineTuned"]["delta_accuracy_pp"]
    s1_gain = table["s1"]["FineTuned"]["delta_accuracy_pp"]
    if legal_gain >= 5.0 and all_gain >= 5.0:
        lines.append("The fine-tuned recognizer meets the strong +5pp legal/all-32 criterion and is a candidate for a new view-agnostic recognizer baseline.")
    elif legal_gain >= 2.0 or all_gain >= 2.0:
        lines.append("The fine-tuning yields a limited +2–5pp domain benefit; it is not a strong view-agnostic breakthrough and should not be complexified without a separate decision.")
    elif s1_gain - max(legal_gain, all_gain) >= 2.0:
        lines.append("The fine-tuning is s1-specialized: the matched s1 gain is larger than legal/all-32 gains, so the adapted improvement does not generalize broadly.")
    else:
        lines.append("The fine-tuning does not produce a meaningful legal/all-32 gain; this route should be stopped under the predefined淘汰标准.")
    lines.extend([
        "",
        "Per-class and per-view deltas are stored separately; no class, viewpoint, split, taxonomy, or existing checkpoint was modified.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "moving_val_used_for_model_selection_and_evaluation_only=true",
        "training_used=policy_train_only",
        "new_rgb_or_skeleton_generated=false",
        "selector_or_fusion_used=false",
        "pretrained_initialization=true",
        "```",
        "",
    ])
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    output_dir: Path,
    data_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    _seed()
    train_rows = _load_train_rows(data_root)
    train_groups = _record_groups(train_rows)
    val_data = _load_inputs(data_root)
    val_labels = np.asarray([int(row["label_id"]) for row in val_data["stage_d"]], dtype=np.int64)
    val_rows = [val_data["stage_a_by_id"][str(row["episode_id"])] for row in val_data["stage_d"]]
    val_groups = _record_groups(val_rows)
    val_selection_skeletons, val_selection_labels, val_selection_summary = _fixed_val_selection(val_groups)
    stgcn_path = data_root / STGCN_CHECKPOINT_RELATIVE
    adapted_path = data_root / ADAPTED_HEAD_RELATIVE
    fine_path = data_root / FINE_CHECKPOINT_RELATIVE
    frozen_model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    fine_model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    adapted_payload = torch.load(adapted_path, map_location=device, weights_only=False)
    adapted_head = LightweightClassifier(FEATURE_DIM).to(device)
    adapted_head.load_state_dict(adapted_payload["state_dict"])
    adapted_head.eval()
    frozen_logits, adapted_logits = _infer_all_views(
        val_data, frozen_model, device, batch_size, adapted_head
    )
    if adapted_logits is None:
        raise RuntimeError("adapted head inference unexpectedly returned no logits")
    training = _train_finetuned(
        fine_model, train_groups, val_selection_skeletons, val_selection_labels,
        device, fine_path,
    )
    finetuned_logits, _ = _infer_all_views(val_data, fine_model, device, batch_size)
    frozen_metrics = _population_metrics(val_labels, frozen_logits, val_data)
    adapted_metrics = _population_metrics(val_labels, adapted_logits, val_data)
    fine_metrics = _population_metrics(val_labels, finetuned_logits, val_data)

    def row_for(population: str) -> dict[str, Any]:
        values = {
            "Frozen": frozen_metrics[population],
            "Adapted": adapted_metrics[population],
            "FineTuned": fine_metrics[population],
        }
        for value in values.values():
            if "accuracy" not in value or "macro_f1" not in value:
                raise ValueError(f"population {population} lacks global accuracy/F1")
        return {
            **values,
            "FineTuned": {
                **values["FineTuned"],
                "delta_accuracy_pp": 100.0 * (values["FineTuned"]["accuracy"] - values["Frozen"]["accuracy"]),
                "delta_macro_f1_pp": 100.0 * (values["FineTuned"]["macro_f1"] - values["Frozen"]["macro_f1"]),
            },
        }

    # Context-balanced candidate accuracy is intentionally reported separately
    # and is not assigned a pseudo Macro-F1.
    summary_table = {
        "s0_current": row_for("s0_current"),
        "s1": row_for("s1"),
        "legal_candidates_micro": row_for("legal_candidates_micro"),
        "all32": row_for("all32"),
    }
    context_balanced = {
        name: metrics["legal_candidates_context_balanced"]
        for name, metrics in (
            ("Frozen", frozen_metrics), ("Adapted", adapted_metrics), ("FineTuned", fine_metrics)
        )
    }
    summary_table["legal_candidates_context_balanced"] = {
        **context_balanced,
        "FineTuned": {
            **context_balanced["FineTuned"],
            "delta_accuracy_pp": 100.0 * (context_balanced["FineTuned"]["accuracy"] - context_balanced["Frozen"]["accuracy"]),
        },
    }
    per_view = _per_view_metrics(
        val_labels,
        frozen_logits,
        adapted_logits,
        finetuned_logits,
        np.asarray([int(row["s1_viewpoint_id"]) for row in val_data["stage_d"]], dtype=np.int64),
    )
    gains = np.asarray([float(per_view[str(i)]["finetuned_minus_frozen_accuracy"]) for i in range(NUM_VIEWS)])
    frequency = np.asarray([float(per_view[str(i)]["s1_selection_frequency"]) for i in range(NUM_VIEWS)])
    best_ids = np.argsort(-gains, kind="mergesort")[:5]
    worst_ids = np.argsort(gains, kind="mergesort")[:5]
    fixed_view_accuracy: dict[str, dict[str, Any]] = {}
    for name, key in (
        ("Frozen", "frozen_accuracy"),
        ("Adapted", "adapted_accuracy"),
        ("FineTuned", "finetuned_accuracy"),
    ):
        values = np.asarray([float(per_view[str(i)][key]) for i in range(NUM_VIEWS)])
        best_view = int(np.argmax(values))
        worst_view = int(np.argmin(values))
        fixed_view_accuracy[name] = {
            "mean_accuracy": float(values.mean()),
            "best_viewpoint": per_view[str(best_view)]["viewpoint"],
            "best_accuracy": float(values[best_view]),
            "worst_viewpoint": per_view[str(worst_view)]["viewpoint"],
            "worst_accuracy": float(values[worst_view]),
            "range_accuracy": float(values.max() - values.min()),
        }
    fine_s1_gain = 100.0 * (
        fine_metrics["s1"]["accuracy"] - frozen_metrics["s1"]["accuracy"]
    )
    fine_legal_gain = 100.0 * (
        fine_metrics["legal_candidates_micro"]["accuracy"]
        - frozen_metrics["legal_candidates_micro"]["accuracy"]
    )
    fine_all32_gain = 100.0 * (
        fine_metrics["all32"]["accuracy"] - frozen_metrics["all32"]["accuracy"]
    )
    viewpoint_analysis = {
        "selection_frequency_gain_spearman": _spearman(frequency, gains),
        "best_gain_viewpoints": [per_view[str(int(i))]["viewpoint"] for i in best_ids],
        "worst_gain_viewpoints": [per_view[str(int(i))]["viewpoint"] for i in worst_ids],
        "gain_range_accuracy": float(gains.max() - gains.min()),
        "fixed_view_accuracy": fixed_view_accuracy,
        "fine_tuned_gain_accuracy_pp": {
            "s0": 100.0 * (
                fine_metrics["s0_current"]["accuracy"]
                - frozen_metrics["s0_current"]["accuracy"]
            ),
            "s1": fine_s1_gain,
            "legal": fine_legal_gain,
            "all32": fine_all32_gain,
            "s1_minus_legal": fine_s1_gain - fine_legal_gain,
            "s1_minus_all32": fine_s1_gain - fine_all32_gain,
        },
    }
    headroom = _privileged_headroom(val_labels, finetuned_logits, val_data)
    history = training["history"]
    best_record = next(record for record in history if record["epoch"] == training["best_epoch"])
    train_losses = np.asarray([float(record["train_loss"]) for record in history])
    val_losses = np.asarray([float(record["val_loss"]) for record in history])
    overfitting = bool(
        len(history) >= 4 and train_losses[-1] < train_losses[0]
        and val_losses[-1] > training["best_val_loss"] + 0.05
    )
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_STGCN_POLICY_MULTIVIEW_FINETUNE",
        "status": "COMPLETED",
        "population": {
            "split": "Moving Val",
            "moving_val_contexts": int(len(val_labels)),
            "moving_val_legal_observations": int(fine_metrics["legal_candidate_observations"]),
            "moving_val_all32_observations": int(fine_metrics["all32_observations"]),
            "moving_val_unique_records": int(len(val_groups)),
        },
        "training": {
            **training,
            "train_contexts": int(len(train_rows)),
            "train_unique_records": int(len(train_groups)),
            "train_available_observations": int(sum(1 + len(row["candidate_pool"]) for row in train_rows)),
            "train_available_candidate_observations": int(sum(len(row["candidate_pool"]) for row in train_rows)),
            "sampler": "one uniformly sampled context then one uniformly sampled current/stay-or-legal-candidate viewpoint per record_id per epoch",
            "samples_per_epoch": int(len(train_groups)),
            "samples_per_record_min": 1,
            "samples_per_record_max": 1,
            "val_selection": val_selection_summary,
            "best_epoch_train_loss": float(best_record["train_loss"]),
            "overfitting_flag": overfitting,
        },
        "checkpoints": {
            "source_frozen_stgcn": str(stgcn_path.resolve()),
            "source_frozen_stgcn_sha256": _sha256(stgcn_path),
            "existing_adapted_head": str(adapted_path.resolve()),
            "existing_adapted_head_sha256": _sha256(adapted_path),
            "finetuned_checkpoint": str(fine_path.resolve()),
        },
        "protocol": {
            "num_classes": NUM_CLASSES,
            "labels": list(LABELS),
            "feature_dim": FEATURE_DIM,
            "legal_action_set": "Stage-A candidate_pool observations; no archive navigation substitution",
            "all32_population": "observation-level diagnostic only",
            "loss": "12-class cross entropy",
            "initialization": "pretrained clean/BABEL reduced12 checkpoint",
        },
        "metrics": {
            "Frozen": frozen_metrics,
            "Adapted": adapted_metrics,
            "FineTuned": fine_metrics,
        },
        "summary_table": summary_table,
        "legal_candidates_context_balanced": context_balanced,
        "headroom": headroom,
        "viewpoint_analysis": viewpoint_analysis,
        "per_class_metrics": _per_class_metrics(
            val_labels, val_data,
            {"Frozen": frozen_logits, "Adapted": adapted_logits, "FineTuned": finetuned_logits},
        ),
        "leakage_audit": {
            "policy_test_used": False,
            "moving_val_used_for_gradient_updates": False,
            "new_rgb_or_skeleton_generated": False,
            "selector_or_active_policy_used": False,
            "fusion_used": False,
            "future_candidate_observation_used_for_training_input": False,
            "pretrained_initialization": True,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "inference_batch_size": batch_size,
            "elapsed_seconds": float(time.time() - started),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(result["training"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "per_view_metrics.json").write_text(
        json.dumps(per_view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps(result["per_class_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps({
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "backbone_lr": BACKBONE_LR,
        "classification_head_lr": HEAD_LR,
        "train_split": "Policy Train Stage-A multi-view observations",
        "val_split": "Moving Val Stage-A/Stage-D matched contexts",
        "policy_test_used": False,
        "new_rgb_or_skeleton_generated": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(output_dir, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    args = parser.parse_args()
    result = run(
        args.output_dir.resolve(), args.data_root.resolve(),
        _require_cuda(args.device), args.batch_size,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "train_contexts": result["training"]["train_contexts"],
        "train_records": result["training"]["train_unique_records"],
        "moving_val_contexts": result["population"]["moving_val_contexts"],
        "best_epoch": result["training"]["best_epoch"],
        "summary_table": result["summary_table"],
        "headroom": result["headroom"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
