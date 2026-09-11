#!/usr/bin/env python3
"""Fair rerun of reduced12 ST-GCN Policy-Train multi-view fine-tuning.

This is the final exposure/LR audit: each record contributes 16 observations
per epoch (without replacement when possible), while the pretrained ST-GCN is
fine-tuned with conservative differential learning rates.  Only Policy Train
is used for gradient updates and Moving Val is used for checkpoint selection.
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
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification
from activeview.scripts.experiments.run_reduced12_stgcn_policy_multiview_finetune import (
    ADAPTED_HEAD_RELATIVE,
    FEATURE_DIM,
    NUM_CLASSES,
    NUM_VIEWS,
    STGCN_CHECKPOINT_RELATIVE,
    LightweightClassifier,
    _infer_all_views,
    _load_inputs,
    _load_train_rows,
    _per_class_metrics,
    _per_view_metrics,
    _population_metrics,
    _privileged_headroom,
    _record_groups,
    _require_cuda,
    _seed,
    _sha256,
    _spearman,
    _viewpoint_skeleton,
)


SEED = 42
NUM_OBSERVATIONS_PER_RECORD = 16
MAX_EPOCHS = 15
PATIENCE = 4
TRAIN_BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 512
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "stgcn_policy_multiview_finetune_v2"
)
CHECKPOINT_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "stgcn_policy_multiview_finetune_v2/stgcn_finetuned_best.pth"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sample_record_balanced_epoch(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample 16 unique observations per record whenever the pool allows it."""
    rng = np.random.default_rng(SEED + epoch)
    skeletons: list[np.ndarray] = []
    labels: list[int] = []
    sampled_keys: list[tuple[str, int]] = []
    sampled_contexts: list[str] = []
    for record_id in sorted(groups):
        entries: dict[tuple[str, int], tuple[Path, int, str, int]] = {}
        for row in groups[record_id]:
            archive_path = Path(str(row["current_view"]["skeleton_source_path"]))
            current_id = int(row["current_view"]["viewpoint_id"])
            option_ids = [current_id] + [
                int(item["viewpoint_id"])
                for item in row["candidate_pool"]
                if int(item["viewpoint_id"]) != current_id
            ]
            for viewpoint_id in option_ids:
                key = (str(archive_path), viewpoint_id)
                entries.setdefault(
                    key, (archive_path, viewpoint_id, str(row["episode_id"]), int(row["label_id"]))
                )
        if not entries:
            raise ValueError(f"record {record_id} has no train observations")
        values = list(entries.values())
        replace = len(values) < NUM_OBSERVATIONS_PER_RECORD
        selected_indices = rng.choice(
            len(values), size=NUM_OBSERVATIONS_PER_RECORD, replace=replace
        )
        for selected_index in np.asarray(selected_indices).reshape(-1):
            archive_path, viewpoint_id, episode_id, label = values[int(selected_index)]
            skeletons.append(_viewpoint_skeleton(archive_path, viewpoint_id))
            labels.append(label)
            sampled_keys.append((str(archive_path), viewpoint_id))
            sampled_contexts.append(episode_id)
    return (
        np.stack(skeletons).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        {
            "records": len(groups),
            "observations": len(skeletons),
            "unique_records": len(groups),
            "unique_observations": len(set(sampled_keys)),
            "observations_per_record": NUM_OBSERVATIONS_PER_RECORD,
            "sampled_contexts": sampled_contexts,
        },
    )


def _load_archive(archive_path: Path) -> np.ndarray:
    with np.load(archive_path, allow_pickle=False) as archive:
        viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
    if viewpoint_ids.shape != (NUM_VIEWS,) or not np.array_equal(
        viewpoint_ids, np.arange(NUM_VIEWS, dtype=np.int64)
    ):
        raise ValueError(f"non-canonical viewpoint ids in {archive_path}")
    if skeletons.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeletons).all():
        raise ValueError(f"invalid skeleton archive in {archive_path}")
    return skeletons


def _evaluate_val(
    model: torch.nn.Module,
    data: Mapping[str, Any],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate s0, s1 and legal candidates without touching all32 views."""
    model.eval()
    totals = defaultdict(float)
    counts = defaultdict(int)
    s1_ids = np.asarray(
        [int(row["s1_viewpoint_id"]) for row in data["stage_d"]], dtype=np.int64
    )
    for row_index, archive_path in enumerate(data["archive_paths"]):
        skeletons = _load_archive(archive_path)
        legal_ids = [int(value) for value in data["legal_ids"][row_index]]
        current_id = int(data["current_ids"][row_index])
        selected_ids = list(dict.fromkeys([current_id, int(s1_ids[row_index])] + legal_ids))
        selected_skeletons = skeletons[np.asarray(selected_ids, dtype=np.int64)]
        logits_chunks: list[np.ndarray] = []
        for start in range(0, len(selected_skeletons), batch_size):
            batch = torch.from_numpy(selected_skeletons[start:start + batch_size]).to(
                device, non_blocking=True
            )
            with torch.inference_mode():
                logits_chunks.append(model(batch).cpu().numpy())
        logits = np.concatenate(logits_chunks, axis=0)
        index_by_view = {viewpoint_id: index for index, viewpoint_id in enumerate(selected_ids)}
        target = int(labels[row_index])
        for name, viewpoint_id in (("s0", current_id), ("s1", int(s1_ids[row_index]))):
            value = logits[index_by_view[viewpoint_id]][None, :]
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)
            totals[f"{name}_loss"] += float(
                nn.functional.cross_entropy(torch.from_numpy(value).to(device), target_tensor).item()
            )
            counts[f"{name}_n"] += 1
            totals[f"{name}_correct"] += float(int(np.argmax(value[0]) == target))
        legal_logits = logits[[index_by_view[value] for value in legal_ids]]
        legal_targets = torch.full(
            (len(legal_logits),), target, dtype=torch.long, device=device
        )
        totals["legal_loss"] += float(
            nn.functional.cross_entropy(
                torch.from_numpy(legal_logits).to(device), legal_targets, reduction="sum"
            ).item()
        )
        totals["legal_correct"] += float(np.sum(np.argmax(legal_logits, axis=1) == target))
        counts["legal_n"] += len(legal_logits)
    return {
        "s0_loss": totals["s0_loss"] / counts["s0_n"],
        "s1_loss": totals["s1_loss"] / counts["s1_n"],
        "legal_loss": totals["legal_loss"] / counts["legal_n"],
        "s0_accuracy": totals["s0_correct"] / counts["s0_n"],
        "s1_accuracy": totals["s1_correct"] / counts["s1_n"],
        "legal_accuracy": totals["legal_correct"] / counts["legal_n"],
        "legal_observations": counts["legal_n"],
    }


def _train_finetuned(
    model: torch.nn.Module,
    train_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    val_data: Mapping[str, Any],
    val_labels: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter for name, parameter in model.named_parameters()
                    if not name.startswith("fc.")
                ],
                "lr": BACKBONE_LR,
            },
            {"params": list(model.fc.parameters()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    total_steps = 0
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
        gradient_norms: list[float] = []
        epoch_steps = 0
        for batch_skeletons, batch_labels in loader:
            batch_skeletons = batch_skeletons.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_skeletons)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_norms.append(float(norm.detach().cpu()))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            epoch_steps += 1
        total_steps += epoch_steps
        val_stats = _evaluate_val(model, val_data, val_labels, device, INFERENCE_BATCH_SIZE)
        val_loss = float(val_stats["legal_loss"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_s0_accuracy": float(val_stats["s0_accuracy"]),
            "val_s1_accuracy": float(val_stats["s1_accuracy"]),
            "val_legal_accuracy": float(val_stats["legal_accuracy"]),
            "val_legal_loss": val_loss,
            "sampled_records": int(sampling["unique_records"]),
            "sampled_observations": int(sampling["observations"]),
            "unique_observations": int(sampling["unique_observations"]),
            "optimizer_steps": epoch_steps,
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "max_gradient_norm": float(np.max(gradient_norms)),
        }
        history.append(record)
        print(
            f"[stgcn-finetune-v2] epoch={epoch:02d}/{MAX_EPOCHS} "
            f"train_loss={record['train_loss']:.6f} legal_val_loss={val_loss:.6f} "
            f"s0_acc={record['val_s0_accuracy']:.6f} s1_acc={record['val_s1_accuracy']:.6f} "
            f"legal_acc={record['val_legal_accuracy']:.6f} grad={record['mean_gradient_norm']:.6f} "
            f"steps={epoch_steps}",
            flush=True,
        )
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
                    "selection_criterion": "Moving-Val legal candidate global cross-entropy",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"[stgcn-finetune-v2] early_stop epoch={epoch}", flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError("no fine-tuned checkpoint was selected")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "backbone_lr": BACKBONE_LR,
        "classification_head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": TRAIN_BATCH_SIZE,
        "observations_per_record": NUM_OBSERVATIONS_PER_RECORD,
        "seed": SEED,
        "history": history,
        "total_optimizer_steps": total_steps,
        "checkpoint": str(checkpoint_path.resolve()),
        "selection_criterion": "Moving-Val legal candidate global cross-entropy",
    }


def _fixed_view_summary(per_view: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, key in (
        ("Frozen", "frozen_accuracy"),
        ("Adapted", "adapted_accuracy"),
        ("FineTuned", "finetuned_accuracy"),
    ):
        values = np.asarray([float(per_view[str(i)][key]) for i in range(NUM_VIEWS)])
        best = int(np.argmax(values))
        worst = int(np.argmin(values))
        output[name] = {
            "mean_accuracy": float(values.mean()),
            "best_viewpoint": per_view[str(best)]["viewpoint"],
            "best_accuracy": float(values[best]),
            "worst_viewpoint": per_view[str(worst)]["viewpoint"],
            "worst_accuracy": float(values[worst]),
            "range_accuracy": float(values.max() - values.min()),
        }
    return output


def _row(
    frozen: Mapping[str, Any], adapted: Mapping[str, Any], previous: Mapping[str, Any], fine: Mapping[str, Any]
) -> dict[str, Any]:
    result = {"Frozen": dict(frozen), "Adapted": dict(adapted), "PreviousSparse": dict(previous), "FineTuned": dict(fine)}
    result["PreviousSparse"].update({
        "delta_accuracy_pp": 100.0 * (previous["accuracy"] - frozen["accuracy"]),
        "delta_macro_f1_pp": 100.0 * (previous["macro_f1"] - frozen["macro_f1"]),
    })
    result["FineTuned"].update({
        "delta_accuracy_pp": 100.0 * (fine["accuracy"] - frozen["accuracy"]),
        "delta_macro_f1_pp": 100.0 * (fine["macro_f1"] - frozen["macro_f1"]),
    })
    return result


def _write_analysis(output_dir: Path, result: Mapping[str, Any]) -> None:
    table = result["summary_table"]
    lines = [
        "# Reduced12 ST-GCN Policy-Train Fine-tuning v2",
        "",
        "This is the final exposure/LR fairness rerun. The pretrained clean/BABEL reduced12 ST-GCN was fine-tuned end-to-end with 12-class cross entropy. Each record contributes 16 sampled observations per epoch; Moving Val legal-candidate global cross-entropy selects the checkpoint.",
        "",
        "## Data and training",
        "",
        f"Policy Train contexts={result['training']['train_contexts']}, unique records={result['training']['train_unique_records']}, available observations={result['training']['train_available_observations']}. Each epoch samples {result['training']['samples_per_epoch']} observations ({result['training']['observations_per_record']} per record), with unique observations per epoch={result['training']['history'][0]['unique_observations']}. Total optimizer steps={result['training']['total_optimizer_steps']}. Moving Val contexts={result['population']['moving_val_contexts']}, legal observations={result['population']['moving_val_legal_observations']}, all32 observations={result['population']['moving_val_all32_observations']}.",
        "",
        "## Matched Moving-Val results (Acc / global Macro-F1)",
        "",
        "| Population | Frozen | Previous sparse fine-tune | New v2 fine-tune | New Δ vs Frozen |",
        "|---|---:|---:|---:|---:|",
    ]
    for population in ("s0_current", "s1", "legal_candidates_micro", "all32"):
        row = table[population]
        lines.append(
            f"| {population} | {row['Frozen']['accuracy']:.6f}/{row['Frozen']['macro_f1']:.6f} | {row['PreviousSparse']['accuracy']:.6f}/{row['PreviousSparse']['macro_f1']:.6f} | {row['FineTuned']['accuracy']:.6f}/{row['FineTuned']['macro_f1']:.6f} | {row['FineTuned']['delta_accuracy_pp']:+.3f}pp/{row['FineTuned']['delta_macro_f1_pp']:+.3f}pp |"
        )
    headroom = result["headroom"]
    gains = result["viewpoint_analysis"]["fine_tuned_gain_accuracy_pp"]
    fixed = result["viewpoint_analysis"]["fixed_view_accuracy"]
    lines.extend([
        "",
        "## Privileged ceiling",
        "",
        f"GT-TrueLogP Oracle={headroom['GT-TrueLogP Oracle']['accuracy']:.6f}/{headroom['GT-TrueLogP Oracle']['macro_f1']:.6f}; GT-Margin Oracle={headroom['GT-Margin Oracle']['accuracy']:.6f}/{headroom['GT-Margin Oracle']['macro_f1']:.6f}; Legal AnyCorrect coverage={headroom['Legal AnyCorrect Coverage']['rate']:.6f} ({headroom['Legal AnyCorrect Coverage']['count']}/{headroom['Legal AnyCorrect Coverage']['contexts']}). AnyCorrect is coverage only.",
        "",
        "## View generalization",
        "",
        f"FineTuned−Frozen gains: s0={gains['s0']:+.3f}pp, s1={gains['s1']:+.3f}pp, legal={gains['legal']:+.3f}pp, all32={gains['all32']:+.3f}pp; s1−legal={gains['s1_minus_legal']:+.3f}pp; s1−all32={gains['s1_minus_all32']:+.3f}pp.",
        f"Fixed-view mean/best/worst accuracy: Frozen={fixed['Frozen']['mean_accuracy']:.6f}/{fixed['Frozen']['best_viewpoint']} ({fixed['Frozen']['best_accuracy']:.6f})/{fixed['Frozen']['worst_viewpoint']} ({fixed['Frozen']['worst_accuracy']:.6f}); FineTuned={fixed['FineTuned']['mean_accuracy']:.6f}/{fixed['FineTuned']['best_viewpoint']} ({fixed['FineTuned']['best_accuracy']:.6f})/{fixed['FineTuned']['worst_viewpoint']} ({fixed['FineTuned']['worst_accuracy']:.6f}).",
        "",
        "## Training validity",
        "",
        f"Best epoch={result['training']['best_epoch']}; epoch-1/5/10/15 train losses={result['training']['epoch_loss_summary']}; legal-val selection criterion={result['training']['selection_criterion']}.",
    ])
    legal_gain = float(table["legal_candidates_micro"]["FineTuned"]["delta_accuracy_pp"])
    all_gain = float(table["all32"]["FineTuned"]["delta_accuracy_pp"])
    if legal_gain >= 5.0 and all_gain >= 5.0:
        lines.append("The +5pp legal/all32 strong-success criterion is met; this recognizer can become the route-1 baseline.")
    elif legal_gain >= 2.0 or all_gain >= 2.0:
        lines.append("The rerun provides limited +2–5pp domain benefit; do not add fine-tuning complexity without a separate decision.")
    else:
        lines.append("The legal/all32 gain remains below +2pp; full ST-GCN fine-tuning is terminated and should not be further tuned.")
    lines.extend([
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "moving_val_used_for_checkpoint_selection_and_evaluation_only=true",
        "new_rgb_or_skeleton_generated=false",
        "pretrained_initialization=true",
        "```",
    ])
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    _seed()
    train_rows = _load_train_rows(data_root)
    train_groups = _record_groups(train_rows)
    val_data = _load_inputs(data_root)
    val_labels = np.asarray([int(row["label_id"]) for row in val_data["stage_d"]], dtype=np.int64)
    stgcn_path = data_root / STGCN_CHECKPOINT_RELATIVE
    adapted_path = data_root / ADAPTED_HEAD_RELATIVE
    checkpoint_path = data_root / CHECKPOINT_RELATIVE
    frozen_model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    fine_model, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    adapted_payload = torch.load(adapted_path, map_location=device, weights_only=False)
    adapted_head = LightweightClassifier(FEATURE_DIM).to(device)
    adapted_head.load_state_dict(adapted_payload["state_dict"])
    adapted_head.eval()
    frozen_logits, adapted_logits = _infer_all_views(
        val_data, frozen_model, device, INFERENCE_BATCH_SIZE, adapted_head
    )
    if adapted_logits is None:
        raise RuntimeError("adapted head inference unexpectedly returned no logits")
    training = _train_finetuned(fine_model, train_groups, val_data, val_labels, device, checkpoint_path)
    finetuned_logits, _ = _infer_all_views(val_data, fine_model, device, INFERENCE_BATCH_SIZE)
    frozen_metrics = _population_metrics(val_labels, frozen_logits, val_data)
    adapted_metrics = _population_metrics(val_labels, adapted_logits, val_data)
    finetuned_metrics = _population_metrics(val_labels, finetuned_logits, val_data)
    previous_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/stgcn_policy_multiview_finetune/result.json"
    previous_result = _read_json(previous_path)
    previous_metrics = previous_result["metrics"]["FineTuned"]
    table = {
        population: _row(
            frozen_metrics[population], adapted_metrics[population],
            previous_metrics[population], finetuned_metrics[population]
        )
        for population in ("s0_current", "s1", "legal_candidates_micro", "all32")
    }
    context_balanced = {
        name: metrics["legal_candidates_context_balanced"]
        for name, metrics in (
            ("Frozen", frozen_metrics), ("Adapted", adapted_metrics), ("PreviousSparse", previous_metrics),
            ("FineTuned", finetuned_metrics),
        )
    }
    per_view = _per_view_metrics(
        val_labels, frozen_logits, adapted_logits, finetuned_logits,
        np.asarray([int(row["s1_viewpoint_id"]) for row in val_data["stage_d"]], dtype=np.int64),
    )
    gains = np.asarray([float(per_view[str(i)]["finetuned_minus_frozen_accuracy"]) for i in range(NUM_VIEWS)])
    frequency = np.asarray([float(per_view[str(i)]["s1_selection_frequency"]) for i in range(NUM_VIEWS)])
    viewpoint_analysis = {
        "selection_frequency_gain_spearman": _spearman(frequency, gains),
        "fixed_view_accuracy": _fixed_view_summary(per_view),
        "gain_range_accuracy": float(gains.max() - gains.min()),
        "fine_tuned_gain_accuracy_pp": {
            "s0": table["s0_current"]["FineTuned"]["delta_accuracy_pp"],
            "s1": table["s1"]["FineTuned"]["delta_accuracy_pp"],
            "legal": table["legal_candidates_micro"]["FineTuned"]["delta_accuracy_pp"],
            "all32": table["all32"]["FineTuned"]["delta_accuracy_pp"],
            "s1_minus_legal": table["s1"]["FineTuned"]["delta_accuracy_pp"] - table["legal_candidates_micro"]["FineTuned"]["delta_accuracy_pp"],
            "s1_minus_all32": table["s1"]["FineTuned"]["delta_accuracy_pp"] - table["all32"]["FineTuned"]["delta_accuracy_pp"],
        },
    }
    history = training["history"]
    epoch_lookup = {int(record["epoch"]): float(record["train_loss"]) for record in history}
    training.update({
        "train_contexts": len(train_rows),
        "train_unique_records": len(train_groups),
        "train_available_observations": int(sum(1 + len(row["candidate_pool"]) for row in train_rows)),
        "samples_per_epoch": len(train_groups) * NUM_OBSERVATIONS_PER_RECORD,
        "samples_per_record": NUM_OBSERVATIONS_PER_RECORD,
        "epoch_loss_summary": {
            str(epoch): epoch_lookup.get(epoch) for epoch in (1, 5, 10, 15)
        },
        "overfitting_flag": bool(
            len(history) >= 4 and history[-1]["train_loss"] < history[0]["train_loss"]
            and history[-1]["val_loss"] > training["best_val_loss"] + 0.05
        ),
    })
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_STGCN_POLICY_MULTIVIEW_FINETUNE_V2",
        "status": "COMPLETED",
        "population": {
            "split": "Moving Val",
            "moving_val_contexts": len(val_labels),
            "moving_val_legal_observations": int(finetuned_metrics["legal_candidate_observations"]),
            "moving_val_all32_observations": int(finetuned_metrics["all32_observations"]),
        },
        "training": training,
        "checkpoints": {
            "source_frozen_stgcn": str(stgcn_path.resolve()),
            "source_frozen_stgcn_sha256": _sha256(stgcn_path),
            "previous_sparse_result": str(previous_path.resolve()),
            "finetuned_checkpoint": str(checkpoint_path.resolve()),
        },
        "protocol": {
            "num_classes": NUM_CLASSES,
            "labels": list(LABELS),
            "feature_dim": FEATURE_DIM,
            "loss": "12-class cross entropy",
            "sampler": "record_id -> flattened current/legal observations; 16 per record per epoch",
            "validation_selection_criterion": "Moving-Val legal candidate global cross-entropy",
            "backbone_lr": BACKBONE_LR,
            "classification_head_lr": HEAD_LR,
            "all32_population": "observation-level diagnostic only",
        },
        "metrics": {
            "Frozen": frozen_metrics,
            "Adapted": adapted_metrics,
            "PreviousSparse": previous_metrics,
            "FineTuned": finetuned_metrics,
        },
        "summary_table": table,
        "legal_candidates_context_balanced": context_balanced,
        "headroom": _privileged_headroom(val_labels, finetuned_logits, val_data),
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
            "pretrained_initialization": True,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elapsed_seconds": float(time.time() - started),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(training, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "per_view_metrics.json").write_text(
        json.dumps(per_view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps(result["per_class_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps({
        "seed": SEED,
        "samples_per_record": NUM_OBSERVATIONS_PER_RECORD,
        "backbone_lr": BACKBONE_LR,
        "classification_head_lr": HEAD_LR,
        "max_epochs": MAX_EPOCHS,
        "train_split": "Policy Train Stage-A multi-view observations",
        "val_split": "Moving Val Stage-A/Stage-D matched contexts",
        "validation_selection_criterion": "legal candidate global cross-entropy",
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
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_root.resolve(), _require_cuda(args.device))
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "train_contexts": result["training"]["train_contexts"],
        "train_records": result["training"]["train_unique_records"],
        "samples_per_epoch": result["training"]["samples_per_epoch"],
        "total_optimizer_steps": result["training"]["total_optimizer_steps"],
        "moving_val_contexts": result["population"]["moving_val_contexts"],
        "best_epoch": result["training"]["best_epoch"],
        "summary_table": result["summary_table"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
