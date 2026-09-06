#!/usr/bin/env python3
"""Train the pretrained-history identity and JR variants on Train/Val only.

The formal WM-E and ST-GCN artifacts remain frozen.  A 540->256->128 history
identity encoder is selected on Val identity Macro-F1, then used to initialise
two independent Multi-positive JR runs: frozen identity and fine-tuned
identity.  Test paths are intentionally absent from this entry point.
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
from activeview.methods.joint_revision.history_aware import (
    HISTORY_LATENT_DIM,
    HistoryIdentityEncoder,
)
from activeview.methods.joint_revision.model import JointRevision, select_actions
from activeview.methods.joint_revision.pretrained_history_aware import (
    PretrainedHistoryAwareJointRevision,
)
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _metrics,
    _terminal_predictions,
    _validate_rows_cache,
)
from activeview.scripts.train.train_reduced14_history_aware_jr import (
    _history_inputs,
    _load_normal_jr,
    _load_prior_methods,
    _method_metrics,
)
from activeview.scripts.train.train_reduced14_joint_revision import _examples, _orders


SEED = 42
NUM_CLASSES = 14
EPOCHS_IDENTITY = 20
EPOCHS_JR = 20
BATCH_SIZE = 512
IDENTITY_LR = 1e-3
JR_LR = 1e-3
FINETUNE_IDENTITY_LR = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_ID = 0.2
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/pretrained_history_jr"
CHECKPOINT_DIR_NAME = "activeview_reduced14_eight_placement_v1"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _IdentityDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, history: np.ndarray, labels: np.ndarray) -> None:
        self.history = history
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": torch.from_numpy(self.history[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class _JRDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: tuple[np.ndarray, ...], history: np.ndarray) -> None:
        self.current, self.candidates, self.mask, self.fallback, self.positive, self.labels = arrays
        self.history = history

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current": torch.from_numpy(self.current[index]),
            "candidates": torch.from_numpy(self.candidates[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "fallback": torch.tensor(int(self.fallback[index]), dtype=torch.long),
            "positive": torch.from_numpy(self.positive[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "history": torch.from_numpy(self.history[index]),
        }


def _classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(
        labels * NUM_CLASSES + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0
        recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "count": int(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean(f1)),
    }


def _train_identity(
    train_history: np.ndarray,
    train_labels: np.ndarray,
    val_history: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _seed()
    model = HistoryIdentityEncoder(NUM_CLASSES).to(device)
    loader = DataLoader(
        _IdentityDataset(train_history, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=IDENTITY_LR)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_key = (-np.inf, -np.inf)
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS_IDENTITY + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            logits = model(batch["history"].float().to(device))[1]
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(val_history), BATCH_SIZE):
                logits = model(
                    torch.from_numpy(val_history[start : start + BATCH_SIZE]).to(device)
                )[1]
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        metrics = _classification_metrics(np.concatenate(predictions), val_labels)
        stats = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_accuracy": float(metrics["accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
        }
        history.append(stats)
        print(
            f"identity epoch {epoch}/{EPOCHS_IDENTITY} loss={stats['loss']:.6f} "
            f"val_acc={stats['val_accuracy']:.6f} val_f1={stats['val_macro_f1']:.6f}",
            flush=True,
        )
        key = (stats["val_macro_f1"], stats["val_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "input_dim": 540,
                    "seed": SEED,
                    "epoch": epoch,
                    "val_identity": metrics,
                },
                checkpoint,
            )

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return (
        {
            "epochs": EPOCHS_IDENTITY,
            "batch_size": BATCH_SIZE,
            "learning_rate": IDENTITY_LR,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_val_identity": best_metrics,
            "final_loss": history[-1]["loss"],
            "loss_history": history,
        },
        model.state_dict(),
    )


def _jr_loss(
    scores: torch.Tensor,
    posterior: torch.Tensor,
    identity_logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = batch["mask"].bool().to(scores.device)
    fallback = batch["fallback"].long().to(scores.device)
    positive = batch["positive"].float().to(scores.device)
    labels = batch["label"].long().to(scores.device)
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
        posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none"
    ).reshape_as(scores)
    post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    jr_loss = (main + 0.25 * bce + 0.05 * post).mean()
    identity_loss = nn.functional.cross_entropy(identity_logits, labels)
    return jr_loss + LAMBDA_ID * identity_loss, jr_loss.detach(), identity_loss.detach()


def _select_actions(
    model: PretrainedHistoryAwareJointRevision,
    arrays: tuple[np.ndarray, ...],
    rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]],
    device: torch.device,
) -> tuple[list[int | None], np.ndarray]:
    current, candidates, mask, _fallback, _positive, _labels, history = arrays
    actions: list[int | None] = []
    identity_predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(rows))
            scores, _posterior, identity_logits = model(
                torch.from_numpy(current[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(candidates[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(mask[start:stop]).to(device=device, dtype=torch.bool),
                torch.from_numpy(history[start:stop]).to(device=device, dtype=torch.float32),
            )
            batch_mask = torch.from_numpy(mask[start:stop]).to(device)
            choices = scores.masked_fill(~batch_mask, -1e9).argmax(dim=1).cpu().numpy()
            identity_predictions.append(identity_logits.argmax(dim=1).cpu().numpy())
            for offset, choice in enumerate(choices.tolist()):
                episode_order = list(orders[str(rows[start + offset]["episode_id"])])
                actions.append(None if choice == 0 else int(episode_order[choice - 1]))
    return actions, np.concatenate(identity_predictions, axis=0)


def _train_jr_variant(
    name: str,
    train_arrays: tuple[np.ndarray, ...],
    val_arrays: tuple[np.ndarray, ...],
    train_history: np.ndarray,
    val_history: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    val_cache: Mapping[str, np.ndarray],
    val_orders: Mapping[str, Sequence[int]],
    identity_state: Mapping[str, torch.Tensor],
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed()
    model = PretrainedHistoryAwareJointRevision(NUM_CLASSES).to(device)
    model.history_identity.load_state_dict(identity_state)
    if name == "PretrainedFrozenJR":
        for parameter in model.history_identity.parameters():
            parameter.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=JR_LR, weight_decay=WEIGHT_DECAY)
        identity_lr = 0.0
    else:
        identity_params = list(model.history_identity.parameters())
        jr_params = [
            p for key, p in model.named_parameters() if not key.startswith("history_identity.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": jr_params, "lr": JR_LR},
                {"params": identity_params, "lr": FINETUNE_IDENTITY_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        identity_lr = FINETUNE_IDENTITY_LR
    loader = DataLoader(
        _JRDataset(train_arrays, train_history),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    history: list[dict[str, float]] = []
    best_key = (-np.inf, -np.inf)
    best_epoch = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS_JR + 1):
        model.train()
        losses: list[float] = []
        jr_losses: list[float] = []
        identity_losses: list[float] = []
        for batch in loader:
            scores, posterior, identity_logits = model(
                batch["current"].float().to(device),
                batch["candidates"].float().to(device),
                batch["mask"].bool().to(device),
                batch["history"].float().to(device),
            )
            loss, jr_loss, identity_loss = _jr_loss(scores, posterior, identity_logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            jr_losses.append(float(jr_loss.cpu()))
            identity_losses.append(float(identity_loss.cpu()))
        actions, identity_predictions = _select_actions(
            model, val_arrays, val_rows, val_orders, device
        )
        terminal = _method_metrics(actions, val_rows, val_cache)
        val_identity = _classification_metrics(identity_predictions, val_arrays[5])
        stats = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "jr_loss": float(np.mean(jr_losses)),
            "identity_loss": float(np.mean(identity_losses)),
            "val_accuracy": float(terminal["terminal"]["accuracy"]),
            "val_macro_f1": float(terminal["terminal"]["macro_f1"]),
            "val_identity_accuracy": float(val_identity["accuracy"]),
            "val_identity_macro_f1": float(val_identity["macro_f1"]),
        }
        history.append(stats)
        print(
            f"{name} epoch {epoch}/{EPOCHS_JR} loss={stats['loss']:.6f} "
            f"val_acc={stats['val_accuracy']:.6f} val_id={stats['val_identity_accuracy']:.6f}",
            flush=True,
        )
        key = (stats["val_accuracy"], stats["val_macro_f1"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "epoch": epoch,
                    "lambda_id": LAMBDA_ID,
                    "identity_lr": identity_lr,
                    "jr_lr": JR_LR,
                    "val_terminal": terminal["terminal"],
                    "val_identity": val_identity,
                },
                checkpoint,
            )

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    actions, identity_predictions = _select_actions(model, val_arrays, val_rows, val_orders, device)
    method = _method_metrics(actions, val_rows, val_cache)
    identity = _classification_metrics(identity_predictions, val_arrays[5])
    return (
        method,
        {
            "epochs": EPOCHS_JR,
            "batch_size": BATCH_SIZE,
            "jr_learning_rate": JR_LR,
            "identity_learning_rate": identity_lr,
            "weight_decay": WEIGHT_DECAY,
            "lambda_identity": LAMBDA_ID,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_key[0],
            "best_val_macro_f1": best_key[1],
            "final_total_loss": history[-1]["loss"],
            "loss_history": history,
            "identity_head": identity,
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["methods"]
    normal = methods["NormalMultiPositiveJR"]["terminal"]
    frozen = methods["PretrainedFrozenJR"]["terminal"]
    finetune = methods["PretrainedFinetuneJR"]["terminal"]
    identity = result["identity_heads"]
    lines = [
        "# Pretrained History Identity → Multi-positive JR (Val)",
        "",
        "A 540→256→128 history identity encoder was trained on Train and selected by Val identity Macro-F1. Its weights initialise two independent JR variants; WM-E, ST-GCN, taxonomy and split remain frozen.",
        "",
        "| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "NormalMultiPositiveJR",
        "HistoryAwareMultiPositiveJR",
        "PretrainedFrozenJR",
        "PretrainedFinetuneJR",
        "PrivilegedJR",
        "GTLabelPrivilegedJR",
        "SafeOracle",
    ):
        item = methods[name]
        lines.append(
            f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | "
            f"{item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Standalone pretrained identity: Accuracy={identity['standalone_pretrained']['accuracy']:.6f}, Macro-F1={identity['standalone_pretrained']['macro_f1']:.6f}.",
            f"Feature-history standalone reference: Accuracy={identity['feature_history_reference']['accuracy']:.6f}, Macro-F1={identity['feature_history_reference']['macro_f1']:.6f}.",
            f"Pretrained Frozen JR identity head: Accuracy={identity['PretrainedFrozenJR']['accuracy']:.6f}, Macro-F1={identity['PretrainedFrozenJR']['macro_f1']:.6f}; Finetune: Accuracy={identity['PretrainedFinetuneJR']['accuracy']:.6f}, Macro-F1={identity['PretrainedFinetuneJR']['macro_f1']:.6f}.",
            f"Frozen minus Normal JR: ΔAccuracy={frozen['accuracy'] - normal['accuracy']:+.6f}, ΔMacro-F1={frozen['macro_f1'] - normal['macro_f1']:+.6f}.",
            f"Finetune minus Normal JR: ΔAccuracy={finetune['accuracy'] - normal['accuracy']:+.6f}, ΔMacro-F1={finetune['macro_f1'] - normal['macro_f1']:+.6f}.",
            f"Normal→Privileged Accuracy gap={methods['PrivilegedJR']['terminal']['accuracy'] - normal['accuracy']:+.6f}; Frozen→Privileged gap={methods['PrivilegedJR']['terminal']['accuracy'] - frozen['accuracy']:+.6f}; Finetune→Privileged gap={methods['PrivilegedJR']['terminal']['accuracy'] - finetune['accuracy']:+.6f}.",
            "",
            "Interpretation: compare the pretrained identity heads with the prior Feature-history MLP and compare each JR variant with Normal Multi-positive JR. Checkpoint selection used Val only; this experiment does not make a Test claim.",
            "",
            "Leakage audit: `split=val`, `test_used=false`, no Test path is accessed, formal WM-E/ST-GCN and old JR checkpoints are unchanged.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_evaluate(data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    root = data_root.resolve() / "datasets/policy_reduced14_kneel_eight_placement_v1"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    train_cache = _load_npz(root / "counterfactual_cache/train.npz")
    val_cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate_rows_cache(train_rows, train_cache, "train")
    _validate_rows_cache(val_rows, val_cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val rows must explicitly carry policy_split=val")
    train_orders = _orders(data_root, train_rows)
    val_orders = _orders(data_root, val_rows)
    train_base, train_stats = _examples(train_rows, train_cache, train_orders)
    val_base, _ = _examples(val_rows, val_cache, val_orders)
    train_history = _history_inputs(train_rows, train_cache)
    val_history = _history_inputs(val_rows, val_cache)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)

    checkpoint_dir = data_root.resolve() / "checkpoints" / CHECKPOINT_DIR_NAME
    identity_checkpoint = checkpoint_dir / "pretrained_history_identity_best.pth"
    identity_training, identity_state = _train_identity(
        train_history,
        train_labels,
        val_history,
        val_labels,
        device,
        identity_checkpoint,
    )
    standalone_identity = identity_training["best_val_identity"]
    train_arrays = train_base
    val_arrays = val_base
    frozen_method, frozen_training = _train_jr_variant(
        "PretrainedFrozenJR",
        train_arrays,
        (*val_arrays, val_history),
        train_history,
        val_history,
        train_rows,
        val_rows,
        val_cache,
        val_orders,
        identity_state,
        device,
        checkpoint_dir / "pretrained_frozen_jr_best.pth",
    )
    finetune_method, finetune_training = _train_jr_variant(
        "PretrainedFinetuneJR",
        train_arrays,
        (*val_arrays, val_history),
        train_history,
        val_history,
        train_rows,
        val_rows,
        val_cache,
        val_orders,
        identity_state,
        device,
        checkpoint_dir / "pretrained_finetune_jr_best.pth",
    )
    normal_method = _load_normal_jr(data_root.resolve(), val_rows, val_cache, val_orders, device)
    prior = _load_prior_methods(len(val_rows))
    prior_result = REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_identity/result.json"
    feature_history_reference = {"accuracy": 0.4843000877844554, "macro_f1": 0.5007947438748389}
    if prior_result.exists():
        prior_identity = json.loads(prior_result.read_text(encoding="utf-8"))
        feature_history_reference = prior_identity.get("classifier_metrics", {}).get(
            "Feature_history_MLP", feature_history_reference
        )
    methods = {
        "NormalMultiPositiveJR": normal_method,
        "HistoryAwareMultiPositiveJR": {
            **json.loads(
                (REPO_ROOT / "experiments/reduced14_eight_placement_v1/history_aware_jr_v1/result.json").read_text(encoding="utf-8")
            )["methods"]["HistoryAwareMultiPositiveJR"]
        },
        "PretrainedFrozenJR": frozen_method,
        "PretrainedFinetuneJR": finetune_method,
        **prior,
    }
    identity_heads = {
        "standalone_pretrained": standalone_identity,
        "feature_history_reference": feature_history_reference,
        "PretrainedFrozenJR": frozen_training["identity_head"],
        "PretrainedFinetuneJR": finetune_training["identity_head"],
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_PRETRAINED_HISTORY_JR",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {
            "train_contexts": len(train_rows),
            "val_moving_contexts": len(val_rows),
        },
        "methods": methods,
        "identity_heads": identity_heads,
        "pretraining": {
            **identity_training,
            "checkpoint": {
                "path": str(identity_checkpoint),
                "sha256": _sha256(identity_checkpoint),
            },
        },
        "training": {
            "train_stats": train_stats,
            "frozen": frozen_training,
            "finetune": finetune_training,
        },
        "comparisons": {
            "PretrainedFrozenJR_minus_NormalJR": {
                "accuracy": frozen_method["terminal"]["accuracy"] - normal_method["terminal"]["accuracy"],
                "macro_f1": frozen_method["terminal"]["macro_f1"] - normal_method["terminal"]["macro_f1"],
            },
            "PretrainedFinetuneJR_minus_NormalJR": {
                "accuracy": finetune_method["terminal"]["accuracy"] - normal_method["terminal"]["accuracy"],
                "macro_f1": finetune_method["terminal"]["macro_f1"] - normal_method["terminal"]["macro_f1"],
            },
            "NormalJR_to_PrivilegedJR": {
                "accuracy": prior["PrivilegedJR"]["terminal"]["accuracy"] - normal_method["terminal"]["accuracy"],
                "macro_f1": prior["PrivilegedJR"]["terminal"]["macro_f1"] - normal_method["terminal"]["macro_f1"],
            },
        },
        "protocol": {
            "history_input": "s0/s1 256-D ST-GCN features + s0/s1 14-D logp",
            "history_encoder": "Linear(540,256) -> GELU -> Linear(256,128) -> GELU",
            "identity_head": "Linear(128,14)",
            "jr_input": "current state + history latent + refined identity logits + legal candidates",
            "loss": "L_JR + 0.2 * CrossEntropy(refined_action_logits, label)",
            "checkpoint_selection": "identity Val Macro-F1; JR Val terminal Accuracy then Macro-F1",
            "candidate_budget": "ALL_LEGAL",
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_checkpoint_selection": True,
            "formal_wm_e_modified": False,
            "formal_stgcn_modified": False,
            "old_jr_checkpoint_modified": False,
            "true_future_recognition_as_model_input": False,
            "candidate_identity_changed": False,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pretrained history identity and JR on Train/Val only")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pretrained history JR requires CUDA; CPU fallback is disabled")
    train_and_evaluate(args.data_root.resolve(), device)


if __name__ == "__main__":
    main()
