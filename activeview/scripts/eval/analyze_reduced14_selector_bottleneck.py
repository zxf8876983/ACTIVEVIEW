#!/usr/bin/env python3
"""Decompose the reduced14 selector bottleneck on Val only.

The script compares the frozen WM-E candidate ranking with the frozen normal
Multi-Positive Joint Revision (JR), then fits a separate *diagnostic* JR on
Train with ``true_logp`` substituted for the candidate recognition input.  The
privileged model is never used by the formal pipeline and no Test data are
reachable from this entry point.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, load_pairwise_and_azimuths
from activeview.methods.joint_revision.model import JointRevision, select_actions


SEED = 42
NUM_CLASSES = 14
VIEW_COUNT = 32
EPOCHS = 20
BATCH_SIZE = 512


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _source_map(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {
        (str(row["scene_id"]), str(row["region"]), str(row["record_id"])):
        str(root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz")
        for row in rows
    }


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _source_map(data_root, rows)
    pairwise, azimuths = load_pairwise_and_azimuths(
        data_root,
        rows,
        source,
        pair_root=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/pairwise_viewpoint_geodesic",
    )
    output: dict[str, list[int]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]))
        output[str(row["episode_id"])] = candidate_order(
            row,
            int(row["s1_viewpoint_id"]),
            {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])},
            pairwise[key],
            azimuths[key],
        )
    return output


def _validate_rows_cache(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], split: str) -> None:
    if split == "test":
        raise ValueError("Test is locked for selector bottleneck diagnostics")
    row_ids = [str(row["episode_id"]) for row in rows]
    cache_ids = [str(value) for value in cache["episode_ids"].tolist()]
    if len(row_ids) != len(set(row_ids)) or len(cache_ids) != len(set(cache_ids)):
        raise ValueError(f"duplicate {split} episode IDs")
    if row_ids != cache_ids:
        raise ValueError(f"{split} feature/cache episode IDs are not exactly aligned")
    expected = (len(rows), VIEW_COUNT, NUM_CLASSES)
    for name in ("imagined_logp", "true_logp"):
        if cache[name].shape != expected:
            raise ValueError(f"unexpected {name} shape: {cache[name].shape}")


class _JRDataset(Dataset[dict[str, torch.Tensor]]):
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


def _build_examples(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
) -> tuple[tuple[np.ndarray, ...], dict[str, int]]:
    """Build the existing JR tensors, with the supplied candidate logp input."""
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    current_all: list[np.ndarray] = []
    candidates_all: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    fallbacks: list[int] = []
    positives: list[np.ndarray] = []
    labels: list[int] = []
    positive_counts: list[int] = []
    multi = single = none = 0
    max_actions = 1 + VIEW_COUNT - 2

    for row in rows:
        i = index[str(row["episode_id"])]
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        logp0 = np.asarray(cache["current_logp_s0"][i], dtype=np.float32)
        logp1 = np.asarray(cache["current_logp_s1"][i], dtype=np.float32)
        probs0, probs1 = np.exp(logp0), np.exp(logp1)
        current = np.asarray(
            [
                *probs0,
                *probs1,
                -np.sum(probs0 * logp0),
                -np.sum(probs1 * logp1),
                np.max(probs0),
                np.max(probs1),
                np.sort(probs1)[-1] - np.sort(probs1)[-2],
                np.sort(probs0)[-1] - np.sort(probs0)[-2],
            ],
            dtype=np.float32,
        )
        values = [np.concatenate([logp1, np.zeros(9, dtype=np.float32), [1.0]])]
        values.extend(
            np.concatenate([cache["imagined_logp"][i, candidate], cache["candidate_descriptor"][i, candidate], [0.0]])
            for candidate in candidates
        )
        action_correct = np.asarray(
            [int(np.argmax(logp1) == int(row["label_id"]))]
            + [int(np.argmax(cache["true_logp"][i, candidate]) == int(row["label_id"])) for candidate in candidates],
            dtype=np.float32,
        )
        positive_count = int(action_correct.sum())
        if positive_count:
            positive_counts.append(positive_count)
            multi += int(positive_count > 1)
            single += int(positive_count == 1)
            fallback = int(np.flatnonzero(action_correct)[0])
        else:
            none += 1
            label = int(row["label_id"])
            target_scores = [float(logp1[label])] + [float(cache["true_logp"][i, candidate, label]) for candidate in candidates]
            fallback = int(np.argmax(target_scores))
        padded = np.zeros((max_actions, NUM_CLASSES + 10), dtype=np.float32)
        padded[: len(values)] = np.asarray(values, dtype=np.float32)
        positive = np.zeros(max_actions, dtype=np.float32)
        positive[: len(values)] = action_correct
        mask = np.asarray([True] * len(values) + [False] * (max_actions - len(values)), dtype=bool)
        current_all.append(current)
        candidates_all.append(padded)
        masks.append(mask)
        fallbacks.append(fallback)
        positives.append(positive)
        labels.append(int(row["label_id"]))
    stats = {
        "contexts": len(labels),
        "positive_contexts": multi + single,
        "multi_positive_contexts": multi,
        "single_positive_contexts": single,
        "no_positive_contexts": none,
        "mean_positive_count": float(np.mean(positive_counts)) if positive_counts else 0.0,
    }
    arrays = (
        np.asarray(current_all),
        np.asarray(candidates_all),
        np.asarray(masks),
        np.asarray(fallbacks),
        np.asarray(positives),
        np.asarray(labels, dtype=np.int64),
    )
    return arrays, stats


def _train_privileged(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    device: torch.device,
) -> tuple[JointRevision, dict[str, Any]]:
    """Train an independent diagnostic JR with true_logp as candidate input."""
    _seed()
    privileged_cache = dict(cache)
    privileged_cache["imagined_logp"] = np.asarray(cache["true_logp"], dtype=np.float32)
    arrays, stats = _build_examples(rows, privileged_cache, orders)
    loader = DataLoader(_JRDataset(arrays), batch_size=BATCH_SIZE, shuffle=True)
    model = JointRevision(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
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
                posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none"
            ).reshape_as(scores)
            post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (main + 0.25 * bce + 0.05 * post).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        print(f"privileged JR epoch {epoch}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    model.eval()
    return model, {**stats, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "final_loss": history[-1], "loss_history": history}


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(truth * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0
        recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(pred == truth)) if truth.size else 0.0,
        "macro_f1": float(np.mean(f1)),
    }


def _terminal_predictions(
    actions: Sequence[int | None], cache: Mapping[str, np.ndarray], index: Mapping[str, int], rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[bool]]:
    predictions: list[int] = []
    positives: list[bool] = []
    for row, action in zip(rows, actions):
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        if action is None:
            prediction = int(np.argmax(cache["current_logp_s1"][i]))
        else:
            prediction = int(np.argmax(cache["true_logp"][i, int(action)]))
        predictions.append(prediction)
        positives.append(prediction == label)
    return predictions, positives


def _safe_oracle_actions(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], index: Mapping[str, int],
) -> list[int | None]:
    actions: list[int | None] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        current = float(cache["current_logp_s1"][i, label])
        candidates = list(orders[str(row["episode_id"])])
        utilities = [(candidate, float(cache["true_logp"][i, candidate, label] - current)) for candidate in candidates]
        if not utilities:
            actions.append(None)
            continue
        best = max(utilities, key=lambda item: (item[1], -candidates.index(item[0])))
        actions.append(None if best[1] <= 0.0 else int(best[0]))
    return actions


def _wm_top1_actions(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], index: Mapping[str, int],
) -> tuple[list[int | None], dict[str, Any]]:
    actions: list[int | None] = []
    hits: list[bool] = []
    top3_hits: list[bool] = []
    oracle_exists: list[bool] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        candidates = list(orders[str(row["episode_id"])])
        if not candidates:
            actions.append(None)
            hits.append(False)
            top3_hits.append(False)
            oracle_exists.append(False)
            continue
        real_positive = np.asarray([np.argmax(cache["true_logp"][i, candidate]) == label for candidate in candidates], dtype=bool)
        ranking = np.argsort(-cache["imagined_logp"][i, candidates, label], kind="stable")
        hits.append(bool(real_positive[ranking[0]]))
        top3_hits.append(bool(np.any(real_positive[ranking[: min(3, len(ranking))]])))
        oracle_exists.append(bool(real_positive.any()))
        actions.append(int(candidates[int(ranking[0])]))
    exists = np.asarray(oracle_exists, dtype=bool)
    hit_array = np.asarray(hits, dtype=bool)
    top3_array = np.asarray(top3_hits, dtype=bool)
    return actions, {
        "contexts": len(rows),
        "oracle_positive_exists_contexts": int(exists.sum()),
        "top1_positive_hit_rate": float(hit_array.mean()) if hit_array.size else 0.0,
        "top3_positive_hit_rate": float(top3_array.mean()) if top3_array.size else 0.0,
        "oracle_positive_top1_hit_rate": float(hit_array[exists].mean()) if exists.any() else None,
        "oracle_positive_top3_hit_rate": float(top3_array[exists].mean()) if exists.any() else None,
    }


def _action_diagnostics(
    actions: Sequence[int | None], rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], index: Mapping[str, int],
) -> dict[str, Any]:
    _, positives = _terminal_predictions(actions, cache, index, rows)
    counts = {"stay": sum(action is None for action in actions), "move": sum(action is not None for action in actions)}
    return {"positive_hit_rate": float(np.mean(positives)) if positives else 0.0, "positive_count": int(sum(positives)), "action_counts": counts}


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    data_root = data_root.resolve()
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    train_cache = _load_npz(root / "counterfactual_cache/train.npz")
    val_cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate_rows_cache(train_rows, train_cache, "train")
    _validate_rows_cache(val_rows, val_cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val rows must explicitly carry policy_split=val")
    val_orders = _orders(data_root, val_rows)
    train_orders = _orders(data_root, train_rows)
    val_index = {str(value): i for i, value in enumerate(val_cache["episode_ids"].tolist())}

    wm_actions, wm_metrics = _wm_top1_actions(val_rows, val_cache, val_orders, val_index)
    normal_checkpoint = data_root / "checkpoints/activeview_reduced14_eight_placement_v1/joint_revision_multi_positive.pth"
    payload = torch.load(normal_checkpoint, map_location=device, weights_only=False)
    if int(payload.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError(f"normal JR checkpoint is not reduced14: {normal_checkpoint}")
    normal_model = JointRevision(num_classes=NUM_CLASSES).to(device)
    normal_model.load_state_dict(payload.get("model_state_dict", payload["state_dict"]))
    normal_model.eval()
    normal_actions = select_actions(normal_model, val_cache, val_rows, val_orders, budget="ALL_LEGAL", device=device)

    privileged_model, training = _train_privileged(train_rows, train_cache, train_orders, device)
    privileged_val_cache = dict(val_cache)
    privileged_val_cache["imagined_logp"] = np.asarray(val_cache["true_logp"], dtype=np.float32)
    privileged_actions = select_actions(privileged_model, privileged_val_cache, val_rows, val_orders, budget="ALL_LEGAL", device=device)
    safe_actions = _safe_oracle_actions(val_rows, val_cache, val_orders, val_index)

    labels = [int(row["label_id"]) for row in val_rows]
    wm_terminal, wm_terminal_positive = _terminal_predictions(wm_actions, val_cache, val_index, val_rows)
    normal_terminal, normal_positive = _terminal_predictions(normal_actions, val_cache, val_index, val_rows)
    privileged_terminal, privileged_positive = _terminal_predictions(privileged_actions, val_cache, val_index, val_rows)
    safe_terminal, safe_positive = _terminal_predictions(safe_actions, val_cache, val_index, val_rows)
    normal_jr_diag = _action_diagnostics(normal_actions, val_rows, val_cache, val_index)
    privileged_diag = _action_diagnostics(privileged_actions, val_rows, val_cache, val_index)
    safe_diag = _action_diagnostics(safe_actions, val_rows, val_cache, val_index)

    # Recompute the per-context WM-E hit flags to keep the paired audit explicit.
    wm_flags: list[bool] = []
    for row in val_rows:
        i = val_index[str(row["episode_id"])]
        candidates = list(val_orders[str(row["episode_id"])])
        label = int(row["label_id"])
        if not candidates:
            wm_flags.append(False)
            continue
        ranking = np.argsort(-val_cache["imagined_logp"][i, candidates, label], kind="stable")
        wm_flags.append(bool(np.argmax(val_cache["true_logp"][i, candidates[int(ranking[0])]]) == label))
    wm_hit = np.asarray(wm_flags, dtype=bool)
    normal_hit = np.asarray(normal_positive, dtype=bool)
    correction = {
        "contexts": len(val_rows),
        "wm_e_top1_correct_jr_wrong": int(np.sum(wm_hit & ~normal_hit)),
        "wm_e_top1_correct_jr_wrong_rate": float(np.mean(wm_hit & ~normal_hit)),
        "wm_e_top1_wrong_jr_correct": int(np.sum(~wm_hit & normal_hit)),
        "wm_e_top1_wrong_jr_correct_rate": float(np.mean(~wm_hit & normal_hit)),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_SELECTOR_BOTTLENECK",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "training_performed": True,
        "population": {"moving_contexts": len(val_rows), "train_contexts_for_privileged_diagnostic": len(train_rows)},
        "wm_e_top1_positive": wm_metrics,
        "normal_jr": {
            **normal_jr_diag,
            "terminal": _metrics(normal_terminal, labels),
            "candidate_positive_hit_when_move": float(np.mean([p for a, p in zip(normal_actions, normal_positive) if a is not None])) if any(a is not None for a in normal_actions) else None,
        },
        "privileged_jr": {
            **privileged_diag,
            "terminal": _metrics(privileged_terminal, labels),
            "candidate_positive_hit_when_move": float(np.mean([p for a, p in zip(privileged_actions, privileged_positive) if a is not None])) if any(a is not None for a in privileged_actions) else None,
        },
        "safe_oracle": {**safe_diag, "terminal": _metrics(safe_terminal, labels)},
        "wm_e_top1_terminal": _metrics(wm_terminal, labels),
        "selector_correction_audit": correction,
        "training": training,
        "provenance": {
            "normal_jr_checkpoint": str(normal_checkpoint),
            "privileged_candidate_input": "true_logp substituted for imagined_logp; diagnostic only",
            "candidate_ordering": "frozen candidate_order / ALL_LEGAL",
            "terminal_recognition": "true archived candidate logp through frozen ST-GCN",
        },
        "leakage_flags": {
            "test_used": False,
            "val_used_for_training": False,
            "true_logp_used_as_privileged_diagnostic_input": True,
            "true_logp_used_for_normal_jr_input": False,
            "true_logp_used_for_candidate_identity": False,
            "future_observation_rendered": False,
        },
    }
    output_dir = Path("experiments/reduced14_eight_placement_v1/selector_bottleneck")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(output_dir / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    wm = result["wm_e_top1_positive"]
    normal = result["normal_jr"]
    privileged = result["privileged_jr"]
    safe = result["safe_oracle"]
    correction = result["selector_correction_audit"]
    text = "\n".join(
        [
            "# Reduced14 Selector Bottleneck (Val)",
            "",
            "This is a Val-only diagnostic. The formal WM-E/JR/ST-GCN artifacts were read-only; the privileged JR was trained on Train only and was not used by the formal method.",
            "",
            f"- Moving contexts: {result['population']['moving_contexts']}",
            f"- WM-E candidate Top-1 positive hit (all contexts): {wm['top1_positive_hit_rate']:.6f}",
            f"- WM-E candidate Top-3 positive hit (all contexts): {wm['top3_positive_hit_rate']:.6f}",
            f"- Oracle-positive contexts: {wm['oracle_positive_exists_contexts']}",
            f"- WM-E Top-1/Top-3 conditional on oracle-positive: {wm['oracle_positive_top1_hit_rate']:.6f} / {wm['oracle_positive_top3_hit_rate']:.6f}",
            "",
            "| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |",
            "|---|---:|---:|---:|---:|",
            f"| Normal JR | {normal['positive_hit_rate']:.6f} | {normal['action_counts']['stay'] / result['population']['moving_contexts']:.6f} | {normal['terminal']['accuracy']:.6f} | {normal['terminal']['macro_f1']:.6f} |",
            f"| Privileged JR (true recognition input) | {privileged['positive_hit_rate']:.6f} | {privileged['action_counts']['stay'] / result['population']['moving_contexts']:.6f} | {privileged['terminal']['accuracy']:.6f} | {privileged['terminal']['macro_f1']:.6f} |",
            f"| SafeOracle | {safe['positive_hit_rate']:.6f} | {safe['action_counts']['stay'] / result['population']['moving_contexts']:.6f} | {safe['terminal']['accuracy']:.6f} | {safe['terminal']['macro_f1']:.6f} |",
            "",
            f"WM-E Top-1 correct but normal JR wrong: {correction['wm_e_top1_correct_jr_wrong']} ({correction['wm_e_top1_correct_jr_wrong_rate']:.6f}).",
            f"WM-E Top-1 wrong but normal JR correct: {correction['wm_e_top1_wrong_jr_correct']} ({correction['wm_e_top1_wrong_jr_correct_rate']:.6f}).",
            "",
            "## Interpretation",
            "",
            "The WM-E ranking is the upstream candidate-recognition reference, while normal JR adds a learned Stay/candidate decision. A large privileged-JR improvement over normal JR with true candidate recognition supplied as a diagnostic input indicates selector/action-scoring loss; a low WM-E Top-1/Top-3 hit indicates an upstream WM-E ranking ceiling. These diagnostics do not alter the formal checkpoints or protocol.",
            "",
            "Leakage audit: `test_used=false`; Val is never used for privileged training; true recognition is used only in the explicitly privileged diagnostic and offline terminal/positive diagnostics.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 WM-E/Joint Revision bottleneck diagnostic")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; privileged JR diagnostic requires GPU")
    analyze(args.data_root, device)


if __name__ == "__main__":
    main()
