#!/usr/bin/env python3
"""Val-only H1 action-identity disambiguation diagnostics.

The frozen Stage-C-v0 H1 candidate is compared with random and
identity-oriented candidate choices.  Candidate beliefs come from the already
trained reduced14 history-identity encoder applied to real archived skeletons;
no formal checkpoint or policy is changed and this entry point has no Test
path.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import (
    candidate_order,
    context_key,
    load_pairwise_and_azimuths,
)
from activeview.methods.joint_revision.history_aware import HistoryIdentityEncoder
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _validate_rows_cache,
)


SEED = 42
NUM_CLASSES = 14
STGCN_FEATURE_DIM = 256
HISTORY_INPUT_DIM = 2 * STGCN_FEATURE_DIM + 2 * NUM_CLASSES
INFERENCE_BATCH_SIZE = 512
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/h1_disambiguation"
DATASET_NAME = "policy_reduced14_kneel_eight_placement_v1"
CHECKPOINT_DIR = "activeview_reduced14_eight_placement_v1"


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


def _source_map(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {
        context_key(row): str(root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz")
        for row in rows
    }


def _h1_orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    sources = _source_map(data_root, rows)
    pairwise, azimuths = load_pairwise_and_azimuths(
        data_root,
        rows,
        sources,
        pair_root=data_root / "datasets" / DATASET_NAME / "pairwise_viewpoint_geodesic",
    )
    orders: dict[str, list[int]] = {}
    for row in rows:
        scene_region = (str(row["scene_id"]), str(row["region"]))
        current = int(row["s0_viewpoint_id"])
        orders[str(row["episode_id"])] = candidate_order(
            row,
            current,
            {current},
            pairwise[scene_region],
            azimuths[scene_region],
        )
    return orders


def _label_names(data_root: Path) -> list[str]:
    mapping_path = data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-train/label_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if set(mapping.values()) != set(range(NUM_CLASSES)):
        raise ValueError(f"non-contiguous reduced14 label mapping: {mapping_path}")
    return [str(name) for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _classification_metrics(predictions: Sequence[int], labels: Sequence[int], names: Sequence[str]) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    if predicted.shape != truth.shape:
        raise ValueError("prediction/label shape mismatch")
    matrix = np.bincount(
        truth * NUM_CLASSES + predicted,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    f1_values: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, name in enumerate(names):
        tp = float(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted_count = int(matrix[:, class_id].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"class_id": class_id, "support": support, "accuracy": recall, "f1": f1}
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(predicted == truth)) if truth.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _load_identity(data_root: Path, device: torch.device) -> tuple[HistoryIdentityEncoder, Path]:
    checkpoint = data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = HistoryIdentityEncoder(NUM_CLASSES).to(device)
    state = payload.get("model_state_dict", payload["state_dict"])
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _candidate_beliefs(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    device: torch.device,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Infer identity beliefs for every real legal H1 candidate."""
    stgcn_checkpoint = data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth"
    stgcn, _ = load_checkpoint(stgcn_checkpoint, NUM_CLASSES, str(device))
    identity, _ = _load_identity(data_root, device)
    sources = _source_map(data_root, rows)
    beliefs: list[np.ndarray | None] = [None] * len(rows)
    skeleton_buffer: list[np.ndarray] = []
    row_buffer: list[int] = []
    cached_logp_buffer: list[np.ndarray] = []
    max_logp_error = 0.0
    logp_error_sum = 0.0
    logp_error_count = 0
    candidate_count = 0

    def flush() -> None:
        nonlocal max_logp_error, logp_error_sum, logp_error_count
        if not skeleton_buffer:
            return
        skeletons = torch.from_numpy(np.asarray(skeleton_buffer, dtype=np.float32)).to(device)
        with torch.inference_mode():
            features = stgcn.forward_features(skeletons)
            computed_logp = torch.log_softmax(stgcn.fc(features), dim=-1).cpu().numpy().astype(np.float32)
            s0_features = np.stack([
                np.asarray(rows[index]["s0_feature"], dtype=np.float32)[:STGCN_FEATURE_DIM]
                for index in row_buffer
            ])
            s0_logp = np.asarray(cache["current_logp_s0"], dtype=np.float32)[row_buffer]
            candidate_logp = np.asarray(cached_logp_buffer, dtype=np.float32)
            error = np.abs(computed_logp - candidate_logp)
            max_logp_error = max(max_logp_error, float(error.max(initial=0.0)))
            logp_error_sum += float(error.sum())
            logp_error_count += int(error.size)
            history_input = np.concatenate([
                s0_features,
                features.cpu().numpy().astype(np.float32),
                s0_logp,
                candidate_logp,
            ], axis=1)
            candidate_belief = torch.softmax(
                identity(torch.from_numpy(history_input).to(device))[1], dim=-1,
            ).cpu().numpy().astype(np.float32)
        offset = 0
        for index in dict.fromkeys(row_buffer):
            count = len(orders[str(rows[index]["episode_id"])])
            beliefs[index] = candidate_belief[offset : offset + count]
            offset += count
        skeleton_buffer.clear()
        row_buffer.clear()
        cached_logp_buffer.clear()

    for row_index, row in enumerate(rows):
        source = Path(sources[context_key(row)])
        with np.load(source, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
        by_id = {int(value): position for position, value in enumerate(ids.tolist())}
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        if not candidates:
            beliefs[row_index] = np.empty((0, NUM_CLASSES), dtype=np.float32)
            continue
        # Keep a context intact so one row is assigned exactly once when a
        # buffered inference batch is flushed.
        if skeleton_buffer and len(skeleton_buffer) + len(candidates) > INFERENCE_BATCH_SIZE:
            flush()
        for candidate in candidates:
            skeleton_buffer.append(skeletons[by_id[candidate]])
            row_buffer.append(row_index)
            cached_logp_buffer.append(np.asarray(cache["true_logp"][row_index, candidate], dtype=np.float32))
            candidate_count += 1
    flush()
    if any(value is None for value in beliefs):
        raise RuntimeError("missing candidate belief for at least one Val context")
    return [value for value in beliefs if value is not None], {
        "candidate_hypothesis_samples": candidate_count,
        "mean_candidates_per_context": float(candidate_count / len(rows)) if rows else 0.0,
        "max_candidate_logp_abs_error": max_logp_error,
        "mean_candidate_logp_abs_error": float(logp_error_sum / logp_error_count) if logp_error_count else 0.0,
    }


def _selector_metrics(
    name: str,
    selected_indices: Sequence[int],
    beliefs: Sequence[np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    s0_predictions: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    selected_beliefs = np.stack([beliefs[index][choice] for index, choice in enumerate(selected_indices)])
    predictions = selected_beliefs.argmax(axis=1).astype(np.int64)
    entropy = -np.sum(selected_beliefs * np.log(np.clip(selected_beliefs, 1e-12, None)), axis=1)
    s0_wrong = s0_predictions != labels
    correction = float(np.mean(predictions[s0_wrong] == labels[s0_wrong])) if np.any(s0_wrong) else 0.0
    metrics = _classification_metrics(predictions, labels, label_names)
    return {
        "selector": name,
        "selected_count": int(len(selected_indices)),
        "stay_rate": 0.0,
        "mean_entropy": float(np.mean(entropy)),
        "s0_error_count": int(np.sum(s0_wrong)),
        "s0_error_correction_rate": correction,
        "identity": metrics,
    }


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    selectors = result["selectors"]
    comparison = result["comparison"]
    lines = [
        "# Reduced14 H1 Disambiguation Potential (Val)",
        "",
        "Each Val moving context enumerates legal real archived viewpoints from s0. The frozen history-identity encoder scores the resulting [s0, candidate] history; only the candidate selection rule changes.",
        "",
        f"Val moving contexts: {result['population']['val_moving_contexts']}; candidate hypotheses: {result['population']['candidate_hypothesis_samples']}. Test was not read.",
        "",
        "## History identity after H1 selection",
        "",
        "| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("Frozen_current_H1", "Random_H1", "Min_entropy_H1", "Max_margin_H1", "IdentityOracle_H1"):
        item = selectors[name]
        lines.append(
            f"| {name} | {item['identity']['accuracy']:.6f} | {item['identity']['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |"
        )
    lines.extend([
        "",
        f"IdentityOracle minus Frozen H1 identity Accuracy: {comparison['identity_oracle_minus_frozen_accuracy']:+.6f}.",
        f"Contexts with at least one candidate predicted correctly by the history classifier: {comparison['contexts_with_correct_candidate']} / {result['population']['val_moving_contexts']} ({comparison['contexts_with_correct_candidate_rate']:.6f}).",
        f"Frozen H1 selected candidate equals the Stage-C-v0 recorded s1 viewpoint for {comparison['frozen_h1_matches_recorded_s1']} / {result['population']['val_moving_contexts']} contexts.",
        "",
        "## Interpretation",
        "",
        "The IdentityOracle row is a privileged upper bound because it uses the ground-truth label to choose the candidate. A large gap from Frozen H1 indicates recoverable disambiguation potential in the observed candidate views; a small gap indicates that candidate choice has limited leverage under this frozen identity representation.",
        "",
        "Leakage audit: `test_used=false`; only Val rows, archived Val skeletons, the frozen reduced14 ST-GCN and the already-trained history-identity checkpoint were accessed. No formal checkpoint was modified.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    started = time.perf_counter()
    data_root = data_root.resolve()
    policy_root = data_root / "datasets" / DATASET_NAME
    rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    _validate_rows_cache(rows, cache, "val")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in rows):
        raise ValueError("H1 disambiguation requires explicit Val rows")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    names = _label_names(data_root)
    orders = _h1_orders(data_root, rows)
    beliefs, inference_stats = _candidate_beliefs(data_root, rows, cache, orders, device)
    rng = np.random.default_rng(SEED)
    frozen_indices: list[int] = []
    random_indices: list[int] = []
    min_entropy_indices: list[int] = []
    max_margin_indices: list[int] = []
    oracle_indices: list[int] = []
    contexts_with_correct = 0
    for row, belief in zip(rows, beliefs):
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        if not candidates:
            raise ValueError(f"no legal H1 candidate for {row['episode_id']}")
        recorded_s1 = int(row["s1_viewpoint_id"])
        frozen_indices.append(candidates.index(recorded_s1))
        random_indices.append(int(rng.integers(len(candidates))))
        entropy = -np.sum(belief * np.log(np.clip(belief, 1e-12, None)), axis=1)
        margins = np.sort(belief, axis=1)[:, -1] - np.sort(belief, axis=1)[:, -2]
        min_entropy_indices.append(int(np.argmin(entropy)))
        max_margin_indices.append(int(np.argmax(margins)))
        label = int(row["label_id"])
        oracle_indices.append(int(np.argmax(belief[:, label])))
        contexts_with_correct += int(np.any(np.argmax(belief, axis=1) == label))
    s0_predictions = np.asarray(cache["current_logp_s0"], dtype=np.float32).argmax(axis=1)
    index_sets = {
        "Frozen_current_H1": frozen_indices,
        "Random_H1": random_indices,
        "Min_entropy_H1": min_entropy_indices,
        "Max_margin_H1": max_margin_indices,
        "IdentityOracle_H1": oracle_indices,
    }
    selectors = {
        name: _selector_metrics(name, indices, beliefs, rows, s0_predictions, names)
        for name, indices in index_sets.items()
    }
    frozen_matches = sum(
        int(int(row["s1_viewpoint_id"]) == int(orders[str(row["episode_id"])][choice]))
        for row, choice in zip(rows, frozen_indices)
    )
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_H1_DISAMBIGUATION",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "population": {
            "val_moving_contexts": len(rows),
            **inference_stats,
        },
        "selectors": selectors,
        "comparison": {
            "identity_oracle_minus_frozen_accuracy": selectors["IdentityOracle_H1"]["identity"]["accuracy"] - selectors["Frozen_current_H1"]["identity"]["accuracy"],
            "contexts_with_correct_candidate": contexts_with_correct,
            "contexts_with_correct_candidate_rate": float(contexts_with_correct / len(rows)) if rows else 0.0,
            "frozen_h1_matches_recorded_s1": frozen_matches,
            "s0_accuracy": float(np.mean(s0_predictions == labels)),
            "s0_error_count": int(np.sum(s0_predictions != labels)),
        },
        "artifacts": {
            "identity_checkpoint": str((data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth").resolve()),
            "identity_checkpoint_sha256": _sha256(data_root / "checkpoints" / CHECKPOINT_DIR / "pretrained_history_identity_best.pth"),
            "stgcn_checkpoint": str((data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth").resolve()),
            "val_features": str((policy_root / "stage_d/features/val.jsonl").resolve()),
            "val_cache": str((policy_root / "counterfactual_cache/val.npz").resolve()),
        },
        "protocol": {
            "selector_candidates": "all legal candidate_order viewpoints from s0, excluding s0",
            "frozen_h1": "recorded Stage-C-v0 s1_viewpoint_id",
            "random_seed": SEED,
            "identity_oracle": "argmax candidate belief at ground-truth label",
            "history_input": "[feature_s0, feature_candidate, logp_s0, logp_candidate] = 540-D",
            "candidate_features": "frozen ST-GCN forward_features on archived skeleton",
        },
        "leakage_flags": {
            "test_used": False,
            "formal_checkpoint_modified": False,
            "future_observation_rendered": False,
            "ground_truth_label_used_for_selection": "IdentityOracle_H1 only",
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 H1 disambiguation diagnostics")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H1 disambiguation requires CUDA; CPU fallback is disabled")
    analyze(args.data_root, device)


if __name__ == "__main__":
    main()
