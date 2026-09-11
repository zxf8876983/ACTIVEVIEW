#!/usr/bin/env python3
"""Audit matched legal-action privileged oracle ceilings on reduced12 Val.

The action set is exactly ``stay/current + Stage-A legal candidates`` for the
Moving Val contexts.  Frozen recognizer evidence comes from the existing
counterfactual cache.  The already-trained adapted single-view head is
evaluated on the cached legal candidate features and, for its all-32 coverage
diagnostic, on the remaining archived skeleton views.  No selector is trained
and no Test artifact is opened.
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
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    NUM_CLASSES,
    classification,
    gt_margin,
)
from activeview.scripts.experiments.run_reduced12_single_view_classifier_adaptation import (
    FEATURE_DIM,
    LightweightClassifier,
)


SEED = 42
NUM_VIEWS = 32
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
ARCHIVE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")
STGCN_CHECKPOINT_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
ADAPTED_HEAD_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "single_view_classifier_adaptation/learned_head_best.pth"
)
TARGET_FEATURE_RELATIVE = Path(
    "diagnostics/reduced12_recognizer_clean_reference_audit/candidate_features.npz"
)
TARGET_FEATURE_META_RELATIVE = Path(
    "diagnostics/reduced12_future_recognition_evidence_prediction/"
    "val_moving_target_features.json"
)
TARGET_FEATURE_ARRAY_RELATIVE = Path(
    "diagnostics/reduced12_future_recognition_evidence_prediction/"
    "val_moving_target_features.npz"
)
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "matched_privileged_oracle_audit"
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.clip(np.exp(shifted).sum(axis=axis, keepdims=True), 1e-12, None))


def _load_inputs(data_root: Path) -> dict[str, Any]:
    policy_root = data_root / POLICY_RELATIVE
    stage_d = _read_jsonl(policy_root / "stage_d/features/val.jsonl")
    if not stage_d:
        raise ValueError("Stage-D Val is empty")
    moving_ids = {str(row["episode_id"]) for row in stage_d}

    stage_c_by_id: dict[str, dict[str, Any]] = {}
    with (policy_root / "stage_c/features/val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["episode_id"]) in moving_ids:
                    stage_c_by_id[str(row["episode_id"])] = row

    stage_a_by_id: dict[str, dict[str, Any]] = {}
    with (policy_root / "stage_a/episodes/val_episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["episode_id"]) in moving_ids:
                    stage_a_by_id[str(row["episode_id"])] = row

    cache_path = policy_root / "counterfactual_cache/val.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {key: np.asarray(archive[key]) for key in archive.files}
    cache_ids = [str(value) for value in cache["episode_ids"].tolist()]
    stage_d_ids = [str(row["episode_id"]) for row in stage_d]
    if cache_ids != stage_d_ids:
        raise ValueError("counterfactual cache and Stage-D Moving Val order differ")
    if len(stage_d) != 10080:
        raise ValueError(f"expected 10080 Moving Val contexts, found {len(stage_d)}")
    if len(stage_c_by_id) != len(stage_d) or len(stage_a_by_id) != len(stage_d):
        raise ValueError("Stage-A/Stage-C do not cover every Moving Val context")

    legal_ids: list[list[int]] = []
    current_ids: list[int] = []
    archive_paths: list[Path] = []
    alignment_errors: list[str] = []
    for index, row in enumerate(stage_d):
        episode_id = str(row["episode_id"])
        stage_c = stage_c_by_id[episode_id]
        stage_a = stage_a_by_id[episode_id]
        c_ids = [int(value) for value in stage_c["candidate_viewpoint_ids"]]
        a_pool = stage_a.get("candidate_pool", [])
        a_ids = [int(value["viewpoint_id"]) for value in a_pool]
        if c_ids != a_ids:
            alignment_errors.append(f"{episode_id}: Stage-A/Stage-C candidates")
        current = int(stage_a["current_view"]["viewpoint_id"])
        if current != int(stage_c["current_viewpoint_id"]):
            alignment_errors.append(f"{episode_id}: current viewpoint")
        if int(cache["label_id"][index]) != int(row["label_id"]):
            alignment_errors.append(f"{episode_id}: label")
        archive_path = Path(str(stage_a["current_view"]["skeleton_source_path"]))
        if not archive_path.is_file():
            raise FileNotFoundError(f"missing archived skeleton: {archive_path}")
        legal_ids.append(c_ids)
        current_ids.append(current)
        archive_paths.append(archive_path)
    if alignment_errors:
        raise ValueError(f"action-set alignment errors: {alignment_errors[:3]}")

    target_meta_path = data_root / TARGET_FEATURE_META_RELATIVE
    target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    target_array_path = data_root / TARGET_FEATURE_ARRAY_RELATIVE
    target = np.asarray(np.load(target_array_path, allow_pickle=False)["candidate_feature"])
    if target.shape != (len(stage_d), 21, FEATURE_DIM):
        raise ValueError(f"unexpected legal candidate feature cache shape {target.shape}")
    if target_meta.get("row_signature") != _signature(stage_c_by_id[eid] for eid in stage_d_ids):
        raise ValueError("candidate feature cache row order/signature is not matched")
    if target_meta.get("test_used", True):
        raise ValueError("candidate feature cache provenance indicates Test use")
    for index, ids in enumerate(legal_ids):
        values = target[index, : len(ids)]
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) < 1e-8):
            raise ValueError(f"invalid legal candidate feature at row {index}")

    return {
        "stage_d": stage_d,
        "stage_c_by_id": stage_c_by_id,
        "stage_a_by_id": stage_a_by_id,
        "cache": cache,
        "legal_ids": legal_ids,
        "current_ids": current_ids,
        "archive_paths": archive_paths,
        "candidate_features": target.astype(np.float32),
        "target_meta": target_meta,
        "policy_root": policy_root,
        "cache_path": cache_path,
    }


def _classification_with_method(
    labels: np.ndarray, predictions: np.ndarray, name: str, move_rate: float,
) -> dict[str, Any]:
    result = classification(labels, predictions)
    result.update({"method": name, "move_rate": float(move_rate), "stay_rate": float(1.0 - move_rate)})
    return result


def _original_candidate_feature_check(
    model: torch.nn.Module,
    candidate_features: np.ndarray,
    legal_ids: Sequence[Sequence[int]],
    true_logp: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Check that the legal feature cache and true-logp cache describe views."""
    features: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for index, ids in enumerate(legal_ids):
        features.extend(candidate_features[index, : len(ids)])
        references.extend(true_logp[index, ids])
    values = np.asarray(features, dtype=np.float32)
    expected = np.asarray(references, dtype=np.float32)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(torch.log_softmax(model.fc(batch), dim=-1).cpu().numpy())
    observed = np.concatenate(outputs, axis=0)
    return {
        "samples": int(len(values)),
        "max_logp_abs_error": float(np.max(np.abs(observed - expected))),
        "prediction_mismatches": int(np.sum(np.argmax(observed, axis=1) != np.argmax(expected, axis=1))),
    }


def _adapted_logits(
    data: Mapping[str, Any],
    stgcn: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return adapted logits for all 32 views and adapted s0 logits."""
    rows = data["stage_d"]
    n = len(rows)
    all_logits = np.full((n, NUM_VIEWS, NUM_CLASSES), np.nan, dtype=np.float32)
    legal_features = data["candidate_features"]
    legal_ids = data["legal_ids"]
    flat_features: list[np.ndarray] = []
    flat_locations: list[tuple[int, int]] = []
    for index, ids in enumerate(legal_ids):
        for slot, viewpoint_id in enumerate(ids):
            flat_features.append(legal_features[index, slot])
            flat_locations.append((index, int(viewpoint_id)))
    feature_array = np.asarray(flat_features, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(feature_array), batch_size):
            batch = torch.from_numpy(feature_array[start : start + batch_size]).to(device)
            logits = head(batch).cpu().numpy()
            for offset, (row_index, viewpoint_id) in enumerate(
                flat_locations[start : start + batch.shape[0]]
            ):
                all_logits[row_index, viewpoint_id] = logits[offset]

    requested = 0
    skeleton_batch: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []
    archive_checked = 0

    def flush() -> None:
        nonlocal skeleton_batch, locations
        if not skeleton_batch:
            return
        batch_array = np.asarray(skeleton_batch, dtype=np.float32)
        tensor = torch.from_numpy(batch_array).to(device, non_blocking=True)
        with torch.inference_mode():
            values = head(stgcn.forward_features(tensor)).cpu().numpy()
        for offset, (row_index, viewpoint_id) in enumerate(locations):
            all_logits[row_index, viewpoint_id] = values[offset]
        skeleton_batch = []
        locations = []

    for row_index, archive_path in enumerate(data["archive_paths"]):
        with np.load(archive_path, allow_pickle=False) as archive:
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
            navigable = np.asarray(archive["viewpoint_is_navigable"], dtype=bool)
        if viewpoint_ids.shape != (NUM_VIEWS,) or not np.array_equal(
            viewpoint_ids, np.arange(NUM_VIEWS, dtype=np.int64)
        ):
            raise ValueError(f"non-canonical viewpoint ids in {archive_path}")
        if skeletons.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeletons).all():
            raise ValueError(f"invalid skeleton archive shape/content: {archive_path}")
        if navigable.shape != (NUM_VIEWS,):
            raise ValueError(f"invalid navigation mask in {archive_path}")
        legal = set(legal_ids[row_index])
        # Stage-A's recorded candidate_pool is the formal dynamic legal set.
        # The archive navigation flags are retained as metadata but are not
        # substituted here: their historical serialization is not the Stage-A
        # legality definition used by the accepted oracle results.
        for viewpoint_id in range(NUM_VIEWS):
            if viewpoint_id in legal:
                continue
            requested += 1
            skeleton_batch.append(skeletons[viewpoint_id])
            locations.append((row_index, viewpoint_id))
            if len(skeleton_batch) >= batch_size:
                flush()
        archive_checked += 1
        if archive_checked % 500 == 0:
            print(f"[adapted-all32] archives={archive_checked}/{n}", flush=True)
    flush()
    if not np.isfinite(all_logits).all():
        raise RuntimeError("adapted all-32 inference left missing viewpoints")

    s0_features = np.asarray(
        [np.asarray(row["s0_feature"], dtype=np.float32)[:FEATURE_DIM] for row in rows],
        dtype=np.float32,
    )
    with torch.inference_mode():
        s0_logits = head(torch.from_numpy(s0_features).to(device)).cpu().numpy()
    return all_logits, s0_logits, {
        "legal_feature_samples": int(len(feature_array)),
        "nonlegal_view_skeletons_inferred": int(requested),
        "archives_checked": int(archive_checked),
    }


def _select_metrics(
    labels: np.ndarray,
    current_logp: np.ndarray,
    candidate_logp: np.ndarray,
    legal_ids: Sequence[Sequence[int]],
    candidate_logits: np.ndarray | None,
    current_logits: np.ndarray | None,
    recognizer_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    n = len(labels)
    names = ["Current/Stay", "GT-TrueLogP Oracle", "GTMargin Oracle"]
    if recognizer_name == "adapted":
        names.append("MaxConfidence")
    predictions = {name: np.empty(n, dtype=np.int64) for name in names}
    selected_views = {name: np.empty(n, dtype=np.int64) for name in names}
    selected_is_stay = {name: np.zeros(n, dtype=bool) for name in names}
    any_correct = np.zeros(n, dtype=bool)
    action_counts = np.asarray([1 + len(ids) for ids in legal_ids], dtype=np.int64)

    for index, label in enumerate(labels):
        ids = legal_ids[index]
        candidate_values = candidate_logp[index, ids]
        options = np.concatenate((current_logp[index][None, :], candidate_values), axis=0)
        option_ids = np.asarray([-1] + [int(value) for value in ids], dtype=np.int64)
        true_scores = options[:, int(label)]
        margins = np.asarray([gt_margin(item, int(label)) for item in options])
        probabilities = _softmax(options)
        max_conf = np.max(probabilities, axis=1)
        choices = {
            "Current/Stay": 0,
            "GT-TrueLogP Oracle": int(np.argmax(true_scores)),
            "GTMargin Oracle": int(np.argmax(margins)),
        }
        if recognizer_name == "adapted":
            choices["MaxConfidence"] = int(np.argmax(max_conf))
        for name, choice in choices.items():
            predictions[name][index] = int(np.argmax(options[choice]))
            selected_views[name][index] = int(option_ids[choice])
            selected_is_stay[name][index] = choice == 0
        any_correct[index] = bool(np.any(np.argmax(options, axis=1) == int(label)))

    metrics: dict[str, Any] = {}
    for name in names:
        metrics[name] = _classification_with_method(
            labels,
            predictions[name],
            name,
            float(np.mean(~selected_is_stay[name])),
        )
    details = {
        "selected_viewpoint_ids": selected_views,
        "selected_is_stay": selected_is_stay,
        "predictions": predictions,
        "any_correct": any_correct,
        "legal_action_count_mean": float(np.mean(action_counts)),
        "legal_action_count_min": int(np.min(action_counts)),
        "legal_action_count_max": int(np.max(action_counts)),
        "candidate_count_mean": float(np.mean(action_counts - 1)),
        "candidate_count_min": int(np.min(action_counts - 1)),
        "candidate_count_max": int(np.max(action_counts - 1)),
    }
    all32_coverage = None
    if candidate_logits is not None:
        all32_predictions = np.argmax(candidate_logits, axis=2)
        all32_correct = np.any(all32_predictions == labels[:, None], axis=1)
        all32_coverage = {
            "count": int(np.sum(all32_correct)),
            "rate": float(np.mean(all32_correct)),
            "contexts": int(n),
            "viewpoints_per_context": NUM_VIEWS,
        }
    return metrics, details, {"all32_any_correct": all32_coverage}


def _sample_rows(
    data: Mapping[str, Any],
    original: Mapping[str, Any],
    adapted: Mapping[str, Any],
    sample_count: int = 20,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(data["stage_d"]), size=sample_count, replace=False))
    labels = np.asarray([int(row["label_id"]) for row in data["stage_d"]], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index in indices:
        ids = data["legal_ids"][int(index)]
        cache_row = data["cache"]
        views: list[dict[str, Any]] = []
        current_id = int(data["current_ids"][int(index)])
        for viewpoint_id in ids:
            frozen_logp = np.asarray(cache_row["true_logp"][index, viewpoint_id])
            adapted_logit = np.asarray(adapted["candidate_logits"][index, viewpoint_id])
            views.append({
                "viewpoint_id": int(viewpoint_id),
                "frozen_predicted_class": int(np.argmax(frozen_logp)),
                "frozen_true_class_logp": float(frozen_logp[labels[index]]),
                "frozen_gt_margin": float(gt_margin(frozen_logp, int(labels[index]))),
                "frozen_max_confidence": float(np.max(_softmax(frozen_logp))),
                "adapted_predicted_class": int(np.argmax(adapted_logit)),
                "adapted_true_class_logp": float(
                    torch.log_softmax(torch.from_numpy(adapted_logit), dim=0).numpy()[labels[index]]
                ),
                "adapted_max_confidence": float(np.max(_softmax(adapted_logit))),
            })
        rows.append({
            "episode_id": str(data["stage_d"][index]["episode_id"]),
            "record_id": str(data["stage_d"][index]["record_id"]),
            "scene_id": str(data["stage_d"][index]["scene_id"]),
            "placement_id": str(data["stage_d"][index]["region"]),
            "label_id": int(labels[index]),
            "current_viewpoint_id": current_id,
            "legal_viewpoint_ids": [int(value) for value in ids],
            "current_frozen_predicted_class": int(np.argmax(cache_row["current_logp_s0"][index])),
            "current_adapted_predicted_class": int(np.argmax(adapted["current_logits"][index])),
            "views": views,
            "frozen_selected_viewpoints": {
                name: int(values[index]) for name, values in original["selected_viewpoint_ids"].items()
            },
            "adapted_selected_viewpoints": {
                name: int(values[index]) for name, values in adapted["selected_viewpoint_ids"].items()
            },
            "frozen_any_correct_viewpoints": [
                int(current_id)
            ] if int(np.argmax(cache_row["current_logp_s0"][index])) == int(labels[index]) else [],
        })
        rows[-1]["frozen_any_correct_viewpoints"] += [
            int(value) for value in ids
            if int(np.argmax(cache_row["true_logp"][index, value])) == int(labels[index])
        ]
        rows[-1]["adapted_any_correct_viewpoints"] = [
            int(current_id)
        ] if int(np.argmax(adapted["current_logits"][index])) == int(labels[index]) else []
        rows[-1]["adapted_any_correct_viewpoints"] += [
            int(value) for value in ids
            if int(np.argmax(adapted["candidate_logits"][index, value])) == int(labels[index])
        ]
    return rows


def _analysis(result: Mapping[str, Any]) -> str:
    frozen = result["recognizers"]["Frozen original ST-GCN"]
    adapted = result["recognizers"]["Adapted single-view head"]
    legal = result["legal_action_set"]
    old_ref = result["historical_references"]["frozen_legal_oracle_accuracy"]
    discrepancy = frozen["GT-TrueLogP Oracle"]["accuracy"] - old_ref
    lines = [
        "# Matched Privileged Oracle Audit",
        "",
        "Moving Val only (10,080 contexts). The formal action set is each context's separate stay/current action plus its Stage-A legal reachable candidate pool; all-32 is reported only as an adapted-head coverage diagnostic.",
        "",
        "## Action-set alignment",
        "",
        f"Candidate actions (excluding stay): mean/min/max={legal['candidate_count_mean']:.3f}/{legal['candidate_count_min']}/{legal['candidate_count_max']}; legal actions including stay: mean/min/max={legal['legal_action_count_mean']:.3f}/{legal['legal_action_count_min']}/{legal['legal_action_count_max']}. Stage-A/Stage-C/cache alignment errors: {result['alignment']['action_set_errors']}.",
        "",
        "## Overall metrics",
        "",
        "| Recognizer / method | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ]
    for recognizer_name, methods in result["recognizers"].items():
        for method_name, metric in methods.items():
            if method_name in {"All-32 AnyCorrect Coverage", "AnyCorrect Coverage"}:
                continue
            lines.append(f"| {recognizer_name} / {method_name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric['move_rate']:.6f} |")
    lines.extend([
        "",
        f"Frozen legal AnyCorrect Coverage: {frozen['AnyCorrect Coverage']['coverage_rate']:.6f} ({frozen['AnyCorrect Coverage']['coverage_count']}/{result['population']['moving_val_contexts']}). Adapted legal AnyCorrect Coverage: {adapted['AnyCorrect Coverage']['coverage_rate']:.6f} ({adapted['AnyCorrect Coverage']['coverage_count']}/{result['population']['moving_val_contexts']}). Adapted unrestricted all-32 AnyCorrect Coverage: {adapted['All-32 AnyCorrect Coverage']['rate']:.6f} ({adapted['All-32 AnyCorrect Coverage']['count']}/{result['population']['moving_val_contexts']}).",
        "",
        "## Formal oracle interpretation",
        "",
        f"The frozen GT-TrueLogP legal oracle Accuracy is {frozen['GT-TrueLogP Oracle']['accuracy']:.6f}; compared with the historical ~72.8% reference, the discrepancy is {discrepancy * 100:+.3f}pp. Frozen GTMargin is {frozen['GTMargin Oracle']['accuracy']:.6f}. Because the action set and final prediction rule are matched, this is the formal privileged H0 oracle for the frozen recognizer. AnyCorrect is coverage only, not Oracle Accuracy.",
        "",
        "The adapted-head legal oracle uses the same action set and actual selected-view argmax; its MaxConfidence row is a no-GT selector. The unrestricted all-32 adapted number is not a formal policy oracle and should not be mixed with the legal-action result.",
        "",
        "## Historical/protocol audit",
        "",
        "The 95.754% adapted all-32 coverage (if reproduced) is inflated relative to the formal legal action space because it permits every lattice viewpoint rather than Stage-A reachable candidates. Legal candidate count and stay/current alignment are therefore reported explicitly.",
        "",
        "## Leakage and boundaries",
        "",
        "- `policy_test_used=false`; only Stage-A/Stage-C/Stage-D Val and Val counterfactual/archive artifacts were opened.",
        "- No model was trained, no selector was trained, and no RGB/skeleton/perception data was generated.",
        "- GT label and true candidate logp are used only for privileged oracle selection/coverage; final predictions always come from the selected recognizer output.",
        "- Future archived skeletons are used for terminal evidence and adapted all-32 diagnostic inference only; they are not deployable policy inputs.",
        "",
    ])
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device, inference_batch_size: int) -> dict[str, Any]:
    started = time.time()
    _seed()
    data = _load_inputs(data_root)
    cache = data["cache"]
    labels = np.asarray([int(row["label_id"]) for row in data["stage_d"]], dtype=np.int64)
    stgcn_checkpoint = data_root / STGCN_CHECKPOINT_RELATIVE
    adapted_checkpoint = data_root / ADAPTED_HEAD_RELATIVE
    if not stgcn_checkpoint.is_file() or not adapted_checkpoint.is_file():
        raise FileNotFoundError("required frozen recognizer checkpoint is missing")
    stgcn, _ = load_checkpoint(stgcn_checkpoint, NUM_CLASSES, str(device))
    payload = torch.load(adapted_checkpoint, map_location=device, weights_only=False)
    head = LightweightClassifier(FEATURE_DIM).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    current_logp = np.asarray(cache["current_logp_s0"], dtype=np.float32)
    candidate_logp = np.asarray(cache["true_logp"], dtype=np.float32)
    cache_check = _original_candidate_feature_check(
        stgcn, data["candidate_features"], data["legal_ids"], candidate_logp,
        device, inference_batch_size,
    )
    adapted_candidate_logits, adapted_current_logits, inference_summary = _adapted_logits(
        data, stgcn, head, device, inference_batch_size,
    )
    adapted_candidate_logp = _log_softmax(adapted_candidate_logits, axis=2)
    adapted_current_logp = _log_softmax(adapted_current_logits, axis=1)
    original_metrics, original_details, original_cov = _select_metrics(
        labels, current_logp, candidate_logp, data["legal_ids"], None, None, "frozen"
    )
    adapted_metrics, adapted_details, adapted_cov = _select_metrics(
        labels, adapted_current_logp, adapted_candidate_logp, data["legal_ids"],
        adapted_candidate_logits, adapted_current_logits, "adapted",
    )
    for metrics, details, coverage in ((original_metrics, original_details, original_cov), (adapted_metrics, adapted_details, adapted_cov)):
        any_correct = details.pop("any_correct")
        metrics["AnyCorrect Coverage"] = {
            "method": "AnyCorrect Coverage",
            "accuracy": None,
            "macro_f1": None,
            "coverage_count": int(np.sum(any_correct)),
            "coverage_rate": float(np.mean(any_correct)),
            "contexts": int(len(labels)),
            "move_rate": None,
            "stay_rate": None,
        }
        if coverage.get("all32_any_correct") is not None:
            metrics["All-32 AnyCorrect Coverage"] = coverage["all32_any_correct"]
        details["any_correct"] = any_correct

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_MATCHED_PRIVILEGED_ORACLE_AUDIT",
        "status": "COMPLETED",
        "population": {"split": "Moving Val", "moving_val_contexts": int(len(labels))},
        "labels": list(LABELS),
        "legal_action_set": {
            "definition": "stay/current + Stage-A legal reachable candidates; invalid/nonfinite archives rejected",
            "candidate_count_mean": original_details["candidate_count_mean"],
            "candidate_count_min": original_details["candidate_count_min"],
            "candidate_count_max": original_details["candidate_count_max"],
            "legal_action_count_mean": original_details["legal_action_count_mean"],
            "legal_action_count_min": original_details["legal_action_count_min"],
            "legal_action_count_max": original_details["legal_action_count_max"],
        },
        "recognizers": {
            "Frozen original ST-GCN": original_metrics,
            "Adapted single-view head": adapted_metrics,
        },
        "alignment": {
            "stage_a_stage_c_cache_order": True,
            "action_set_errors": 0,
            "cache_episode_order_matches_stage_d": True,
            "candidate_feature_cache": cache_check,
            "adapted_inference": inference_summary,
        },
        "historical_references": {
            "frozen_legal_oracle_accuracy": 0.7282738095238095,
            "old_frozen_anycorrect_coverage": 0.7282738095238095,
            "adapted_all32_anycorrect_reference": 0.95754,
        },
        "gaps": {
            "frozen_gt_true_logp_minus_current_accuracy": float(original_metrics["GT-TrueLogP Oracle"]["accuracy"] - original_metrics["Current/Stay"]["accuracy"]),
            "adapted_gt_true_logp_minus_current_accuracy": float(adapted_metrics["GT-TrueLogP Oracle"]["accuracy"] - adapted_metrics["Current/Stay"]["accuracy"]),
            "adapted_all32_minus_adapted_legal_coverage": float(adapted_metrics["All-32 AnyCorrect Coverage"]["rate"] - adapted_metrics["AnyCorrect Coverage"]["coverage_rate"]),
        },
        "leakage_audit": {
            "policy_test_used": False,
            "training_used": False,
            "selector_training_used": False,
            "new_rgb_or_skeleton_generated": False,
            "gt_label_used_for_privileged_selection_only": True,
            "true_candidate_logp_used_for_privileged_selection_only": True,
            "future_skeleton_used_for_terminal_or_all32_adapted_diagnostic": True,
            "deployable": False,
        },
        "artifacts": {
            "stage_a_val": str((data["policy_root"] / "stage_a/episodes/val_episodes.jsonl").resolve()),
            "stage_c_val": str((data["policy_root"] / "stage_c/features/val.jsonl").resolve()),
            "stage_d_val": str((data["policy_root"] / "stage_d/features/val.jsonl").resolve()),
            "counterfactual_val": str(data["cache_path"].resolve()),
            "stgcn_checkpoint": str(stgcn_checkpoint.resolve()),
            "stgcn_checkpoint_sha256": _sha256(stgcn_checkpoint),
            "adapted_head_checkpoint": str(adapted_checkpoint.resolve()),
            "adapted_head_checkpoint_sha256": _sha256(adapted_checkpoint),
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds": float(time.time() - started),
        },
    }
    sample_data = {
        "selected_viewpoint_ids": original_details["selected_viewpoint_ids"],
    }
    adapted_sample_data = {
        "selected_viewpoint_ids": adapted_details["selected_viewpoint_ids"],
        "candidate_logits": adapted_candidate_logits,
        "current_logits": adapted_current_logits,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_samples.json").write_text(
        json.dumps(_sample_rows(data, sample_data, adapted_sample_data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(json.dumps({
        "seed": SEED,
        "device": str(device),
        "inference_batch_size": inference_batch_size,
        "moving_val_contexts": len(labels),
        "policy_test_used": False,
        "training_used": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=256)
    args = parser.parse_args()
    result = run(
        args.output_dir.resolve(), args.data_root.resolve(),
        _require_cuda(args.device), args.inference_batch_size,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["population"]["moving_val_contexts"],
        "frozen": result["recognizers"]["Frozen original ST-GCN"],
        "adapted": result["recognizers"]["Adapted single-view head"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
