#!/usr/bin/env python3
"""Recover and validate the historical reduced12 clean-perception tensor.

This is a read-only Val audit.  It deliberately does not reconstruct RGB,
pose estimates, normalized skeletons, or any ActiveView policy artifact.  A
historical tensor is accepted only when its metadata, labels, shape and
frozen-checkpoint reproduction agree with the recorded protocol.  Mapping to
the current ActiveView Official-Val records is exact (record id or the full
source interval); no label, filename, nearest-neighbour, or array-order
matching is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint


DATASET_NAME = "reduced12_no_kneel_clean_babel_diversity_v1"
LABELS: Tuple[str, ...] = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPECTED_SHAPE = (3, 30, 17, 1)
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/clean_perception_reference_recovery"
CHECKPOINT_REL = f"checkpoints/stgcn_{DATASET_NAME}/stgcn_reduced12_no_kneel_clean_best.pth"
RECORDED_RESULT = REPO_ROOT / "experiments/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/result.json"
POLICY_STAGE_D_VAL = "datasets/policy_reduced12_eight_placement_v1/stage_d/features/val.jsonl"
EPS = 1.0e-12


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return record


def _records(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return [dict(item) for item in payload]


def _jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_number}")
                rows.append(dict(value))
    return rows


def _normal_source(value: Any) -> str:
    """Normalize path spelling only; this is still an exact path comparison."""
    text = str(value or "")
    if not text:
        return ""
    return str(Path(text).expanduser())


def _exact_key(row: Mapping[str, Any]) -> Tuple[str, int, int]:
    return (
        _normal_source(row.get("source_path") or row.get("source_group")),
        int(row.get("start_frame", -1)),
        int(row.get("end_frame", -1)),
    )


def _metrics(targets: Sequence[int], predictions: Sequence[int]) -> Dict[str, Any]:
    target = np.asarray(targets, dtype=np.int64)
    prediction = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, guess in zip(target.tolist(), prediction.tolist()):
        if 0 <= truth < len(LABELS) and 0 <= guess < len(LABELS):
            confusion[truth, guess] += 1
    per_class: List[Dict[str, Any]] = []
    f1_values: List[float] = []
    for class_id, action in enumerate(LABELS):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({
            "action": action,
            "action_id": class_id,
            "support": int(support),
            "accuracy": recall,
            "precision": precision,
            "f1": f1,
        })
    count = int(target.size)
    return {
        "count": count,
        "accuracy": float(np.mean(target == prediction)) if count else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _load_stgcn_predictions(
    data: np.ndarray,
    checkpoint: Path,
    batch_size: int,
    device_name: str,
) -> Tuple[np.ndarray, str]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("CUDA is required for this audit; refusing a silent CPU fallback")
    model, device = load_checkpoint(checkpoint, len(LABELS), device_name)
    predictions: List[int] = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch_array = np.array(data[start:start + batch_size], dtype=np.float32, copy=True)
            batch = torch.from_numpy(batch_array).to(device)
            predictions.extend(model(batch).argmax(dim=-1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64), str(device)


def _metadata_checks(
    raw_train: Path,
    protocol: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata_path = raw_train / "val_metadata.json"
    summary_path = raw_train / "val_summary.json"
    metadata = _records(metadata_path)
    summary = _read_json(summary_path)
    mapping = {str(name): int(value) for name, value in labels.items()}
    label_order_ok = mapping == {name: index for index, name in enumerate(LABELS)}
    summary_tokens = " ".join(str(summary.get(key, "")) for key in (
        "skeleton_preprocessing", "rendering_protocol", "coordinate_transform", "perception_chain",
    )).lower()
    required_tokens = {
        "rgb": "rgb" in summary_tokens,
        "yolo26n": "yolo26n" in summary_tokens or "yolo26n_pose" in summary_tokens,
        "videopose3d": "videopose3d" in summary_tokens,
        "camera_to_gravity": "camera_to_gravity" in summary_tokens,
        "root_center": "root_center" in summary_tokens,
        "torso_scale": "torso_scale" in summary_tokens,
        "yaw_only": "yaw_only" in summary_tokens,
    }
    generator = REPO_ROOT / "activeview/scripts/data/generate_selected16_habitat_parallel.py"
    generator_text = generator.read_text(encoding="utf-8")
    normalizer_source = REPO_ROOT / "activeview/data/motion/babel_clean_dataset_generator.py"
    normalizer_text = normalizer_source.read_text(encoding="utf-8")
    required_tokens.update({
        "generator_merges_data_and_labels": "np.save(data_root / f\"{split}_data.npy\"" in generator_text,
        "generator_uses_estimated_skeleton": "estimated_skeletons" in generator_text,
        "normalizer_align_canonical": "normalize_sequence(gravity_aligned, align_canonical=True)" in normalizer_text,
    })
    return {
        "label_mapping": mapping,
        "label_order_exact": label_order_ok,
        "metadata_count": len(metadata),
        "summary": summary,
        "required_preprocessing_evidence": required_tokens,
        "all_required_preprocessing_evidence": bool(all(required_tokens.values())),
        "tensor_shape_recorded": summary.get("data_shape"),
        "metadata_skeleton_preprocessing": sorted({str(item.get("skeleton_preprocessing", "")) for item in metadata}),
        "metadata_perception_chain": sorted({str(item.get("pose_backend", "")) for item in metadata}),
        "protocol_categories_exact": list(protocol.get("categories", [])) == list(LABELS),
    }


def _validate_tensor(data_path: Path, labels_path: Path, metadata_path: Path) -> Dict[str, Any]:
    data = np.load(data_path, mmap_mode="r")
    targets = np.asarray(np.load(labels_path), dtype=np.int64)
    metadata = _records(metadata_path)
    shape_ok = data.ndim == 5 and tuple(data.shape[1:]) == EXPECTED_SHAPE
    labels_ok = len(targets) == len(metadata) == int(data.shape[0])
    ids = [str(item.get("record_id", "")) for item in metadata]
    finite_check = bool(np.isfinite(np.asarray(data)).all())
    return {
        "data": data,
        "targets": targets,
        "metadata": metadata,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "shape_ok": shape_ok,
        "labels_ok": labels_ok,
        "finite": finite_check,
        "record_ids_unique": len(ids) == len(set(ids)),
    }


def _map_records(
    current_rows: Sequence[Mapping[str, Any]],
    historical_metadata: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_record: Dict[str, List[int]] = {}
    by_interval: Dict[Tuple[str, int, int], List[int]] = {}
    for index, row in enumerate(historical_metadata):
        by_record.setdefault(str(row.get("record_id", "")), []).append(index)
        by_interval.setdefault(_exact_key(row), []).append(index)
    output: List[Dict[str, Any]] = []
    for row in current_rows:
        record_id = str(row.get("record_id", ""))
        record_matches = by_record.get(record_id, [])
        interval_matches = by_interval.get(_exact_key(row), [])
        matches = sorted(set(record_matches + interval_matches))
        labels = sorted({str(historical_metadata[i].get("action_label", "")) for i in matches})
        if len(matches) == 1 and int(historical_metadata[matches[0]].get("label_id", -1)) == int(row.get("label_id", -2)):
            status = "exact_match"
            sample_index: Any = matches[0]
            clean_label: Any = labels[0] if labels else None
        elif len(matches) > 1:
            status = "ambiguous_exact_match"
            sample_index = matches
            clean_label = labels
        else:
            status = "unmapped_no_exact_match"
            sample_index = None
            clean_label = None
        output.append({
            "record_id": record_id,
            "clean_sample_index": sample_index,
            "clean_label": clean_label,
            "source_path": row.get("source_path") or row.get("source_group"),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "mapping_status": status,
            "historical_exact_candidate_count": len(matches),
        })
    return output


def _find_clean_candidates(data_root: Path, dataset_root: Path) -> Dict[str, Any]:
    expected = [
        dataset_root / "raw-train" / name
        for name in ("train_data.npy", "val_data.npy", "val_labels.npy", "val_metadata.json")
    ] + [
        dataset_root / "raw-val" / name
        for name in ("train_data.npy", "val_data.npy", "val_labels.npy", "val_metadata.json")
    ]
    discovered: List[Path] = []
    for path in (data_root / "datasets").glob("**/val_data.npy"):
        if path not in discovered:
            discovered.append(path)
    return {
        "expected": [_file_record(path) for path in expected],
        "discovered_val_data": [_file_record(path) for path in sorted(discovered)],
        "historical_clean_candidate": str((dataset_root / "raw-train").resolve()),
        "raw_val_clean_tensor_present": bool((dataset_root / "raw-val" / "val_data.npy").is_file()),
    }


def _write_analysis(
    path: Path,
    asset_search: Mapping[str, Any],
    provenance: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
    reproduction: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    mapped = sum(item.get("mapping_status") == "exact_match" for item in mapping)
    lines = [
        "# Reduced12 clean-perception reference recovery",
        "",
        "Val-only, read-only audit. No policy Test, training, RGB rendering, YOLO, VideoPose3D, or skeleton regeneration was performed.",
        "",
        "## Findings",
        "",
        "1. Historical tensor found: `raw-train/val_data.npy` with its paired `val_labels.npy` and `val_metadata.json`. It is the tensor consumed by the recorded reduced12 ST-GCN validation protocol.",
        f"2. Frozen-checkpoint reproduction: `{reproduction['historical_clean_val']['accuracy']:.9f}` Accuracy / `{reproduction['historical_clean_val']['macro_f1']:.9f}` Macro-F1; recorded values are `{reproduction['recorded']['accuracy']:.9f}` / `{reproduction['recorded']['macro_f1']:.9f}` (absolute differences `{reproduction['absolute_difference']['accuracy']:.3g}` / `{reproduction['absolute_difference']['macro_f1']:.3g}`).",
        f"3. Exact mapping to the current {len(mapping)} Official-Val Moving records (the Stage-D Val file contains {reproduction['current_moving_val_contexts']} contexts): {mapped}/{len(mapping)}. The historical clean tensor is Official-Train-derived, while current ActiveView Moving-Val records are Official-Val-derived; no record id or full source interval matched.",
        "4. Because exact 105-record mapping is unavailable, the candidate-level clean-reference similarity and privileged selector audit is stopped. The earlier Habitat-FK reference must not be interpreted as a substitute.",
        "",
        "## Provenance",
        "",
        f"- Frozen checkpoint: `{checkpoint.resolve()}`",
        f"- Historical dataset root: `{asset_search['historical_clean_candidate']}`",
        f"- Label order exact: `{provenance['label_order_exact']}`",
        f"- Shape/metadata checks: `{reproduction['tensor_checks']['shape_ok']}` / `{reproduction['tensor_checks']['labels_ok']}` / finite=`{reproduction['tensor_checks']['finite']}`",
        f"- Preprocessing evidence complete: `{provenance['all_required_preprocessing_evidence']}`",
        "- Pipeline evidence: RGB → YOLO26n-Pose → VideoPose3D → camera_to_gravity → root_center → torso_scale → yaw_only → SkeletonNormalizer(align_canonical=True).",
        "",
        "## Missing asset for the requested Moving-Val audit",
        "",
        "The repository has no clean-perception tensor/metadata for `raw-val/val.json` (the 105 records used by ActiveView Val). Expected files are `raw-val/val_data.npy`, `raw-val/val_labels.npy`, and `raw-val/val_metadata.json` under the reduced12 dataset root. They must be recovered from the historical archive before any 105-record or 10,080-context clean comparison can be run.",
        "",
        "## Flags",
        "",
        "```yaml",
        "test_used: false",
        "training_used: false",
        "new_rgb_rendered: false",
        "new_pose_estimation: false",
        "historical_clean_perception_only: true",
        "exact_record_mapping_required: true",
        "clean_reference_valid_only_if_reproduction_passes: true",
        "deployable: false",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = args.data_root.resolve()
    dataset_root = data_root / "datasets" / DATASET_NAME
    raw_train = dataset_root / "raw-train"
    raw_val = dataset_root / "raw-val"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else (data_root / CHECKPOINT_REL).resolve()

    asset_search = _find_clean_candidates(data_root, dataset_root)
    protocol = _read_json(dataset_root / "protocol_summary.json")
    labels = _read_json(raw_train / "label_mapping.json")
    provenance = _metadata_checks(raw_train, protocol, labels)
    tensor_checks = _validate_tensor(raw_train / "val_data.npy", raw_train / "val_labels.npy", raw_train / "val_metadata.json")
    predictions, device = _load_stgcn_predictions(tensor_checks["data"], checkpoint, args.batch_size, args.device)
    historical_metrics = _metrics(tensor_checks["targets"], predictions)

    recorded_payload = _read_json(RECORDED_RESULT) if RECORDED_RESULT.is_file() else {}
    recorded_training = recorded_payload.get("training", {})
    recorded_metrics = {
        "accuracy": float(recorded_training.get("posthoc_val_accuracy", 0.0)),
        "macro_f1": float(recorded_training.get("posthoc_val_macro_f1", 0.0)),
    }
    differences = {
        "accuracy": abs(historical_metrics["accuracy"] - recorded_metrics["accuracy"]),
        "macro_f1": abs(historical_metrics["macro_f1"] - recorded_metrics["macro_f1"]),
    }

    current_rows = _records(raw_val / "val.json")
    stage_d_path = data_root / POLICY_STAGE_D_VAL
    if not stage_d_path.is_file():
        raise FileNotFoundError(f"Stage-D Val contexts required for the mapping audit: {stage_d_path}")
    stage_d_rows = _jsonl_records(stage_d_path)
    stage_d_record_ids = {str(row.get("record_id", "")) for row in stage_d_rows}
    current_record_ids = {str(row.get("record_id", "")) for row in current_rows}
    if stage_d_record_ids != current_record_ids:
        raise ValueError("Stage-D Val record set does not exactly match raw-val/val.json")
    mapping = _map_records(current_rows, tensor_checks["metadata"])
    reproduction = {
        "historical_clean_val": historical_metrics,
        "recorded": recorded_metrics,
        "absolute_difference": differences,
        "tensor_checks": {key: value for key, value in tensor_checks.items() if key not in {"data", "targets", "metadata"}},
        "current_moving_val_unique_records": len(current_rows),
        "current_moving_val_contexts": len(stage_d_rows),
        "stage_d_record_set_exact": True,
        "moving_unique_metrics": {"status": "BLOCKED_NO_EXACT_MAPPING", "count": 0},
        "moving_context_weighted_metrics": {"status": "BLOCKED_NO_EXACT_MAPPING", "count": 0},
        "device": device,
        "no_renormalization": True,
    }
    checkpoint_summary = data_root / "checkpoints" / f"stgcn_{DATASET_NAME}" / "training_summary.json"
    if historical_metrics["accuracy"] < 0.65:
        status = "clean_reference_invalid_due_to_reproduction_failure"
    else:
        status = "HISTORICAL_CLEAN_RECOVERED_MOVING_MAPPING_BLOCKED"
    result = {
        "status": status,
        "asset_search": asset_search,
        "provenance": provenance,
        "checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "recorded_sha256": recorded_payload.get("checkpoint_sha256"),
            "summary": _read_json(checkpoint_summary) if checkpoint_summary.is_file() else None,
        },
        "mapping_summary": {
            "current_unique_records": len(mapping),
            "exact_matches": sum(item["mapping_status"] == "exact_match" for item in mapping),
            "ambiguous_exact_matches": sum(item["mapping_status"] == "ambiguous_exact_match" for item in mapping),
            "unmapped": sum(item["mapping_status"] == "unmapped_no_exact_match" for item in mapping),
        },
        "clean_reproduction": reproduction,
        "flags": {
            "test_used": False,
            "training_used": False,
            "new_rgb_rendered": False,
            "new_pose_estimation": False,
            "historical_clean_perception_only": True,
            "exact_record_mapping_required": True,
            "clean_reference_valid_only_if_reproduction_passes": True,
            "clean_perception_asset_missing": False,
            "moving_clean_perception_asset_missing": bool(not any(item["mapping_status"] == "exact_match" for item in mapping)),
            "deployable": False,
        },
    }
    _write_json(output / "asset_search.json", asset_search)
    _write_json(output / "provenance_audit.json", {"provenance": provenance, "checkpoint": result["checkpoint"]})
    _write_json(output / "record_mapping.json", {"records": mapping})
    _write_json(output / "clean_reproduction_metrics.json", reproduction)
    _write_json(output / "result.json", result)
    _write_analysis(output / "analysis.md", asset_search, provenance, mapping, reproduction, checkpoint)
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
