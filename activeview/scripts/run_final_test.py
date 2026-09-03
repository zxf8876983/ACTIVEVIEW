#!/usr/bin/env python3
"""Run the explicitly unlocked EXP057 frozen Final Test protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root

MANIFEST = REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_method_manifest.json"
PROTOCOL = REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_test_protocol.json"
TEST_RGB_CACHE = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_test_final_union")
N_CLASSES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_f1(predicted: Sequence[int], labels: Sequence[int]) -> float:
    p = np.asarray(predicted, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    confusion = np.bincount(y * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    values: list[float] = []
    for cls in range(N_CLASSES):
        tp = float(confusion[cls, cls])
        precision = tp / float(confusion[:, cls].sum()) if confusion[:, cls].sum() else 0.0
        recall = tp / float(confusion[cls, :].sum()) if confusion[cls, :].sum() else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def _classification(predicted: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    p = np.asarray(predicted, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("Final Test classification population is empty or misaligned")
    return {"accuracy": float(np.mean(p == y)), "macro_f1": _macro_f1(p, y), "count": int(p.size)}


def _baseline(data_root: Path) -> dict[str, dict[str, float]]:
    """Evaluate frozen Stage-C initial predictions for FULL and MOVING."""
    from activeview.active_view.stage_d_dataset import load_jsonl

    path = data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/test_predictions.jsonl"
    rows = load_jsonl(path)
    if not rows or any(str(row.get("policy_split", "")).lower() != "test" for row in rows):
        raise ValueError(f"explicit policy_split=test required: {path}")
    full_pred = [int(row["current_predicted_label_id"]) for row in rows]
    full_labels = [int(row["label_id"]) for row in rows]
    moving = [row for row in rows if not bool(row["predicted_stays"])]
    moving_pred = [int(row["current_predicted_label_id"]) for row in moving]
    moving_labels = [int(row["label_id"]) for row in moving]
    return {"full": _classification(full_pred, full_labels), "moving": _classification(moving_pred, moving_labels)}


def _metric_pair(payload: dict[str, Any], key: str) -> dict[str, dict[str, float]]:
    metrics = payload[key]
    return {"full": metrics["full"], "moving": metrics["moving_subset"]}


def run_final_test(*, data_root: Path, device: torch.device, manifest_path: Path = MANIFEST, protocol_path: Path = PROTOCOL, output_dir: Path | None = None, rgb_cache_dir: Path = TEST_RGB_CACHE) -> dict[str, Any]:
    """Run all frozen methods after explicit Test unlock."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    test_state = manifest.get("test", {})
    if not bool(test_state.get("unlocked", False)):
        raise RuntimeError("Final Test is locked; explicit human unlock is required")
    if bool(test_state.get("used", False)):
        raise RuntimeError("Final Test has already been used by this freeze manifest")

    wm_path = Path(manifest["wm_e"]["path"])
    jr_multi = Path(manifest["joint_revision"]["checkpoint_path"])
    stgcn_path = Path(manifest["stgcn"]["checkpoint_path"])
    original_spec = next(item for item in protocol["methods"] if item["id"] == "ORIGINAL_JR_H2")
    jr_original = Path(original_spec["checkpoint"])
    for label, path, expected in (("WM-E", wm_path, manifest["wm_e"]["sha256"]), ("Original JR", jr_original, original_spec["sha256"]), ("Multi-positive JR", jr_multi, manifest["joint_revision"]["sha256"]), ("ST-GCN", stgcn_path, manifest["stgcn"]["sha256"])):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"{label} checkpoint provenance mismatch: {path}")

    # Import the evaluator only after the lock check; this path is never used in this freeze round.
    from activeview.scripts.run_exp051_r2_real_eval import run as run_r2

    out = (output_dir or (REPO_ROOT / "experiments/stage_d/FINAL_TEST")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = run_r2(data_root.resolve(), jr_original, device, wm_path, out / "original_jr_h2", split="test", rgb_cache_dir=rgb_cache_dir.resolve())
    multi = run_r2(data_root.resolve(), jr_multi, device, wm_path, out / "multi_positive_jr_h2", split="test", rgb_cache_dir=rgb_cache_dir.resolve())
    baseline = _baseline(data_root.resolve())
    methods = {"INITIAL_BASELINE": baseline, "H1_REAL": _metric_pair(original, "h1_real"), "ORIGINAL_JR_H2": _metric_pair(original, "h2_real"), "MULTI_POSITIVE_JR_H2": _metric_pair(multi, "h2_real")}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for name, left in (("MULTI_MINUS_INITIAL", "INITIAL_BASELINE"), ("MULTI_MINUS_H1", "H1_REAL"), ("MULTI_MINUS_ORIGINAL_H2", "ORIGINAL_JR_H2")):
        deltas[name] = {population: {metric: methods["MULTI_POSITIVE_JR_H2"][population][metric] - methods[left][population][metric] for metric in ("accuracy", "macro_f1")} for population in ("full", "moving")}
    result: dict[str, Any] = {"experiment_id": "FINAL_TEST", "status": "COMPLETED", "split": "test", "methods": methods, "deltas": deltas, "test_used": True, "training_performed": False, "protocol": protocol, "provenance": {"manifest": str(manifest_path.resolve()), "protocol": str(protocol_path.resolve()), "wm_e": str(wm_path.resolve()), "original_jr": str(jr_original.resolve()), "multi_positive_jr": str(jr_multi.resolve()), "stgcn": str(stgcn_path.resolve())}}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# EXP057 Final Test", "", "| Method | Full Acc | Full Macro-F1 | Moving Acc | Moving Macro-F1 |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {name} | {values['full']['accuracy']:.6f} | {values['full']['macro_f1']:.6f} | {values['moving']['accuracy']:.6f} | {values['moving']['macro_f1']:.6f} |" for name, values in methods.items())
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the explicitly authorized frozen EXP057 Final Test")
    parser.add_argument("--split", choices=("test",), default=None, help="must be explicitly supplied after Test unlock")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/stage_d/FINAL_TEST")
    parser.add_argument("--rgb-cache-dir", type=Path, default=TEST_RGB_CACHE)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split is None:
        print("EXP057 freeze manifest loaded; no evaluation requested (Test remains locked).")
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not bool(manifest.get("test", {}).get("unlocked", False)):
        raise RuntimeError("Final Test is locked; update the freeze manifest only after explicit human authorization")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback for Final Test")
    data_root = (args.data_root or get_data_root()).resolve()
    result = run_final_test(data_root=data_root, device=device, manifest_path=args.manifest.resolve(), protocol_path=args.protocol.resolve(), output_dir=args.output_dir.resolve(), rgb_cache_dir=args.rgb_cache_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
