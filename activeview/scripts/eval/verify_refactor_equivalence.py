#!/usr/bin/env python3
"""文件用途：
    验证项目结构重构前后冻结 Test 方法的预测结果是否保持一致。

主要输入：
    - 已冻结 Test 数据和 cache
    - WM-E、Original JR、Multi-positive JR 与 ST-GCN checkpoint
    - 已保存的正式 FINAL_TEST golden result
主要输出：
    - 重构后重新计算的 Test 指标、差值和 PASS/FAIL 结论。
项目角色：
    - 仅用于代码重构后的数值等价性验证，不用于模型选择或调参。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.evaluation.evaluator import summarize_methods
from activeview.methods.baselines.policies import (
    _by_id,
    _selected_row,
    build_baseline_trajectories,
    build_single_step_oracles,
)
from activeview.methods.active_view.rollout import run_real_observation_evaluation


N_CLASSES = 16
TOLERANCE = 1e-8
GOLDEN = {
    "NoMove": {
        "full": {"accuracy": 0.4126615362276753, "macro_f1": 0.38178632403054913},
        "moving": {"accuracy": 0.2629397385481985, "macro_f1": 0.2554756653321506},
    },
    "Random": {
        "full": {"accuracy": 0.4087411064324089, "macro_f1": 0.38205773707285734},
        "moving": {"accuracy": 0.36263152300988416, "macro_f1": 0.34946117348635053},
    },
    "FrozenStageCv0": {
        "full": {"accuracy": 0.625381152896762, "macro_f1": 0.5637051739097438},
        "moving": {"accuracy": 0.5743437134658306, "macro_f1": 0.5305012321646901},
    },
    "SafeOracle": {
        "full": {"accuracy": 0.8449252214316829, "macro_f1": 0.8111121904686223},
        "moving": {"accuracy": 0.8254862365819959, "macro_f1": 0.8004618896969267},
    },
    "H1_REAL": {
        "full": {"accuracy": 0.6735153187164222, "macro_f1": 0.6230498724709267},
        "moving": {"accuracy": 0.6448081623977043, "macro_f1": 0.6124965376813798},
    },
    "ORIGINAL_JR_H2": {
        "full": {"accuracy": 0.680557572237549, "macro_f1": 0.6291110635147539},
        "moving": {"accuracy": 0.6551174407482198, "macro_f1": 0.6218520494358288},
    },
    "MULTI_POSITIVE_JR_H2": {
        "full": {"accuracy": 0.6848410047916365, "macro_f1": 0.6277488276320746},
        "moving": {"accuracy": 0.6613880327346158, "macro_f1": 0.6229838484529353},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _random_rows(records: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for record in records:
        actions: list[int | None] = [None] + [int(item["viewpoint_id"]) for item in record["candidates"]]
        selected_id = actions[int(rng.integers(0, len(actions)))]
        if selected_id is None:
            output.append(_selected_row(record, None, moves=0, cost=0.0))
            continue
        candidate = _by_id(record)[selected_id]
        output.append(_selected_row(record, selected_id, moves=1, cost=float(candidate["geodesic_distance_m"])))
    return output


def _metric_pair(summary: Mapping[str, Any]) -> dict[str, float | int]:
    recognition = summary["recognition"]
    return {
        "accuracy": float(recognition["accuracy"]),
        "macro_f1": float(recognition["macro_f1"]),
        "count": int(recognition["n"]),
    }


def _check_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    if any(str(row.get("policy_split", "")).lower() != split for row in rows):
        raise ValueError(f"{name} requires explicit policy_split={split}")


def _baseline_metrics(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    stage_b_path = data_root / "datasets/policy_v11_5/stage_b/utility_labels/test.jsonl"
    v0_path = data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/test_predictions.jsonl"
    stage_b_rows = load_jsonl(stage_b_path)
    v0_rows = load_jsonl(v0_path)
    _check_split(stage_b_rows, "test", "Stage-B Test rows")
    _check_split(v0_rows, "test", "frozen-v0 Test rows")
    stage_ids = [str(row["episode_id"]) for row in stage_b_rows]
    v0_ids = [str(row["episode_id"]) for row in v0_rows]
    if len(stage_ids) != len(v0_ids) or set(stage_ids) != set(v0_ids):
        raise ValueError("Stage-B and frozen-v0 Test episode IDs are not aligned")

    baselines = build_baseline_trajectories(stage_b_rows, v0_rows)
    oracles = build_single_step_oracles(stage_b_rows)
    method_rows: dict[str, Sequence[Mapping[str, Any]]] = {
        "NoMove": baselines["NoMove"],
        "Random": _random_rows(stage_b_rows, 42),
        "FrozenStageCv0": baselines["FrozenStageCv0"],
        "SafeOracle": oracles["SafeOracle"],
    }
    moving_ids = {str(row["episode_id"]) for row in v0_rows if not bool(row["predicted_stays"])}
    moving_rows = {
        name: [row for row in rows if str(row["episode_id"]) in moving_ids]
        for name, rows in method_rows.items()
    }
    classes = [str(index) for index in range(N_CLASSES)]
    full = summarize_methods(method_rows, classes)
    moving = summarize_methods(moving_rows, classes)
    metrics = {
        name: {"full": _metric_pair(full[name]), "moving": _metric_pair(moving[name])}
        for name in method_rows
    }
    return metrics, {"full": len(stage_b_rows), "moving": len(moving_ids)}


def _rollout_metrics(result: Mapping[str, Any], key: str, population: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    values = result[key]
    return {
        "full": {**values["full"], "count": int(population["full"])},
        "moving": {**values["moving_subset"], "count": int(population["moving"])},
    }


def _compare(current: Mapping[str, Any], golden: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    passed = True
    for population in ("full", "moving"):
        output[population] = {}
        for metric in ("accuracy", "macro_f1"):
            value = float(current[population][metric])
            expected = float(golden[population][metric])
            delta = value - expected
            metric_pass = abs(delta) <= TOLERANCE
            output[population][metric] = {
                "current": value,
                "golden": expected,
                "delta": delta,
                "pass": metric_pass,
            }
            passed = passed and metric_pass
    output["pass"] = passed
    return output


def verify(*, data_root: Path, output_dir: Path, device: torch.device, manifest: Path, protocol: Path, rgb_cache_dir: Path) -> dict[str, Any]:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    if not bool(manifest_payload.get("test", {}).get("used")) or not bool(manifest_payload.get("test", {}).get("unlocked")):
        raise RuntimeError("frozen Test manifest must remain used=true and unlocked=true")
    methods = protocol_payload["methods"]
    original_spec = next(item for item in methods if item["id"] == "ORIGINAL_JR_H2")
    wm = Path(manifest_payload["wm_e"]["path"])
    original = Path(original_spec["checkpoint"])
    multi = Path(manifest_payload["joint_revision"]["checkpoint_path"])
    stgcn = Path(manifest_payload["stgcn"]["checkpoint_path"])
    expected_hashes = {
        "WM-E": (wm, manifest_payload["wm_e"]["sha256"]),
        "Original JR": (original, original_spec["sha256"]),
        "Multi-positive JR": (multi, manifest_payload["joint_revision"]["sha256"]),
        "ST-GCN": (stgcn, manifest_payload["stgcn"]["sha256"]),
    }
    checkpoint_hashes = {}
    for name, (path, expected) in expected_hashes.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} checkpoint hash mismatch: {path}")
        checkpoint_hashes[name] = actual

    baseline_metrics, population = _baseline_metrics(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_result = run_real_observation_evaluation(
        data_root,
        original,
        device,
        wm,
        output_dir / "original_jr_h2",
        split="test",
        rgb_cache_dir=rgb_cache_dir,
    )
    multi_result = run_real_observation_evaluation(
        data_root,
        multi,
        device,
        wm,
        output_dir / "multi_positive_jr_h2",
        split="test",
        rgb_cache_dir=rgb_cache_dir,
    )
    if int(original_result["population"]["full"]) != population["full"] or int(multi_result["population"]["full"]) != population["full"]:
        raise RuntimeError("rollout and baseline full populations differ")
    if int(original_result["population"]["moving_subset"]) != population["moving"] or int(multi_result["population"]["moving_subset"]) != population["moving"]:
        raise RuntimeError("rollout and baseline moving populations differ")

    current = {
        **baseline_metrics,
        "H1_REAL": _rollout_metrics(original_result, "h1_real", population),
        "ORIGINAL_JR_H2": _rollout_metrics(original_result, "h2_real", population),
        "MULTI_POSITIVE_JR_H2": _rollout_metrics(multi_result, "h2_real", population),
    }
    comparisons = {name: _compare(current[name], GOLDEN[name]) for name in GOLDEN}
    result: dict[str, Any] = {
        "experiment_id": "REFACTOR_EQUIVALENCE_CHECK",
        "status": "PASS" if all(item["pass"] for item in comparisons.values()) else "FAIL",
        "commit": "unknown",
        "split": "test",
        "population": population,
        "current_metrics": current,
        "golden_metrics": GOLDEN,
        "comparisons": comparisons,
        "checkpoint_hashes": checkpoint_hashes,
        "test_used": True,
        "training_performed": False,
        "algorithm_changed": False,
        "habitat_rendering_performed": False,
        "perception_regenerated": False,
        "official_final_test_overwritten": False,
        "official_test_manifest_modified": False,
        "provenance": {
            "manifest": str(manifest.resolve()),
            "protocol": str(protocol.resolve()),
            "rgb_cache_dir": str(rgb_cache_dir.resolve()),
            "random_seed": 42,
            "tolerance": TOLERANCE,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Refactor Equivalence Check", "", f"- Overall: **{result['status']}**", f"- Population: FULL {population['full']}, MOVING {population['moving']}", "", "| Method | Population | Current Accuracy | Golden Accuracy | Δ | Current Macro-F1 | Golden Macro-F1 | Δ |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in GOLDEN:
        for pop in ("full", "moving"):
            row = comparisons[name][pop]
            lines.append(f"| {name} | {pop} | {row['accuracy']['current']:.16f} | {row['accuracy']['golden']:.16f} | {row['accuracy']['delta']:+.3e} | {row['macro_f1']['current']:.16f} | {row['macro_f1']['golden']:.16f} | {row['macro_f1']['delta']:+.3e} |")
    lines.extend(["", "No official FINAL_TEST result or manifest was modified."])
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/refactor_regression")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_method_manifest.json")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_test_protocol.json")
    parser.add_argument("--rgb-cache-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_test_final"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    result = verify(
        data_root=(args.data_root or get_data_root()).resolve(),
        output_dir=args.output_dir.resolve(),
        device=device,
        manifest=args.manifest.resolve(),
        protocol=args.protocol.resolve(),
        rgb_cache_dir=args.rgb_cache_dir.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
