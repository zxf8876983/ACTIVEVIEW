#!/usr/bin/env python3
"""Launch targeted selected/oracle RGB rendering in an external Habitat env."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization"
FRAME_IDS = (0, 15, 29)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _python_candidates() -> list[Path]:
    candidates: list[Path] = []
    conda = Path("/home/zxf/anaconda3/bin/conda")
    if conda.is_file():
        try:
            result = subprocess.run([str(conda), "env", "list", "--json"], check=True, capture_output=True, text=True)
            for prefix in json.loads(result.stdout).get("envs", []):
                candidates.append(Path(prefix) / "bin/python")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    for root in (Path("/data-1T/zxf/anaconda3/envs"), Path("/home/zxf/anaconda3/envs"), Path.home() / "anaconda3/envs", Path.home() / "miniconda3/envs"):
        if root.is_dir():
            candidates.extend(sorted(root.glob("*/bin/python")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _probe(candidate: Path) -> dict[str, Any]:
    checker = REPO_ROOT / "activeview/scripts/eval/check_habitat_runtime.py"
    result = subprocess.run([str(candidate), str(checker), "--json"], capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"status": "CHECK_FAILED", "error": result.stderr.strip() or result.stdout.strip()}
    payload["habitat_python"] = str(candidate)
    payload["check_returncode"] = result.returncode
    return payload


def _select_python(explicit: Path | None) -> tuple[Path | None, dict[str, Any]]:
    candidates = [explicit.resolve()] if explicit is not None else _python_candidates()
    probes = [(candidate, _probe(candidate)) for candidate in candidates]
    valid = [(candidate, payload) for candidate, payload in probes if payload.get("habitat_import_ok") and payload.get("magnum_import_ok")]
    valid.sort(key=lambda item: bool(item[1].get("cuda_available_in_habitat_env")), reverse=True)
    if not valid:
        return None, {"status": "NO_HABITAT_RUNTIME", "candidates": [payload for _, payload in probes]}
    return valid[0]


def _status_base(habitat_python: Path | None, probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "launcher_python": sys.executable,
        "habitat_python": str(habitat_python) if habitat_python else None,
        "habitat_import_ok": bool(probe.get("habitat_import_ok", False)),
        "magnum_import_ok": bool(probe.get("magnum_import_ok", False)),
        "cuda_available_in_habitat_env": bool(probe.get("cuda_available_in_habitat_env", False)),
        "gpu_name": probe.get("gpu_name"),
        "renderer_initialized": False,
        "rendered_cases": 0,
        "rendered_images": 0,
        "targeted_cases": 12,
        "targeted_viewpoints_per_case": 2,
        "frame_ids": list(FRAME_IDS),
        "full_dataset_rgb_regenerated": False,
        "test_used": False,
        "training_used": False,
    }


def _run_visualization(output: Path) -> None:
    script = REPO_ROOT / "activeview/scripts/eval/visualize_reduced12_selected_vs_oracle.py"
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/activeview-mplconfig")
    subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), check=True, env=env)


def run(habitat_python: Path | None, output: Path, data_root: Path, scene_root: Path, archive_root: Path, motion_manifest: Path) -> int:
    output = output.resolve()
    runtime_status_path = output / "qualitative_rgb/runtime_status.json"
    render_status_path = output / "qualitative_rgb/render_status.json"
    selected, probe = _select_python(habitat_python)
    status = _status_base(selected, probe)
    if selected is None:
        status.update({"status": "BLOCKED_EXTERNAL_RUNTIME", "reason": "No Habitat Python with habitat_sim and magnum was found"})
        _write_json(runtime_status_path, status)
        _write_json(render_status_path, {"status": status["status"], "reason": status["reason"], "new_rgb_rendered": False, "full_dataset_rgb_regenerated": False, "test_used": False, "training_used": False})
        _run_visualization(output)
        return 2
    if not status["cuda_available_in_habitat_env"]:
        status.update({"status": "BLOCKED_EXTERNAL_RUNTIME", "reason": "External Habitat environment cannot access CUDA/NVIDIA driver"})
        _write_json(runtime_status_path, status)
        _write_json(render_status_path, {"status": status["status"], "reason": status["reason"], "habitat_python": str(selected), "new_rgb_rendered": False, "full_dataset_rgb_regenerated": False, "test_used": False, "training_used": False})
        _run_visualization(output)
        return 2
    worker = REPO_ROOT / "activeview/scripts/eval/render_selected_vs_oracle_rgb_worker.py"
    command = [str(selected), str(worker), "--case-manifest", str(output / "case_manifest.json"), "--output", str(output / "qualitative_rgb"), "--data-root", str(data_root.resolve()), "--scene-root", str(scene_root.resolve()), "--archive-root", str(archive_root.resolve()), "--motion-manifest", str(motion_manifest.resolve()), "--runtime-status", str(runtime_status_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(command, cwd=str(REPO_ROOT), check=True, env=env)
    except subprocess.CalledProcessError as exc:
        status.update({"status": "RENDER_FAILED", "reason": f"external worker exited with code {exc.returncode}"})
        _write_json(runtime_status_path, status)
        _write_json(render_status_path, {"status": status["status"], "reason": status["reason"], "habitat_python": str(selected), "new_rgb_rendered": False, "full_dataset_rgb_regenerated": False, "test_used": False, "training_used": False})
        _run_visualization(output)
        return exc.returncode or 2
    _run_visualization(output)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat-python", type=Path, help="absolute path to the external Habitat Conda Python")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=Path("/home/zxf/WorkSpace/code/data/ActiveView"))
    parser.add_argument("--scene-root", type=Path, default=Path("/home/zxf/WorkSpace/code/code/robot/DATA/hm3d-train"))
    parser.add_argument("--archive-root", type=Path, default=Path("/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/habitat-train/00006-00087"))
    parser.add_argument("--motion-manifest", type=Path, default=Path("/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"))
    args = parser.parse_args()
    raise SystemExit(run(args.habitat_python, args.output, args.data_root, args.scene_root, args.archive_root, args.motion_manifest))


if __name__ == "__main__":
    main()
