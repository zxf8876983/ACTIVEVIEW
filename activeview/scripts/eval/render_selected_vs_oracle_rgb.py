#!/usr/bin/env python3
"""Preflight and targeted renderer for the 12 qualitative failure cases.

The actual rendering path is deliberately gated on CUDA.  When the Habitat
GPU backend is unavailable this command writes an explicit blocked status and
exits without creating synthetic RGB or touching the original RGB cache.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root

DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization/qualitative_rgb"
FRAME_IDS = (0, 15, 29)


def _cuda_status() -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "PyTorch CUDA is unavailable"
    except Exception as exc:  # pragma: no cover - environment-specific
        return False, f"PyTorch import/CUDA check failed: {exc}"
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"nvidia-smi cannot communicate with the driver: {exc}"
    return True, "CUDA and NVIDIA driver available"


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    available, reason = _cuda_status()
    status = {
        "status": "READY_FOR_TARGETED_RENDER" if available else "BLOCKED_GPU",
        "reason": reason,
        "targeted_cases": 12,
        "targeted_viewpoints_per_case": 2,
        "frame_ids": list(FRAME_IDS),
        "new_rgb_rendered": False,
        "full_dataset_rgb_regenerated": False,
        "original_rgb_cache_modified": False,
        "test_used": False,
        "training_used": False,
    }
    (output / "render_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if not available:
        print(f"targeted RGB rendering blocked by Habitat/GPU: {reason}")
        return 2
    raise RuntimeError("GPU is available; targeted Habitat renderer must be invoked with the approved offline rendering integration.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    args = parser.parse_args()
    raise SystemExit(run(args.output.resolve()))


if __name__ == "__main__":
    main()
