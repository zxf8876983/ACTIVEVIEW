#!/usr/bin/env python3
"""Check the external Habitat Conda runtime used by targeted rendering."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run() -> tuple[dict[str, object], int]:
    payload: dict[str, object] = {"sys_executable": sys.executable, "python_version": sys.version.split()[0], "habitat_import_ok": False, "magnum_import_ok": False, "cuda_available_in_habitat_env": False, "torch_version_cuda": None, "gpu_name": None, "nvidia_smi_ok": False}
    try:
        import habitat_sim  # noqa: F401

        payload["habitat_import_ok"] = True
    except Exception as exc:  # pragma: no cover - environment-specific
        payload["habitat_error"] = repr(exc)
    try:
        import magnum  # noqa: F401

        payload["magnum_import_ok"] = True
    except Exception as exc:  # pragma: no cover - environment-specific
        payload["magnum_error"] = repr(exc)
    try:
        import torch

        payload["cuda_available_in_habitat_env"] = bool(torch.cuda.is_available())
        payload["torch_version_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment-specific
        payload["torch_error"] = repr(exc)
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        payload["nvidia_smi_ok"] = True
        payload["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        payload["nvidia_smi_error"] = str(exc)
    ok = bool(payload["habitat_import_ok"] and payload["magnum_import_ok"] and payload["cuda_available_in_habitat_env"] and payload["nvidia_smi_ok"])
    payload["status"] = "READY" if ok else "UNAVAILABLE"
    return payload, 0 if ok else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload, code = run()
    print(json.dumps(payload, ensure_ascii=False))
    if not args.json:
        print(f"Habitat runtime: {payload['status']}", file=sys.stderr)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
