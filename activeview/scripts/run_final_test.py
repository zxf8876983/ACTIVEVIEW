#!/usr/bin/env python3
"""Explicit Final Test entry point prepared by EXP057.

This freeze task never invokes it.  A future run must pass ``--split test``
after the human unlocks Test; without that flag the command is a no-op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_method_manifest.json"
PROTOCOL = REPO_ROOT / "experiments/stage_d/EXP057_final_method_freeze/final_test_protocol.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the explicitly authorized frozen Final Test protocol")
    parser.add_argument("--split", choices=("test",), default=None, help="must be explicitly supplied in the unlocked Test round")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split is None:
        print("EXP057 freeze manifest loaded; no evaluation requested (Test remains locked).")
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest["test"]["unlocked"]:
        raise RuntimeError("Final Test is locked; update the freeze manifest only after explicit human authorization")
    raise NotImplementedError("Final Test evaluator is intentionally not executed during EXP057")


if __name__ == "__main__":
    main()
