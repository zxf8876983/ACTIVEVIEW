#!/usr/bin/env python3
"""Generate the reduced12 ActiveView policy dataset for semantic scenes 1--20.

The scene list is derived from the semantic-annotation directory (numeric
prefix order), while all motion records come from the canonical reduced12
Official-Val manifest.  Existing ActiveView generation and placement helpers
are invoked unchanged; this command only supplies their explicit paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root

LOGGER = logging.getLogger("activeview.reduced12_policy_1_20")
SCENE_PREFIX = re.compile(r"^(\d{5})-")


def select_scenes(semantic_root: Path, scene_root: Path, count: int) -> List[str]:
    """Select scenes by semantic-annotation folder numeric prefix."""
    candidates = []
    for path in semantic_root.iterdir():
        if not path.is_dir():
            continue
        match = SCENE_PREFIX.match(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path.name))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = [name for _prefix, name in candidates[:count]]
    if len(selected) != count:
        raise RuntimeError(f"semantic annotation root contains only {len(selected)} numeric scenes")
    missing = [
        scene_id
        for scene_id in selected
        if not (scene_root / scene_id).is_dir()
        or not list((scene_root / scene_id).glob("*.basis.glb"))
        or not list((scene_root / scene_id).glob("*.basis.navmesh"))
    ]
    if missing:
        raise FileNotFoundError(f"HM3D geometry/navmesh missing for selected scenes: {missing}")
    return selected


def _run(command: Sequence[str]) -> None:
    LOGGER.info("$ %s", " ".join(command))
    subprocess.run(list(command), check=True)


def _write_scene_list(path: Path, scenes: Sequence[str], semantic_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "reduced12-semantic-scenes-1-20-v1",
        "selection_rule": "semantic annotation folder numeric prefix ascending",
        "source_semantic_root": str(semantic_root.resolve()),
        "scene_count": len(scenes),
        "scene_ids": list(scenes),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=get_habitat_data_root() / "hm3d-train-semantic-annots",
    )
    parser.add_argument(
        "--motion-manifest",
        type=Path,
        default=data_root
        / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=data_root / "datasets/offline/habitat-train/00006-00087",
    )
    parser.add_argument("--scene-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=data_root / "checkpoints/ultralytics/yolo26n-pose.pt",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-furniture", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scene_count < 1 or args.workers < 1:
        raise ValueError("scene-count and workers must be positive")
    scene_root = args.scene_root.resolve()
    semantic_root = args.semantic_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = dataset_root
    furniture_root = dataset_root / "semantic_furniture"
    placement_root = dataset_root / "placement_sampling_v2"
    scene_list = dataset_root / "scene_selection.json"
    scenes = select_scenes(semantic_root, scene_root, args.scene_count)
    _write_scene_list(scene_list, scenes, semantic_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    visualizer = REPO_ROOT / "activeview/scripts/visualize/visualize_hm3d_semantic_topdown.py"
    if not args.skip_furniture:
        for scene_id in scenes:
            target = furniture_root / scene_id / "furniture_positions.json"
            if target.exists():
                continue
            semantic_dir = semantic_root / scene_id
            semantic_glb = next(semantic_dir.glob("*.semantic.glb"))
            semantic_txt = next(semantic_dir.glob("*.semantic.txt"))
            _run(
                [
                    sys.executable,
                    str(visualizer),
                    "--semantic-glb",
                    str(semantic_glb),
                    "--semantic-txt",
                    str(semantic_txt),
                    "--output-dir",
                    str(target.parent),
                ]
            )
    sampler = REPO_ROOT / "activeview/scripts/data/sample_hm3d_train_placements.py"
    _run(
        [
            sys.executable,
            str(sampler),
            "--scene-list",
            str(scene_list),
            "--scene-root",
            str(scene_root),
            "--semantic-furniture-root",
            str(furniture_root),
            "--output-root",
            str(placement_root),
            "--raw-val-manifest",
            str(args.motion_manifest.resolve()),
            "--num-placements",
            "8",
            "--seed",
            "42",
        ]
    )
    orchestrator = REPO_ROOT / "activeview/scripts/data/generate_hm3d_train_offline.py"
    command = [
        sys.executable,
        str(orchestrator),
        "--scene-list",
        str(scene_list),
        "--scene-root",
        str(scene_root),
        "--semantic-root",
        str(semantic_root),
        "--manifest",
        str(args.motion_manifest.resolve()),
        "--output-root",
        str(output_root),
        "--placement-root",
        str(placement_root),
        "--workers",
        str(args.workers),
        "--image-size",
        str(args.image_size),
        "--target-frames",
        str(args.target_frames),
        "--device",
        args.device,
        "--yolo-weights",
        str(args.yolo_weights.resolve()),
    ]
    if args.max_records is not None:
        command.extend(["--max-records", str(args.max_records)])
    _run(command)
    # The canonical orchestrator also writes scene_selection.json.  Restore
    # the explicit semantic-annotation provenance after it completes.
    _write_scene_list(scene_list, scenes, semantic_root)
    summary_path = output_root / "dataset_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "version": "activeview-policy-reduced12-eight-placement-1-20-v1",
                "source_motion_manifest": str(args.motion_manifest.resolve()),
                "source_semantic_root": str(semantic_root),
                "scene_selection_file": str(scene_list),
                "scene_selection_rule": "semantic annotation folder numeric prefix ascending",
                "scene_count": len(scenes),
                "placements_per_scene": 8,
                "test_used": False,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("POLICY_DATASET_COMPLETE scenes=%d output=%s", len(scenes), output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
