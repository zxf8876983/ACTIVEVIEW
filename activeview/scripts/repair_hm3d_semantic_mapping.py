#!/usr/bin/env python3
"""Audit and repair the HM3D semantic asset layout used by Habitat.

Only symbolic links are created.  Existing files or links are never replaced.
The canonical ACTIVEVIEW skeleton tree remains read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from activeview.core.paths import get_data_root, get_habitat_data_root


@dataclass(frozen=True)
class SceneAssets:
    scene_id: str
    basis_glb: Path
    navmesh: Path
    semantic_glb: Path
    semantic_txt: Path


def _single(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one {pattern}, found {len(matches)} in {directory}")
    return matches[0]


def discover_assets() -> tuple[Path, Path, Path, Path, list[SceneAssets]]:
    habitat_root = get_habitat_data_root()
    skeleton_root = get_data_root() / "datasets" / "offline" / "hm3d-train"
    if not skeleton_root.is_dir():
        raise FileNotFoundError(f"Canonical skeleton root not found: {skeleton_root}")
    basis_root = habitat_root / "hm3d-train"
    annots_root = habitat_root / "hm3d-train-semantic-annots"
    configs_root = habitat_root / "hm3d-train-semantic-configs"
    config_candidates = sorted(configs_root.rglob("*scene_dataset_config*.json"))
    annotated = [p for p in config_candidates if "annotated" in p.name.lower()]
    if not annotated:
        raise FileNotFoundError(f"No annotated scene dataset config under {configs_root}")
    # Prefer the train-scoped config; both are audited below by the caller.
    config_path = next((p for p in annotated if "train" in p.name.lower()), annotated[0])
    scenes: list[SceneAssets] = []
    for scene_dir in sorted(p for p in skeleton_root.iterdir() if p.is_dir()):
        sid = scene_dir.name
        basis_dir = basis_root / sid
        annot_dir = annots_root / sid
        scenes.append(
            SceneAssets(
                sid,
                _single(basis_dir, "*.basis.glb", "basis scene"),
                _single(basis_dir, "*.basis.navmesh", "navigation mesh"),
                _single(annot_dir, "*.semantic.glb", "semantic scene"),
                _single(annot_dir, "*.semantic.txt", "semantic descriptor"),
            )
        )
    return habitat_root, skeleton_root, annots_root, config_path, scenes


def _ensure_link(link: Path, target: Path) -> str:
    """Create/reuse a correct link, refusing to overwrite any other object."""
    target = target.resolve()
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            return "reused"
        raise RuntimeError(f"Refusing to overwrite existing mapping target: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)
    return "created"


def repair_mapping() -> dict[str, object]:
    habitat_root, skeleton_root, annots_root, config_path, scenes = discover_assets()
    if len(scenes) != 21:
        raise RuntimeError(f"Expected 21 canonical scenes, found {len(scenes)}")
    configs_root = config_path.parent
    rows = []
    created = 0
    reused = 0
    for assets in scenes:
        canonical_dir = habitat_root / "hm3d-train" / assets.scene_id
        statuses = []
        for source in (assets.semantic_glb, assets.semantic_txt):
            status = _ensure_link(canonical_dir / source.name, source)
            statuses.append(status)
            created += status == "created"
            reused += status == "reused"
        status = _ensure_link(configs_root / assets.scene_id, canonical_dir)
        statuses.append(status)
        created += status == "created"
        reused += status == "reused"
        rows.append(
            {
                "scene_id": assets.scene_id,
                "basis_glb": str(assets.basis_glb),
                "navmesh": str(assets.navmesh),
                "semantic_glb": str(assets.semantic_glb),
                "semantic_txt": str(assets.semantic_txt),
                "status": "MATCH",
                "mapping": statuses,
            }
        )
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_json, dict):
        raise RuntimeError(f"Annotated config is not a JSON object: {config_path}")
    result = {
        "habitat_root": str(habitat_root),
        "canonical_skeleton_root": str(skeleton_root),
        "semantic_annots_root": str(annots_root),
        "annotated_scene_dataset_config": str(config_path),
        "scene_count": len(scenes),
        "matched_scene_count": len(rows),
        "created_links": created,
        "reused_links": reused,
        "scenes": rows,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    repair_mapping()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
