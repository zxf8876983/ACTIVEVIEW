#!/usr/bin/env python3
"""文件用途：
    执行离线数据生成、划分或缓存构建入口。

主要输入：
    - 命令行参数与已有运行时数据。
主要输出：
    - 数据集、缓存或清单文件。
项目角色：
    - 属于 data 脚本入口，仅调用正式数据模块。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activeview.core.paths import get_habitat_data_root


def _scene_files(scene_dir: Path) -> tuple[Path, Path]:
    glbs = sorted(scene_dir.glob("*.basis.glb"))
    navmeshes = sorted(scene_dir.glob("*.basis.navmesh"))
    if not glbs or not navmeshes:
        raise FileNotFoundError(f"Missing HM3D GLB/navmesh in {scene_dir}")
    return glbs[0], navmeshes[0]


def _shortest_path(pathfinder: Any, start: np.ndarray, end: np.ndarray) -> float | None:
    import habitat_sim

    start_snap = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
    end_snap = np.asarray(pathfinder.snap_point(end), dtype=np.float32)
    if not np.isfinite(start_snap).all() or not np.isfinite(end_snap).all():
        return None
    if not pathfinder.is_navigable(start_snap) or not pathfinder.is_navigable(end_snap):
        return None
    request = habitat_sim.ShortestPath()
    request.requested_start = start_snap
    request.requested_end = end_snap
    if not pathfinder.find_path(request):
        return None
    value = float(request.geodesic_distance)
    return value if np.isfinite(value) and value >= 0.0 else None


def build_cache(scene_dir: Path, candidate_manifest_path: Path, output_root: Path) -> dict[str, Any]:
    import habitat_sim

    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("version") != "semantic-region-v2":
        raise ValueError("Pairwise cache requires semantic-region-v2 candidate metadata")
    scene_glb, navmesh = _scene_files(scene_dir)
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_glb)
    backend.enable_physics = False
    agent = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError(f"Unable to load navmesh {navmesh}")
        output_root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, str] = {}
        for region_payload in candidate_manifest["placements_data"]:
            placement_id = str(region_payload.get("placement_id", region_payload["region"]))
            region = str(region_payload["region"])
            views = sorted(region_payload["viewpoints"], key=lambda item: int(item["viewpoint_id"]))
            ids = [int(item["viewpoint_id"]) for item in views]
            if len(ids) != 32 or len(set(ids)) != 32:
                raise ValueError(f"Expected 32 unique viewpoints for {region}")
            positions = [np.asarray(item.get("snapped_position", item["position"]), dtype=np.float32) for item in views]
            matrix = np.full((32, 32), np.inf, dtype=np.float64)
            np.fill_diagonal(matrix, 0.0)
            for i, start in enumerate(positions):
                for j, end in enumerate(positions):
                    if i == j:
                        continue
                    value = _shortest_path(sim.pathfinder, start, end)
                    if value is not None:
                        matrix[i, j] = value
            payload = {
                "schema": "activeview-pairwise-viewpoint-geodesic-v1",
                "scene_id": scene_dir.name,
                "placement_id": placement_id,
                "region": region,
                "viewpoint_ids": ids,
                "geodesic_distance_m": matrix.tolist(),
                "source_scene_glb": str(scene_glb.resolve()),
                "source_navmesh": str(navmesh.resolve()),
                "source_candidate_manifest": str(candidate_manifest_path.resolve()),
                "rendering_performed": False,
                "perception_performed": False,
            }
            target = output_root / f"{region}.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            outputs[region] = str(target.resolve())
        summary = {"scene_id": scene_dir.name, "regions": outputs, "rendering_performed": False, "perception_performed": False}
        (output_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Raw Habitat scene directory containing GLB/navmesh")
    parser.add_argument("--candidate-manifest", type=Path, required=True, help="Existing offline semantic-region-v2 manifest")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--habitat-root", type=Path, default=get_habitat_data_root())
    args = parser.parse_args()
    # The explicit root is accepted for reproducible command manifests; scene-dir
    # itself must already resolve inside the configured Habitat data tree.
    if args.habitat_root not in args.scene_dir.resolve().parents and args.scene_dir.resolve() != args.habitat_root.resolve():
        raise ValueError("scene-dir must be below --habitat-root")
    print(json.dumps(build_cache(args.scene_dir, args.candidate_manifest, args.output_root), indent=2))


if __name__ == "__main__":
    main()
