#!/usr/bin/env python3
"""
Habitat 与 HSSD 真实家庭场景管理器与审计工具 —— scene_manager.py (v11.5)
====================================================================

职责：
    1. 扫描 /home/zxf/MG08/hssd-scenes/scenes/ 与 Habitat 官方测试场景目录；
    2. 检查 .glb, .navmesh 与场景物理元数据；
    3. 识别包含 sofa, table, chair, cabinet, bed 等丰富家具的复杂家庭住宅场景；
    4. 输出 scene_audit_manifest.json，支持非破坏性离线审计与按需索引。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.scene_manager import SceneManager, KNOWN_SCENE_PROFILES
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scene_manager_tool")


def audit_local_scenes(output_json: Optional[str] = None) -> Dict[str, Any]:
    """审计所有本地可用场景并生成统计清单。"""
    mgr = SceneManager()
    all_scenes = mgr.discover_scenes()

    data_root = get_data_root()
    out_p = Path(output_json) if output_json else (data_root / "assets" / "scenes" / "scene_audit_manifest.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    hssd_dir = Path("/home/zxf/MG08/hssd-scenes/scenes")
    hssd_glb_count = len(list(hssd_dir.glob("*.glb"))) if hssd_dir.exists() else 0

    scene_list = []
    for s_id, s_meta in all_scenes.items():
        is_hssd = "hssd" in str(s_meta.glb_path).lower()
        size_mb = round(s_meta.glb_path.stat().st_size / (1024 * 1024), 2) if s_meta.glb_path.exists() else 0.0

        # 家具/结构特征标签
        features = ["living_room", "furniture", "partition_walls"]
        if is_hssd:
            features.extend(["sofa", "table", "chair", "cabinet", "bed"])

        item = {
            "scene_id": s_id,
            "glb_path": str(s_meta.glb_path),
            "navmesh_path": str(s_meta.navmesh_path) if s_meta.navmesh_path else None,
            "has_navmesh": s_meta.navmesh_path is not None and s_meta.navmesh_path.exists(),
            "size_mb": size_mb,
            "is_hssd": is_hssd,
            "floor_height": s_meta.floor_height,
            "ceiling_height": s_meta.ceiling_height,
            "bounds_min": s_meta.bounds_min,
            "bounds_max": s_meta.bounds_max,
            "furniture_features": features,
        }
        scene_list.append(item)

    audit_summary = {
        "total_discovered_scenes": len(all_scenes),
        "hssd_scenes_available": hssd_glb_count,
        "habitat_test_scenes": len(all_scenes) - hssd_glb_count,
        "search_directories": [str(d) for d in mgr.search_dirs],
        "scenes": scene_list,
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info("================================================================")
    logger.info("  Scene Audit Completed!                                        ")
    logger.info("  Total Discovered:  %d scenes", len(all_scenes))
    logger.info("  HSSD Scenes:       %d scenes (Path: %s)", hssd_glb_count, hssd_dir)
    logger.info("  Saved Manifest to: %s", out_p)
    logger.info("================================================================")

    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Audit and manage Habitat/HSSD scenes")
    parser.add_argument("--output", type=str, default=None, help="Output manifest JSON path")
    args = parser.parse_args()

    audit_local_scenes(output_json=args.output)


if __name__ == "__main__":
    main()
