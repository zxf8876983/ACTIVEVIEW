#!/usr/bin/env python3
"""
Habitat 场景资产检查与下载辅助工具 —— check_habitat_scenes.py
============================================================

职责：
    1. 检查本地环境中的 Habitat 与 HSSD 场景资产可用性；
    2. 输出场景数量、格式 (.glb/.navmesh) 与完整性审计报告；
    3. 提供 HuggingFace / Habitat-Sim 官方数据集获取指引与下载接口 (非强制静默下载)。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.scene_manager import SceneManager
from ea_avs_mvp_v11.core.paths import get_data_root, get_repo_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_habitat_scenes")


def check_habitat_scenes() -> Dict[str, Any]:
    """执行 Habitat 场景资产全量审计。"""
    mgr = SceneManager()
    all_scenes = mgr.discover_scenes()

    primary_scenes = mgr.get_primary_scenes(count=5)

    logger.info("=================================================================")
    logger.info("  ACTIVEVIEW Habitat Scene Asset Audit Report                   ")
    logger.info("=================================================================")
    logger.info("  Total Discovered Scenes: %d", len(all_scenes))
    logger.info("  Search Directories Checked:")
    for d in mgr.search_dirs:
        status = "EXISTS" if d.exists() else "MISSING"
        logger.info("    - [%s] %s", status, d.resolve())

    logger.info("-----------------------------------------------------------------")
    logger.info("  Primary Verified Multi-Scene Set:")
    for idx, s in enumerate(primary_scenes):
        nav_status = "NAV_AVAILABLE" if s.navmesh_path else "GLB_ONLY"
        size_mb = s.glb_path.stat().st_size / (1024 * 1024) if s.glb_path.exists() else 0.0
        logger.info("    [%d] %-20s | %-14s | %6.1f MB | %s",
                    idx + 1, s.scene_id, nav_status, size_mb, s.description)

    logger.info("=================================================================")

    # 检查是否满足多场景训练要求 (至少 3 个场景)
    is_sufficient = len(primary_scenes) >= 3
    if is_sufficient:
        logger.info("  Status: [PASS] Multi-scene requirement satisfied (>= 3 scenes available).")
    else:
        logger.warning("  Status: [WARNING] Scene count < 3. Additional scenes recommended.")
        logger.info("\n  Optional Scene Download Instructions:")
        logger.info("  1. Habitat-Sim Test Scenes (Official):")
        logger.info("     python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/")
        logger.info("  2. HSSD Habitat Synthetic Scenes (HuggingFace):")
        logger.info("     git clone https://huggingface.co/datasets/hssd/hssd-scenes /home/zxf/MG08/hssd-scenes\n")

    return {
        "total_scenes": len(all_scenes),
        "primary_scenes_count": len(primary_scenes),
        "is_sufficient": is_sufficient,
        "primary_scene_ids": [s.scene_id for s in primary_scenes],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit and Check Habitat Scene Assets")
    args = parser.parse_args()
    check_habitat_scenes()


if __name__ == "__main__":
    main()
