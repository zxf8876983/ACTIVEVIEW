"""
AMASS 动作批量转换脚本 —— convert_amass_motions.py
==================================================

输入：
    - motion_asset_manifest.json (AMASS 动作清单与帧切片元数据)
    - Humanoid URDF (neutral_0)

输出：
    - Converted Habitat SMPL-X .pkl 动作文件集合 (保存在 ../../data/ActiveView/assets/motions/converted/)

运行方式：
    python -m ea_avs_mvp_v7.scripts.convert_amass_motions [--action fall_related] [--all]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from tools.motion_assets.data_paths import get_data_root, get_assets_dir
from ea_avs_mvp_v7.humanoid.humanoid_loader import resolve_humanoid_assets
from ea_avs_mvp_v7.motion.motion_converter import AMASSMotionConverter, batch_convert_manifest_motions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_amass_motions")


def main():
    parser = argparse.ArgumentParser(description="Convert AMASS Motions to Habitat Humanoid PKL format")
    parser.add_argument("--config", type=str, default="ea_avs_mvp_v7/configs/v70_humanoid_sim.yaml", help="Path to config yaml")
    parser.add_argument("--manifest", type=str, default=None, help="Path to motion_asset_manifest.json")
    parser.add_argument("--action", type=str, default=None, help="Filter by target action class (e.g. fall_related)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for converted .pkl files")
    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    humanoid_assets = resolve_humanoid_assets(cfg)
    logger.info("Using Humanoid URDF: %s", humanoid_assets.urdf_path)

    data_root = get_data_root()
    manifest_path = Path(args.manifest) if args.manifest else data_root / cfg.get("motion", {}).get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    output_dir = Path(args.output_dir) if args.output_dir else data_root / cfg.get("motion", {}).get("converted_dir", "assets/motions/converted")

    logger.info("Manifest: %s", manifest_path)
    logger.info("Output dir: %s", output_dir)
    if args.action:
        logger.info("Filtering action: %s", args.action)

    converted_files = batch_convert_manifest_motions(
        manifest_path=manifest_path,
        urdf_path=humanoid_assets.urdf_path,
        output_dir=output_dir,
        target_class_filter=args.action,
    )

    print("\n" + "=" * 65)
    print(f"[Conversion Summary] Successfully converted {len(converted_files)} motions:")
    for p in converted_files:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
