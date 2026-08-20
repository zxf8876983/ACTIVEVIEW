"""
AMASS 动作批量转换脚本 —— convert_motions.py
============================================

功能：
    1. 读取 motion_asset_manifest.json；
    2. 使用 AMASSLoader 标准化为 NormalizedMotion；
    3. 使用 MotionConverter 批量转换为 Habitat SMPL-X PKL 格式。

运行方式：
    python -m ea_avs_mvp_v7.scripts.convert_motions [--action fall_related]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.human.humanoid_agent import resolve_humanoid_urdf_path
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_motions")


def main():
    parser = argparse.ArgumentParser(description="Batch Convert AMASS Motions to Habitat PKL")
    parser.add_argument("--action", type=str, default=None, help="Filter by target action class")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for PKLs")
    args = parser.parse_args()

    cfg = load_v7_config()
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)
    converter = MotionConverter(urdf_path)

    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    if not manifest_p.exists():
        logger.error("Manifest not found: %s", manifest_p)
        sys.exit(1)

    with open(manifest_p, "r", encoding="utf-8") as f:
        items = json.load(f)

    if args.action:
        items = [m for m in items if m.get("target_class") == args.action]

    out_base = Path(args.output_dir) if args.output_dir else get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    out_base.mkdir(parents=True, exist_ok=True)

    converted_files = []
    for item in items:
        rel_p = item.get("local_motion_path")
        if not rel_p:
            continue
        npz_p = from_relative_data_path(rel_p)
        if not npz_p.exists():
            continue

        target_class = item.get("target_class", "action")
        sid = item.get("babel_sid", "0")
        out_pkl = out_base / f"{target_class}_{sid}.pkl"

        norm_motion = load_amass_motion(
            npz_path=npz_p,
            start_frame=item.get("start_frame"),
            end_frame=item.get("end_frame"),
            metadata=item,
        )
        p = converter.convert_and_save(norm_motion, out_pkl)
        converted_files.append(p)

    print("\n" + "=" * 65)
    print(f"[Conversion Summary] Successfully converted {len(converted_files)} motions to {out_base}:")
    for p in converted_files:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
