"""
Motion Pipeline 独立验证脚本 —— verify_motion_pipeline.py
=========================================================

功能：
    1. 读取 AMASS 动作数据 (.npz) 并规范化为 NormalizedMotion；
    2. 使用 MotionConverter 转换为 Habitat KinematicHumanoid .pkl 格式；
    3. 校验 joints_array 形状 (N, 216)、transform_array 形状 (N, 4, 4)、四元数归一化与 FPS；
    4. 输出标准格式的 [v7 Motion Pipeline Verification] 报告。

运行方式：
    python -m ea_avs_mvp_v7.scripts.verify_motion_pipeline [--action fall_related]
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
from ea_avs_mvp_v7.motion.joint_mapping import (
    HABITAT_HUMANOID_QUAT_DIM,
    validate_habitat_motion_dict,
)
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_motion_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Verify AMASS to Habitat Motion Conversion Pipeline")
    parser.add_argument("--action", type=str, default="fall_related", help="Action class to verify (default: fall_related)")
    parser.add_argument("--npz-path", type=str, default=None, help="Explicit AMASS npz file path")
    args = parser.parse_args()

    cfg = load_v7_config()
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)
    converter = MotionConverter(urdf_path)

    # 1. 获取目标动作
    if args.npz_path:
        npz_p = Path(args.npz_path).resolve()
        if not npz_p.exists():
            raise FileNotFoundError(f"Specified npz file not found: {npz_p}")
        target_class = args.action
        sid = "custom"
        raw_label = "custom motion"
        item = {"local_motion_path": str(npz_p), "target_class": target_class, "proc_label": raw_label, "babel_sid": sid}
    else:
        manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
        if not manifest_p.exists():
            raise FileNotFoundError(f"Motion manifest not found: {manifest_p}")
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        item = next((m for m in manifest if m.get("target_class") == args.action), None)
        if item is None:
            raise ValueError(f"Action '{args.action}' not found in manifest!")
        npz_p = from_relative_data_path(item["local_motion_path"])
        target_class = item.get("target_class")
        sid = item.get("babel_sid")
        raw_label = item.get("proc_label", item.get("raw_label", "action"))

    motion_id = f"{target_class}_{sid}"
    logger.info("Verifying Motion Pipeline for [%s] -> %s", motion_id, npz_p)

    # 2. 加载与标准化
    norm_motion = load_amass_motion(
        npz_path=npz_p,
        start_frame=item.get("start_frame"),
        end_frame=item.get("end_frame"),
        metadata=item,
    )

    # 3. 转换并校验
    converted_data = converter.convert(norm_motion)
    stats = validate_habitat_motion_dict(converted_data)

    num_frames = stats["num_frames"]
    fps = stats["fps"]
    joints_shape = f"({num_frames}, {stats['joints_dim']})"
    transform_shape = f"{tuple(stats['transforms_shape'])}"
    joint_count = stats["joints_dim"] // 4

    print("\n" + "=" * 65)
    print("[v7 Motion Pipeline Verification]")
    print(f"  - Motion ID:          {motion_id}")
    print(f"  - Action Label:       {raw_label}")
    print(f"  - Frame Number:       {num_frames}")
    print(f"  - FPS:                {fps:.1f}")
    print(f"  - Quaternion Shape:   {joints_shape}")
    print(f"  - Transform Shape:    {transform_shape}")
    print(f"  - Joint Count:        {joint_count}")
    print(f"  - Status:             PASS")
    print("=" * 65)
    print("PASS: v7 Motion Pipeline Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
