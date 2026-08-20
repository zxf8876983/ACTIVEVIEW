"""
Episode 数据结构与 Metadata 校验脚本 —— verify_episode_output.py
==============================================================

功能：
    1. 校验指定 Episode 输出目录或 runs/ 下各 Episode 文件的物理完整性；
    2. 检查 rgb/ 图像文件 (.png) 与 depth/ 深度文件 (.npy)；
    3. 严格验证 metadata.json 是否包含全部必需字段：
       scene_id, episode_id, motion_id, action_class, action_label, robot_pose, camera_pose, human_pose_gt, frames；
    4. 缺少任何字段或文件时直接报错终止。

运行方式：
    python -m ea_avs_mvp_v7.scripts.verify_episode_output [--episode-dir <path>]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_episode_output")

REQUIRED_TOP_LEVEL_FIELDS = [
    "scene_id",
    "episode_id",
    "motion_id",
    "action_class",
    "action_label",
    "robot_pose",
    "camera_pose",
    "human_pose_gt",
    "frames",
]

REQUIRED_FRAME_FIELDS = [
    "frame_id",
    "timestamp",
    "robot_pose",
    "camera_pose",
    "human_pose_gt",
    "rgb_path",
    "depth_path",
]


def verify_episode_dir(ep_dir: Path) -> dict:
    """校验单套 Episode 目录的完整性与元数据规范。"""
    if not ep_dir.exists() or not ep_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {ep_dir}")

    rgb_dir = ep_dir / "rgb"
    depth_dir = ep_dir / "depth"
    meta_path = ep_dir / "metadata.json"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing 'rgb/' directory in {ep_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing 'depth/' directory in {ep_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing 'metadata.json' in {ep_dir}")

    # 读取并验证 metadata.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in meta:
            raise KeyError(f"Missing required top-level field '{field}' in {meta_path}")

    frames = meta["frames"]
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"Empty or invalid 'frames' list in {meta_path}")

    rgb_files = list(rgb_dir.glob("*.png"))
    depth_files = list(depth_dir.glob("*.npy"))

    if len(rgb_files) != len(frames):
        raise ValueError(f"Mismatch: found {len(rgb_files)} RGB files but metadata has {len(frames)} frames")
    if len(depth_files) != len(frames):
        raise ValueError(f"Mismatch: found {len(depth_files)} Depth files but metadata has {len(frames)} frames")

    for f_idx, f_entry in enumerate(frames):
        for f_field in REQUIRED_FRAME_FIELDS:
            if f_field not in f_entry:
                raise KeyError(f"Missing frame field '{f_field}' in frame {f_idx} of {meta_path}")

        rgb_p = from_relative_data_path(f_entry["rgb_path"])
        depth_p = from_relative_data_path(f_entry["depth_path"])
        if not rgb_p.exists():
            raise FileNotFoundError(f"Referenced RGB file missing: {rgb_p}")
        if not depth_p.exists():
            raise FileNotFoundError(f"Referenced Depth file missing: {depth_p}")

    return {
        "episode_id": meta["episode_id"],
        "scene_id": meta["scene_id"],
        "motion_id": meta["motion_id"],
        "action_class": meta["action_class"],
        "action_label": meta["action_label"],
        "num_frames": len(frames),
        "rgb_count": len(rgb_files),
        "depth_count": len(depth_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Episode Dataset Structure and Metadata")
    parser.add_argument("--episode-dir", type=str, default=None, help="Path to episode directory")
    args = parser.parse_args()

    if args.episode_dir:
        ep_dir = Path(args.episode_dir).resolve()
    else:
        runs_dir = get_data_root() / "runs"
        cand_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
        if not cand_dirs:
            raise FileNotFoundError(f"No episode directories found in {runs_dir}")
        ep_dir = sorted(cand_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    logger.info("Verifying Episode: %s", ep_dir)
    res = verify_episode_dir(ep_dir)

    print("\n" + "=" * 65)
    print("[v7 Episode Output Verification]")
    print(f"  - Episode ID:         {res['episode_id']}")
    print(f"  - Scene ID:           {res['scene_id']}")
    print(f"  - Motion ID:          {res['motion_id']}")
    print(f"  - Action Class:       {res['action_class']}")
    print(f"  - Action Label:       {res['action_label']}")
    print(f"  - Valid Frame Count:  {res['num_frames']}")
    print(f"  - RGB Files:          {res['rgb_count']} PASS")
    print(f"  - Depth Files:        {res['depth_count']} PASS")
    print(f"  - Metadata Fields:    ALL REQUIRED FIELDS PRESENT")
    print("=" * 65)
    print("PASS: Episode Structure and Metadata Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
