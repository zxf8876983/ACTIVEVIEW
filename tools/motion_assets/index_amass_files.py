"""
AMASS 文件索引与 Habitat 兼容性检查工具 —— index_amass_files.py
===============================================================

功能：
    1. 扫描本地 ActiveView 数据目录下已解压的 AMASS npz 文件；
    2. 建立 BABEL feat_p 与本地真实文件的鲁棒映射 (消除大小写、前缀下划线、poses/stageii 后缀差异)；
    3. 检查 NPZ 的数据 Schema 是否兼容 Habitat MotionConverterSMPLX 所需字段；
    4. 将 BABEL 时间戳 start_t/end_t 转换为动作 frame index (start_frame/end_frame)；
    5. 生成最终的 motion_asset_manifest.json 元数据清单。
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_paths import (
    get_amass_dir,
    get_assets_dir,
    get_cache_dir,
    to_relative_data_path,
)


def build_amass_disk_index(amass_root: Path) -> Dict[str, Path]:
    """递归遍历 amass_root 下的所有 npz 文件，构建标准化路径索引字典。"""
    index = {}
    if not amass_root.exists():
        return index

    for p in amass_root.rglob("*.npz"):
        if not p.is_file():
            continue
        rel = p.relative_to(amass_root).as_posix()
        norm_key = normalize_path_key(rel)
        index[norm_key] = p
        # 也按文件名结尾索引
        filename = p.name.lower()
        if filename not in index:
            index[filename] = p

    return index


def normalize_path_key(path_str: str) -> str:
    """标准化路径字符串以消除微小命名格式差异。"""
    s = path_str.replace("\\", "/").strip().lower()
    s = s.replace("_poses.npz", ".npz").replace("_stageii.npz", ".npz")
    s = s.replace("-", "_")
    return s


def find_matching_file(feat_p: str, disk_index: Dict[str, Path], amass_root: Path) -> Optional[Path]:
    """在磁盘索引中查找 BABEL feat_p 对应的真实本地 NPZ 文件。"""
    if not feat_p:
        return None

    # 1. 尝试直接拼接
    direct_path = (amass_root / feat_p).resolve()
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    norm_target = normalize_path_key(feat_p)

    # 2. 精确匹配标准化 key
    if norm_target in disk_index:
        return disk_index[norm_target]

    # 3. 模糊后向包含匹配 (去掉最顶层子集名重复情况，如 CMU/CMU/...)
    for k, p in disk_index.items():
        if norm_target.endswith(k) or k.endswith(norm_target):
            return p

    # 4. 按文件名匹配
    target_fname = Path(feat_p).name.lower()
    if target_fname in disk_index:
        return disk_index[target_fname]

    return None


def inspect_npz_schema(npz_path: Path) -> dict:
    """检查单个 AMASS NPZ 文件的数据 schema 与字段完整性。"""
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        has_trans = "trans" in keys
        has_root_orient = "root_orient" in keys
        has_poses = "poses" in keys

        # 获取帧率
        fps = None
        if "mocap_frame_rate" in keys:
            fps = float(data["mocap_frame_rate"])
        elif "frame_rate" in keys:
            fps = float(data["frame_rate"])
        elif "fps" in keys:
            fps = float(data["fps"])

        poses_shape = list(data["poses"].shape) if has_poses else None
        num_frames = poses_shape[0] if (poses_shape and len(poses_shape) > 0) else 0

        # 判断是否满足 Habitat MotionConverterSMPLX 核心要求
        is_compatible = bool(has_trans and has_root_orient and has_poses and fps is not None and fps > 0)

        missing_fields = []
        if not has_trans:
            missing_fields.append("trans")
        if not has_root_orient:
            missing_fields.append("root_orient")
        if not has_poses:
            missing_fields.append("poses")
        if fps is None:
            missing_fields.append("frame_rate")

        return {
            "schema_compatible": is_compatible,
            "missing_fields": missing_fields,
            "fps": fps if fps else 30.0,
            "num_frames": num_frames,
            "poses_shape": poses_shape,
            "keys": sorted(list(keys)),
        }

    except Exception as exc:
        return {
            "schema_compatible": False,
            "missing_fields": [f"read_error:{type(exc).__name__}"],
            "fps": 30.0,
            "num_frames": 0,
            "poses_shape": None,
            "keys": [],
        }


def compute_frame_range(start_t: float, end_t: float, fps: float, num_frames: int) -> Tuple[int, int]:
    """根据开始/结束时间和 fps 计算对应帧索引并 clamp 到合法范围。"""
    if fps <= 0:
        fps = 30.0

    start_frame = max(0, int(round(start_t * fps)))
    if num_frames > 0:
        end_frame = min(num_frames - 1, int(round(end_t * fps)))
    else:
        end_frame = max(start_frame, int(round(end_t * fps)))

    if end_frame < start_frame:
        end_frame = start_frame

    return start_frame, end_frame


def main():
    parser = argparse.ArgumentParser(description="Index Local AMASS Files and Check Habitat Compatibility")
    parser.add_argument("--manifest", type=str, default=None, help="Path to feasibility_manifest.json")
    parser.add_argument("--amass-dir", type=str, default=None, help="Local AMASS dataset directory")
    args = parser.parse_args()

    cache_dir = get_cache_dir("babel_selection")
    amass_cache_dir = get_cache_dir("amass_download")
    amass_dir = Path(args.amass_dir) if args.amass_dir else get_amass_dir()
    manifest_path = Path(args.manifest) if args.manifest else cache_dir / "feasibility_manifest.json"

    if not manifest_path.exists():
        print(f"[Error] Feasibility manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        feasibility_items = json.load(f)

    print(f"[AMASS Indexer] Scanning local AMASS files in {amass_dir}...")
    disk_index = build_amass_disk_index(amass_dir)
    print(f"[AMASS Indexer] Found {len(disk_index)} local npz files in disk index")

    # 导出 disk index 缓存
    index_cache_path = amass_cache_dir / "amass_file_index.json"
    serializable_index = {k: str(p) for k, p in disk_index.items()}
    with open(index_cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable_index, f, indent=2, ensure_ascii=False)

    # 匹配与 Schema 检查
    compatibility_records = []
    final_manifest = []

    for item in feasibility_items:
        feat_p = item.get("feat_p", "")
        matched_path = find_matching_file(feat_p, disk_index, amass_dir)

        if matched_path and matched_path.exists():
            schema_info = inspect_npz_schema(matched_path)
            local_rel_path = to_relative_data_path(matched_path)
            fps = schema_info["fps"]
            num_frames = schema_info["num_frames"]
            start_frame, end_frame = compute_frame_range(item["start_t"], item["end_t"], fps, num_frames)

            final_manifest.append({
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "source_split": item["source_split"],
                "annotation_level": item["annotation_level"],
                "act_cat": item["act_cat"],
                "proc_label": item["proc_label"],
                "raw_label": item["raw_label"],
                "start_t": item["start_t"],
                "end_t": item["end_t"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "num_frames": num_frames,
                "feat_p": feat_p,
                "amass_dataset": item["amass_dataset"],
                "local_motion_path": local_rel_path,
                "schema_compatible": schema_info["schema_compatible"],
                "needs_manual_review": item["needs_manual_review"],
            })

            compatibility_records.append({
                "babel_sid": item["babel_sid"],
                "target_class": item["target_class"],
                "feat_p": feat_p,
                "file_found": True,
                "local_path": local_rel_path,
                "schema_compatible": schema_info["schema_compatible"],
                "missing_fields": ";".join(schema_info["missing_fields"]),
                "fps": fps,
                "num_frames": num_frames,
            })
        else:
            # 文件缺失
            final_manifest.append({
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "source_split": item["source_split"],
                "annotation_level": item["annotation_level"],
                "act_cat": item["act_cat"],
                "proc_label": item["proc_label"],
                "raw_label": item["raw_label"],
                "start_t": item["start_t"],
                "end_t": item["end_t"],
                "start_frame": None,
                "end_frame": None,
                "fps": None,
                "num_frames": 0,
                "feat_p": feat_p,
                "amass_dataset": item["amass_dataset"],
                "local_motion_path": None,
                "schema_compatible": False,
                "needs_manual_review": item["needs_manual_review"],
            })

            compatibility_records.append({
                "babel_sid": item["babel_sid"],
                "target_class": item["target_class"],
                "feat_p": feat_p,
                "file_found": False,
                "local_path": "MISSING",
                "schema_compatible": False,
                "missing_fields": "file_not_downloaded",
                "fps": None,
                "num_frames": 0,
            })

    # 导出 compatibility CSV
    compat_csv_path = amass_cache_dir / "habitat_motion_compatibility.csv"
    if compatibility_records:
        with open(compat_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(compatibility_records[0].keys()))
            writer.writeheader()
            writer.writerows(compatibility_records)
    print(f"[AMASS Indexer] Compatibility report saved to {compat_csv_path}")

    # 导出 final motion_asset_manifest.json
    assets_raw_dir = get_assets_dir("motions/raw")
    manifest_out_path = assets_raw_dir / "motion_asset_manifest.json"
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, ensure_ascii=False)
    print(f"[AMASS Indexer] Motion asset manifest saved to {manifest_out_path}")

    found_cnt = sum(1 for r in compatibility_records if r["file_found"])
    compat_cnt = sum(1 for r in compatibility_records if r["schema_compatible"])
    print(f"\n[AMASS Indexer] Summary: {found_cnt}/{len(feasibility_items)} files found, {compat_cnt}/{len(feasibility_items)} schema compatible.")


if __name__ == "__main__":
    main()
