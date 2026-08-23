"""
AMASS 文件索引与 Habitat Schema 兼容性检查工具 —— index_amass_files.py
========================================================================

功能：
    1. 扫描本地 ActiveView 运行时数据目录下解压的 AMASS npz 文件；
    2. 建立 BABEL feat_p 与本地真实文件的鲁棒映射 (消除大小写、前缀下划线、poses/stageii 后缀差异及多级目录前缀)；
    3. 检查 NPZ 数据 Schema 兼容性 (同时支持 explicit_root_orient 与标准 standard_amass poses[:, :3])；
    4. 将 BABEL 时间戳 start_t/end_t 转换为动作 frame index (start_frame/end_frame)；
    5. 生成标准化 motion_asset_manifest.json (相对 data_root 路径，严禁硬编码绝对路径)；
    6. 严格机器可读退出判定 (17/17 全部通过返回 exit code 0，存在缺失返回 exit code 1)。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_paths import (
    get_amass_dir,
    get_assets_dir,
    get_cache_dir,
    get_data_root,
    to_relative_data_path,
)


def normalize_path_key(path_str: str) -> str:
    """标准化路径字符串以消除微小命名格式与后缀差异。"""
    s = path_str.replace("\\", "/").strip().lower()
    s = s.replace("_poses.npz", ".npz").replace("_stageii.npz", ".npz")
    s = s.replace("-", "_")
    # 去除多余的连续斜杠
    while "//" in s:
        s = s.replace("//", "/")
    return s


def build_amass_disk_index(amass_root: Path) -> Dict[str, Path]:
    """递归遍历 amass_root 下的所有 npz 文件，构建多级标准化路径索引字典。"""
    index = {}
    if not amass_root.exists():
        return index

    for p in amass_root.rglob("*.npz"):
        if not p.is_file():
            continue
        rel = p.relative_to(amass_root).as_posix()
        norm_rel = normalize_path_key(rel)
        index[norm_rel] = p

        # 去除顶层子集前缀后的相对路径 (如 BioMotionLab_NTroje/... 或 rub101/...)
        parts = rel.split("/")
        for i in range(1, len(parts)):
            sub_rel = normalize_path_key("/".join(parts[i:]))
            if sub_rel not in index:
                index[sub_rel] = p

        # 纯文件名 (不含后缀)
        stem_norm = normalize_path_key(p.name)
        if stem_norm not in index:
            index[stem_norm] = p

    return index


def find_matching_file(feat_p: str, disk_index: Dict[str, Path], amass_root: Path) -> Optional[Path]:
    """在磁盘索引中查找 BABEL feat_p 对应的真实本地 NPZ 文件。"""
    if not feat_p:
        return None

    # 1. 尝试直接拼接
    direct_path = (amass_root / feat_p).resolve()
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    norm_target = normalize_path_key(feat_p)

    # 2. 精确匹配标准化相对路径
    if norm_target in disk_index:
        return disk_index[norm_target]

    # 3. 截取 BABEL feat_p 的各个子路径匹配 (例如 CMU/CMU/15/15_04_poses.npz -> CMU/15/15_04.npz -> 15/15_04.npz)
    parts = feat_p.replace("\\", "/").split("/")
    for i in range(1, len(parts)):
        sub_key = normalize_path_key("/".join(parts[i:]))
        if sub_key in disk_index:
            return disk_index[sub_key]

    # 4. 模糊后向包含匹配
    for k, p in disk_index.items():
        if norm_target.endswith(k) or k.endswith(norm_target):
            return p

    # 5. 纯文件名匹配
    target_fname = normalize_path_key(Path(feat_p).name)
    if target_fname in disk_index:
        return disk_index[target_fname]

    return None


def inspect_npz_schema(npz_path: Path) -> dict:
    """检查单个 AMASS NPZ 文件的数据 schema，支持 explicit_root_orient 与 standard_amass。"""
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        has_trans = "trans" in keys
        has_root_orient = "root_orient" in keys
        has_poses = "poses" in keys

        # 获取帧率 (兼容各种 AMASS 帧率键名写法)
        fps = None
        for fps_key in ["mocap_frame_rate", "mocap_framerate", "frame_rate", "framerate", "fps"]:
            if fps_key in keys:
                try:
                    val = float(data[fps_key])
                    if val > 0:
                        fps = val
                        break
                except Exception:
                    pass

        poses_shape = list(data["poses"].shape) if has_poses else None
        num_frames = poses_shape[0] if (poses_shape and len(poses_shape) > 0) else 0

        # 判断 Schema 类型
        missing_fields = []
        if not has_trans:
            missing_fields.append("trans")
        if not has_poses:
            missing_fields.append("poses")
        if fps is None or fps <= 0:
            missing_fields.append("frame_rate")

        schema_compatible = False
        schema_type = "incompatible"
        root_orient_source = "none"

        if has_trans and has_poses and fps is not None and fps > 0:
            # 检查 poses 维度
            if len(poses_shape) >= 2 and poses_shape[1] >= 3:
                if has_root_orient:
                    schema_type = "explicit_root_orient"
                    root_orient_source = "root_orient"
                    schema_compatible = True
                else:
                    schema_type = "standard_amass"
                    root_orient_source = "poses[:, :3]"
                    schema_compatible = True
            else:
                missing_fields.append("poses_dim_less_than_3")

        return {
            "schema_compatible": schema_compatible,
            "schema_type": schema_type,
            "root_orient_source": root_orient_source,
            "missing_fields": missing_fields,
            "fps": fps if fps else 30.0,
            "num_frames": num_frames,
            "poses_shape": poses_shape,
            "keys": sorted(list(keys)),
        }

    except Exception as exc:
        return {
            "schema_compatible": False,
            "schema_type": "read_error",
            "root_orient_source": "none",
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


def index_and_validate_feasibility_manifest(
    feasibility_items: List[dict],
    amass_dir: Path,
    output_manifest_path: Path,
    output_compat_csv: Path,
    output_index_json: Path,
) -> Tuple[bool, dict]:
    """执行完整的本地 AMASS 索引、匹配与 Schema 验证。"""
    print(f"[AMASS Indexer] Scanning local AMASS files in {amass_dir}...")
    disk_index = build_amass_disk_index(amass_dir)
    print(f"[AMASS Indexer] Indexed {len(disk_index)} normalized path entries from disk")

    # 导出 disk index 缓存
    output_index_json.parent.mkdir(parents=True, exist_ok=True)
    serializable_index = {k: str(p) for k, p in disk_index.items()}
    with open(output_index_json, "w", encoding="utf-8") as f:
        json.dump(serializable_index, f, indent=2, ensure_ascii=False)

    compatibility_records = []
    final_manifest = []
    missing_items = []

    for item in feasibility_items:
        feat_p = item.get("feat_p", "")
        matched_path = find_matching_file(feat_p, disk_index, amass_dir)

        if matched_path and matched_path.exists():
            schema_info = inspect_npz_schema(matched_path)
            local_rel_path = to_relative_data_path(matched_path)
            fps = schema_info["fps"]
            num_frames = schema_info["num_frames"]
            start_frame, end_frame = compute_frame_range(item["start_t"], item["end_t"], fps, num_frames)

            entry = {
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "proc_label": item["proc_label"],
                "raw_label": item["raw_label"],
                "act_cat": item["act_cat"],
                "start_t": item["start_t"],
                "end_t": item["end_t"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "feat_p": feat_p,
                "amass_dataset": item["amass_dataset"],
                "local_motion_path": local_rel_path,
                "schema_type": schema_info["schema_type"],
                "root_orient_source": schema_info["root_orient_source"],
                "schema_compatible": schema_info["schema_compatible"],
                "needs_manual_review": item["needs_manual_review"],
            }
            final_manifest.append(entry)

            compat_rec = {
                "babel_sid": item["babel_sid"],
                "target_class": item["target_class"],
                "feat_p": feat_p,
                "file_found": True,
                "local_path": local_rel_path,
                "schema_compatible": schema_info["schema_compatible"],
                "schema_type": schema_info["schema_type"],
                "root_orient_source": schema_info["root_orient_source"],
                "missing_fields": ";".join(schema_info["missing_fields"]),
                "fps": fps,
                "num_frames": num_frames,
            }
            compatibility_records.append(compat_rec)

            if not schema_info["schema_compatible"]:
                missing_items.append({
                    "reason": "incompatible_schema",
                    "target_class": item["target_class"],
                    "babel_sid": item["babel_sid"],
                    "feat_p": feat_p,
                    "amass_dataset": item["amass_dataset"],
                    "expected_matching_path": str(matched_path),
                })
        else:
            # 文件缺失
            entry = {
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "proc_label": item["proc_label"],
                "raw_label": item["raw_label"],
                "act_cat": item["act_cat"],
                "start_t": item["start_t"],
                "end_t": item["end_t"],
                "start_frame": None,
                "end_frame": None,
                "fps": None,
                "feat_p": feat_p,
                "amass_dataset": item["amass_dataset"],
                "local_motion_path": None,
                "schema_type": "file_not_found",
                "root_orient_source": "none",
                "schema_compatible": False,
                "needs_manual_review": item["needs_manual_review"],
            }
            final_manifest.append(entry)

            compat_rec = {
                "babel_sid": item["babel_sid"],
                "target_class": item["target_class"],
                "feat_p": feat_p,
                "file_found": False,
                "local_path": "MISSING",
                "schema_compatible": False,
                "schema_type": "file_not_found",
                "root_orient_source": "none",
                "missing_fields": "file_not_found",
                "fps": None,
                "num_frames": 0,
            }
            compatibility_records.append(compat_rec)

            missing_items.append({
                "reason": "file_not_found",
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "feat_p": feat_p,
                "amass_dataset": item["amass_dataset"],
                "expected_matching_path": f"{amass_dir}/{feat_p}",
            })

    # 导出 compatibility CSV
    output_compat_csv.parent.mkdir(parents=True, exist_ok=True)
    if compatibility_records:
        with open(output_compat_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(compatibility_records[0].keys()))
            writer.writeheader()
            writer.writerows(compatibility_records)
    print(f"[AMASS Indexer] Compatibility report saved to {output_compat_csv}")

    # 导出 final motion_asset_manifest.json
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, ensure_ascii=False)
    print(f"[AMASS Indexer] Motion asset manifest saved to {output_manifest_path}")

    found_cnt = sum(1 for r in compatibility_records if r["file_found"])
    compat_cnt = sum(1 for r in compatibility_records if r["schema_compatible"])
    total_cnt = len(feasibility_items)

    std_amass_cnt = sum(1 for r in compatibility_records if r.get("schema_type") == "standard_amass")
    exp_orient_cnt = sum(1 for r in compatibility_records if r.get("schema_type") == "explicit_root_orient")
    incompat_cnt = total_cnt - compat_cnt

    print("\n" + "=" * 60)
    print(f"[AMASS Indexer] Validation Results: file_found = {found_cnt}/{total_cnt}, schema_compatible = {compat_cnt}/{total_cnt}")
    print(f"  - standard_amass count:        {std_amass_cnt}")
    print(f"  - explicit_root_orient count:  {exp_orient_cnt}")
    print(f"  - incompatible/missing count:  {incompat_cnt}")
    print("=" * 60)

    is_all_passed = (found_cnt == total_cnt) and (compat_cnt == total_cnt)
    stats = {
        "file_found_count": found_cnt,
        "schema_compatible_count": compat_cnt,
        "total_count": total_cnt,
        "standard_amass_count": std_amass_cnt,
        "explicit_root_orient_count": exp_orient_cnt,
        "incompatible_count": incompat_cnt,
        "missing_items": missing_items,
        "is_all_passed": is_all_passed,
    }

    if not is_all_passed:
        print("\n[INCOMPLETE] Missing or Incompatible Feasibility Motions:")
        for m in missing_items:
            print(f"  - [{m['target_class']}] sid={m['babel_sid']} | {m['amass_dataset']} | {m['feat_p']} (Reason: {m['reason']})")
        print("\n[Status] INCOMPLETE")
    else:
        print("\n[Status] PASS: All 17 feasibility motion files found and schema compatible!")

    return is_all_passed, stats


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
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        feasibility_items = json.load(f)

    assets_raw_dir = get_assets_dir("motions/raw")
    manifest_out = assets_raw_dir / "motion_asset_manifest.json"
    compat_csv = amass_cache_dir / "habitat_motion_compatibility.csv"
    index_json = amass_cache_dir / "amass_file_index.json"

    all_passed, stats = index_and_validate_feasibility_manifest(
        feasibility_items=feasibility_items,
        amass_dir=amass_dir,
        output_manifest_path=manifest_out,
        output_compat_csv=compat_csv,
        output_index_json=index_json,
    )

    if not all_passed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
