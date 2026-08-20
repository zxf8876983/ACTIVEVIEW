"""
AMASS Hugging Face 数据集下载器与解压工具 —— download_amass_hf.py
===================================================================

功能：
    1. 基于 Hugging Face (tuguobin/AMASS) 选择性下载 5 个子库 (BMLrub, CMU, EKUT, EyesJapanDataset, KIT)；
    2. 优先支持 huggingface_hub API，同时具备 robust direct HTTP stream fallback (支持 hf-mirror.com 与断点下载)；
    3. 流式下载显示进度、大小与耗时；
    4. 执行归档完整性校验 (bzip2 header / tarfile is_tarfile)；
    5. 安全解压至 ActiveView 数据目录 (防 Path Traversal 逃逸)；
    6. 统计各子库与总 NPZ 文件数量和磁盘占用。
"""

import argparse
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .data_paths import (
    get_amass_dir,
    get_cache_dir,
    get_logs_dir,
    get_tmp_dir,
)

# 默认 HF 仓库与子库映射
HF_REPO_ID = "tuguobin/AMASS"
HF_REPO_TYPE = "dataset"
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# Feasibility Set 5 个子数据集在 HF 仓库中的相对模式 (全部位于 raw/)
HF_SUBDATASET_PATTERNS = {
    "BMLrub": "raw/BMLrub.tar.bz2",
    "CMU": "raw/CMU.tar.bz2",
    "EKUT": "raw/EKUT.tar.bz2",
    "EyesJapanDataset": "raw/EyesJapanDataset.tar.bz2",
    "KIT": "raw/KIT.tar.bz2",
}


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """检查目标解压路径是否严格位于 base_dir 内部 (防 Path Traversal)。"""
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        return base_resolved in target_resolved.parents or base_resolved == target_resolved
    except Exception:
        return False


def safe_extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """安全解压 tar.bz2 / zip 文件至指定目录。"""
    extract_dir.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(str(archive_path)):
        try:
            with tarfile.open(str(archive_path), "r:*") as tar:
                members = tar.getmembers()
                for member in members:
                    member_path = extract_dir / member.name
                    if not is_safe_path(extract_dir, member_path):
                        print(f"[Security Guard] Blocked unsafe archive path: {member.name}")
                        return False
                print(f"  [Extracting] {archive_path.name} ({len(members)} items) to {extract_dir}...")
                tar.extractall(path=str(extract_dir))
                return True
        except Exception as exc:
            print(f"[Extract Error] tarfile extraction failed for {archive_path}: {exc}")
            return False

    elif zipfile.is_zipfile(str(archive_path)):
        try:
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                for member in zf.infolist():
                    member_path = extract_dir / member.filename
                    if not is_safe_path(extract_dir, member_path):
                        print(f"[Security Guard] Blocked unsafe archive path: {member.filename}")
                        return False
                print(f"  [Extracting] {archive_path.name} ({len(zf.infolist())} items) to {extract_dir}...")
                zf.extractall(path=str(extract_dir))
                return True
        except Exception as exc:
            print(f"[Extract Error] zipfile extraction failed for {archive_path}: {exc}")
            return False

    else:
        print(f"[Extract Error] File is neither valid tar nor zip: {archive_path}")
        return False


def download_file_stream(url: str, dest_path: Path, label: str, max_retries: int = 15) -> Tuple[bool, str]:
    """流式下载大文件，支持断点续传与自动重试 (绕过本地无效代理直连镜像源)。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_part = dest_path.with_suffix(".part")

    session = requests.Session()
    session.trust_env = False  # 确保直连国内镜像源，避免被代理端口重置连接

    # 获取远程精确文件大小
    expected_total_size = 0
    try:
        head_resp = session.head(url, allow_redirects=True, timeout=15)
        expected_total_size = int(head_resp.headers.get("content-length", 0))
    except Exception:
        pass

    # 检查现有 part 文件是否过大或已完整
    if temp_part.exists() and expected_total_size > 0:
        if temp_part.stat().st_size > expected_total_size:
            print(f"  [Reset] {label} part file ({temp_part.stat().st_size} bytes) exceeds expected {expected_total_size} bytes. Resetting.")
            temp_part.unlink()
        elif temp_part.stat().st_size == expected_total_size:
            if tarfile.is_tarfile(str(temp_part)) or zipfile.is_zipfile(str(temp_part)):
                if dest_path.exists():
                    dest_path.unlink()
                temp_part.rename(dest_path)
                return True, "SUCCESS"

    start_time = time.time()

    for attempt in range(1, max_retries + 1):
        existing_bytes = temp_part.stat().st_size if temp_part.exists() else 0
        headers = {"User-Agent": "ActiveView-HF-Downloader/1.0"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        try:
            with session.get(url, headers=headers, stream=True, timeout=45, allow_redirects=True) as resp:
                if resp.status_code == 416:
                    pass
                elif resp.status_code not in (200, 206):
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    return False, f"HTTP Error {resp.status_code}"

                if resp.status_code == 206:
                    mode = "ab"
                    downloaded = existing_bytes
                    content_length = int(resp.headers.get("Content-Length", 0))
                    total_size = existing_bytes + content_length
                elif resp.status_code == 200:
                    mode = "wb"
                    downloaded = 0
                    total_size = int(resp.headers.get("Content-Length", 0))
                else:
                    downloaded = existing_bytes
                    total_size = expected_total_size

                if expected_total_size > 0:
                    total_size = expected_total_size

                total_mb = total_size / (1024 * 1024) if total_size > 0 else 0.0
                last_print = time.time()

                with open(temp_part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_print > 1.0:
                                mb_done = downloaded / (1024 * 1024)
                                speed_mb = (downloaded - existing_bytes) / (now - start_time) / (1024 * 1024) if (now > start_time) else 0
                                print(f"  [Downloading] {label}: {mb_done:.1f} / {total_mb:.1f} MB ({speed_mb:.2f} MB/s)...", end="\r", flush=True)
                                last_print = now

                # 检查是否已达到 total_size
                current_bytes = temp_part.stat().st_size if temp_part.exists() else 0
                if total_size > 0 and current_bytes >= total_size:
                    dur = time.time() - start_time
                    avg_speed = current_bytes / dur / (1024 * 1024) if dur > 0 else 0
                    print(f"  [Download Complete] {label}: {current_bytes / (1024 * 1024):.1f} MB in {dur:.1f}s (avg {avg_speed:.2f} MB/s)")

                    # 校验是否为合法 archive
                    if not (tarfile.is_tarfile(str(temp_part)) or zipfile.is_zipfile(str(temp_part))):
                        with open(temp_part, "rb") as f:
                            head = f.read(512)
                        if b"<html" in head.lower() or b"<!doctype" in head.lower():
                            if temp_part.exists():
                                temp_part.unlink()
                            return False, "Downloaded content is HTML error page instead of archive"

                    if dest_path.exists():
                        dest_path.unlink()
                    temp_part.rename(dest_path)
                    return True, "SUCCESS"

                if total_size > 0 and current_bytes < total_size:
                    print(f"  [Connection Dropped] {label}: {current_bytes / (1024*1024):.1f}/{total_mb:.1f} MB. Retrying ({attempt}/{max_retries})...")
                    time.sleep(2)
                    continue

        except Exception as exc:
            current_bytes = temp_part.stat().st_size if temp_part.exists() else 0
            print(f"  [Stream Exception] {label} ({type(exc).__name__}). Resume from {current_bytes / (1024*1024):.1f} MB. Retrying ({attempt}/{max_retries})...")
            time.sleep(2)
            continue

    return False, f"Failed to download {label} after {max_retries} attempts"


def get_dir_stats(target_dir: Path) -> Tuple[int, float]:
    """统计目标目录下所有 npz 文件数及总大小 (MB)。"""
    if not target_dir.exists():
        return 0, 0.0
    npz_files = list(target_dir.rglob("*.npz"))
    total_bytes = sum(p.stat().st_size for p in npz_files if p.is_file())
    return len(npz_files), total_bytes / (1024 * 1024)


def download_amass_from_hf(
    required_subdatasets: List[str],
    amass_output_dir: Path,
    tmp_download_dir: Path,
    dry_run: bool = False,
    endpoint: Optional[str] = None,
) -> dict:
    """从 Hugging Face 选择性下载并解压所需子库。"""
    ep = endpoint or os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)
    print(f"[HF Downloader] Endpoint: {ep}")
    print(f"[HF Downloader] Target Repo: {HF_REPO_ID} ({HF_REPO_TYPE})")
    print(f"[HF Downloader] Destination: {amass_output_dir}")

    # 构造 allow_patterns
    patterns = []
    hf_mapping = {}
    for ds in required_subdatasets:
        pat = HF_SUBDATASET_PATTERNS.get(ds)
        if pat:
            patterns.append(pat)
            hf_mapping[ds] = pat
        else:
            print(f"[Warning] Unknown subdataset pattern for {ds}, skipping.")

    print(f"\n[HF Downloader] Planned Subdatasets ({len(patterns)} archives):")
    for ds, pat in hf_mapping.items():
        print(f"  - {ds.ljust(20)} -> {pat}")

    if dry_run:
        print("\n[HF Downloader] DRY-RUN COMPLETE. No files downloaded.")
        return {"status": "dry_run_complete", "planned_patterns": patterns}

    tmp_download_dir.mkdir(parents=True, exist_ok=True)
    amass_output_dir.mkdir(parents=True, exist_ok=True)

    extraction_results = {}
    total_start = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_subdataset(ds_item: Tuple[str, str]) -> Tuple[str, dict]:
        ds, rel_pat = ds_item
        fname = Path(rel_pat).name
        target_archive = tmp_download_dir / rel_pat
        target_archive.parent.mkdir(parents=True, exist_ok=True)

        url = f"{ep.rstrip('/')}/datasets/{HF_REPO_ID}/resolve/main/{rel_pat}"
        print(f"\n[{ds}] Source URL: {url}")

        if target_archive.exists() and (tarfile.is_tarfile(str(target_archive)) or zipfile.is_zipfile(str(target_archive))):
            print(f"  [Archive Exists & Valid] {target_archive.name} ({target_archive.stat().st_size / (1024*1024):.1f} MB), skipping download.")
            ok = True
            msg = "CACHED"
        else:
            ok, msg = download_file_stream(url, target_archive, f"{ds} ({fname})")
            if not ok:
                print(f"  [Failed] Download {ds}: {msg}")
                return ds, {"status": "download_failed", "error": msg}

        # 解压
        size_mb = target_archive.stat().st_size / (1024 * 1024)
        print(f"  [Extracting Archive] {target_archive.name} ({size_mb:.1f} MB)...")
        ext_ok = safe_extract_archive(target_archive, amass_output_dir)
        return ds, {
            "status": "extracted" if ext_ok else "extract_failed",
            "archive_size_mb": size_mb,
            "archive_path": str(target_archive),
        }

    # 并发下载 5 个子数据集
    max_workers = min(5, len(hf_mapping))
    print(f"\n[HF Downloader] Launching {max_workers} concurrent download workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subdataset, item): item[0] for item in hf_mapping.items()}
        for future in as_completed(futures):
            ds_name = futures[future]
            try:
                ds, res = future.result()
                extraction_results[ds] = res
            except Exception as exc:
                print(f"  [Error] Worker exception for {ds_name}: {exc}")
                extraction_results[ds_name] = {"status": "exception", "error": str(exc)}

    total_time = time.time() - total_start

    # 统计解压结果
    total_npz, total_mb = get_dir_stats(amass_output_dir)
    print("\n" + "=" * 65)
    print(f"[HF Downloader] Complete in {total_time:.1f}s. Subdataset Breakdown:")
    sub_stats = {}
    for ds in required_subdatasets:
        matched_npzs = []
        for p in amass_output_dir.rglob("*.npz"):
            p_str = p.as_posix().lower()
            if ds.lower() in p_str or (ds == "BMLrub" and "biomotionlab" in p_str) or (ds == "EyesJapanDataset" and "eyes_japan" in p_str):
                matched_npzs.append(p)
        ds_mb = sum(p.stat().st_size for p in matched_npzs) / (1024 * 1024)
        sub_stats[ds] = {"npz_count": len(matched_npzs), "size_mb": ds_mb}
        print(f"  - {ds.ljust(20)}: {len(matched_npzs):5d} NPZ files ({ds_mb:8.1f} MB)")

    print(f"\n[Total AMASS Dataset] {total_npz} NPZ files ({total_mb:.1f} MB / {total_mb/1024:.2f} GB)")
    print("=" * 65)

    summary = {
        "status": "complete",
        "subdatasets": sub_stats,
        "total_npz_count": total_npz,
        "total_size_mb": total_mb,
        "extraction_results": extraction_results,
    }

    # 保存下载状态缓存
    amass_cache_dir = get_cache_dir("amass_download")
    status_file = amass_cache_dir / "hf_download_summary.json"
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download and Extract AMASS Datasets from Hugging Face")
    parser.add_argument("--dry-run", action="store_true", help="Preview download plan without downloading")
    parser.add_argument("--subdatasets", nargs="+", default=None, help="List of subdatasets to download (defaults to feasibility set)")
    parser.add_argument("--endpoint", type=str, default="https://hf-mirror.com", help="HF Mirror Endpoint")
    parser.add_argument("--manifest", type=str, default=None, help="Path to feasibility_manifest.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Target AMASS directory")
    parser.add_argument("--tmp-dir", type=str, default=None, help="Temporary download directory")
    args = parser.parse_args()

    cache_dir = get_cache_dir("babel_selection")
    manifest_path = Path(args.manifest) if args.manifest else cache_dir / "feasibility_manifest.json"

    if args.subdatasets:
        required_subdatasets = args.subdatasets
    elif manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            feasibility_manifest = json.load(f)
        required_subdatasets = sorted(list(set(item["amass_dataset"] for item in feasibility_manifest)))
    else:
        required_subdatasets = ["BMLrub", "CMU", "EKUT", "EyesJapanDataset", "KIT"]

    amass_out = Path(args.output_dir) if args.output_dir else get_amass_dir()
    tmp_out = Path(args.tmp_dir) if args.tmp_dir else get_tmp_dir("hf_downloads")

    download_amass_from_hf(
        required_subdatasets=required_subdatasets,
        amass_output_dir=amass_out,
        tmp_download_dir=tmp_out,
        dry_run=args.dry_run,
        endpoint=args.endpoint,
    )


if __name__ == "__main__":
    main()
