"""
AMASS 子数据集下载器与完整性验证工具 —— download_amass_required.py
===================================================================

功能：
    1. 从环境变量 AMASS_EMAIL / AMASS_PASSWORD 安全读取凭据 (绝不泄露与硬编码)；
    2. 基于 requests.Session 模拟正常网页登录流程并解析 Downloads 页面合法资源；
    3. 支持 --dry-run 预览与 --execute 正式下载；
    4. 执行下载完整性校验 (Content-Type, 文件大小, Archive 完整性)；
    5. 安全解压至 ActiveView 数据目录 (防 Path Traversal 逃逸)；
    6. 登录或下载失败时自动生成 manual_download_required.md / json 人工指引。
"""

import argparse
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .data_paths import (
    get_amass_dir,
    get_cache_dir,
    get_logs_dir,
    get_tmp_dir,
)

AMASS_BASE_URL = "https://amass.is.tue.mpg.de"
AMASS_LOGIN_URL = "https://amass.is.tue.mpg.de/login.php"
AMASS_DOWNLOAD_URL = "https://amass.is.tue.mpg.de/download.php"

# AMASS 子数据集在 BABEL feat_p 第一级名称与 AMASS 网站资源包的名称映射
AMASS_NAME_ALIASES = {
    "cmu": ["cmu", "cmu_mocap"],
    "kit": ["kit", "kit_mocap"],
    "eyesjapandataset": ["eyes_japan_dataset", "eyes_japan", "eyesjapandataset"],
    "bmlrub": ["biomotionlab_ntroje", "bmlrub", "bml_rub"],
    "bmlmovi": ["bmlmovi", "bml_movi"],
    "ekut": ["ekut", "ekut_mocap"],
    "mpihdm05": ["mpi_hdm05", "mpihdm05", "hdm05"],
    "accad": ["accad", "accad_mocap"],
    "sfumocap": ["sfu", "sfu_mocap"],
    "transitiondata": ["transitions", "transitions_mocap"],
    "dlsynth": ["dlsynth"],
    "totalcapture": ["totalcapture", "total_capture"],
    "humaneva": ["humaneva"],
    "mosh": ["mosh"],
}


def get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """从当前进程环境变量中安全获取 AMASS 凭据。"""
    email = os.environ.get("AMASS_EMAIL")
    password = os.environ.get("AMASS_PASSWORD")
    if email:
        email = email.strip()
    if password:
        password = password.strip()
    return email, password


def create_session() -> requests.Session:
    """创建并配置带有标准 Browser Headers 的请求 Session。"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return session


def amass_login(session: requests.Session, email: str, password: str) -> Tuple[bool, str]:
    """执行 AMASS 网页端登录认证流程。"""
    try:
        # 1. GET 登录页
        resp = session.get(AMASS_LOGIN_URL, timeout=20)
        if resp.status_code != 200:
            # 尝试直接访问 base
            resp = session.get(AMASS_BASE_URL, timeout=20)
            if resp.status_code != 200:
                return False, f"HTTP Error fetching login page: {resp.status_code}"

        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form")
        action_url = resp.url
        if form and form.get("action"):
            action_url = urljoin(resp.url, form.get("action"))

        # 提取 CSRF token 或所有 hidden / submit 字段
        payload = {}
        if form:
            for input_tag in form.find_all("input"):
                name = input_tag.get("name")
                val = input_tag.get("value", "")
                if name:
                    payload[name] = val

        # 填入账户密码 (AMASS 使用 username / password)
        email_key = "username"
        pass_key = "password"
        if form:
            for input_tag in form.find_all("input"):
                n = input_tag.get("name", "").lower()
                t = input_tag.get("type", "").lower()
                if "user" in n or "email" in n or t == "email":
                    email_key = input_tag.get("name")
                elif "pass" in n or t == "password":
                    pass_key = input_tag.get("name")

        payload[email_key] = email
        payload[pass_key] = password
        if "commit" not in payload:
            payload["commit"] = "Log in"

        session.headers.update({"Referer": resp.url})

        # 2. POST 登录
        post_resp = session.post(action_url, data=payload, timeout=25, allow_redirects=True)
        if post_resp.status_code not in (200, 302):
            return False, f"POST login returned status {post_resp.status_code}"

        # 3. 验证登录状态
        # 检查是否成功进入用户态或可以访问 downloads 页面
        check_resp = session.get(AMASS_DOWNLOAD_URL, timeout=20)
        text_lower = check_resp.text.lower()

        if "logout" in text_lower or "downloads" in text_lower or "sign out" in text_lower:
            # 确认在 downloads 页面没有被重定向回 login
            if "login.html" not in check_resp.url and "login" not in check_resp.url.split("/")[-1]:
                return True, "LOGIN_OK"

        if "invalid" in text_lower or "incorrect" in text_lower or "unauthorized" in text_lower:
            return False, "Invalid AMASS credentials"

        # 若 downloads 页面包含数据下载链接则说明登录成功
        if ".tar.bz2" in text_lower or ".zip" in text_lower or "download" in text_lower:
            return True, "LOGIN_OK"

        return False, "Login state could not be verified from download page"

    except Exception as exc:
        return False, f"Exception during login: {type(exc).__name__}: {str(exc)}"


def parse_downloadable_resources(session: requests.Session) -> Dict[str, Dict[str, str]]:
    """从 Downloads 页面解析所有可下载的 AMASS 子库链接与文件名。"""
    try:
        resp = session.get(AMASS_DOWNLOAD_URL, timeout=25)
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        resources = {}

        # 1. 扫描 openModalLicense 中的真实下载接口链接 (AMASS 2024+ 官方采用 openModalLicense 弹窗下载)
        modal_urls = re.findall(r"openModalLicense\(['\"]([^'\"]+)['\"]", resp.text)
        for raw_u in modal_urls:
            u = html.unescape(raw_u)
            fname = u.split("sfile=")[-1].split("/")[-1].split("&")[0] if "sfile=" in u else u.split("/")[-1].split("?")[0]
            clean_name = re.sub(r"[^a-zA-Z0-9]", "_", fname.lower())
            is_smplh = "smplh" in u.lower()
            is_smplx_neutral = "smplx/neutral" in u.lower()
            tag = "smplh" if is_smplh else ("smplx_neutral" if is_smplx_neutral else "other")
            key = f"{clean_name}_{tag}"
            resources[key] = {
                "url": u,
                "filename": fname,
                "link_text": f"Modal ({tag}) {fname}",
                "format": tag,
            }

        # 2. 扫描所有 a 标签和 download 按钮
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            full_url = urljoin(AMASS_BASE_URL, href)

            # 匹配 tar.bz2, zip, tar.gz
            if any(ext in href.lower() for ext in [".tar.bz2", ".tar.gz", ".zip"]) and "license" not in href.lower():
                filename = href.split("/")[-1].split("?")[0]
                clean_name = re.sub(r"[^a-zA-Z0-9]", "_", filename.lower())
                resources[clean_name] = {
                    "url": full_url,
                    "filename": filename,
                    "link_text": text,
                    "format": "direct_link",
                }

        # 3. 扫描所有 form
        for form in soup.find_all("form"):
            action = form.get("action", "")
            if any(ext in action.lower() for ext in [".tar.bz2", ".tar.gz", ".zip"]) and "license" not in action.lower():
                full_url = urljoin(AMASS_BASE_URL, action)
                filename = action.split("/")[-1].split("?")[0]
                clean_name = re.sub(r"[^a-zA-Z0-9]", "_", filename.lower())
                resources[clean_name] = {
                    "url": full_url,
                    "filename": filename,
                    "link_text": form.get_text(strip=True),
                    "format": "form_action",
                }

        return resources
    except Exception as exc:
        print(f"[Warning] Failed to parse downloads page: {exc}")
        return {}


def match_subdataset_to_resource(
    subdataset_name: str,
    available_resources: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """将 BABEL 的 AMASS 子数据集名称匹配到 Downloads 页面资源。"""
    s_clean = subdataset_name.lower().replace("_", "").replace("-", "")
    aliases = AMASS_NAME_ALIASES.get(s_clean, [s_clean])

    # 候选匹配列表 (排除 mp4/renders 视频包)
    matches = []
    for res_key, res_info in available_resources.items():
        fname = res_info["filename"].lower()
        if "render" in fname:
            continue
        for alias in aliases:
            a_clean = alias.lower().replace("_", "").replace("-", "")
            if a_clean in fname.replace("_", "").replace("-", ""):
                matches.append(res_info)
                break

    if not matches:
        return None

    # 优先级打分：优先 smplh (BABEL 原生)，其次 smplx/neutral
    def score_res(r):
        u = r.get("url", "").lower()
        if "smplh" in u:
            return 0
        elif "smplx/neutral" in u:
            return 1
        elif "smplx" in u:
            return 2
        return 3

    matches.sort(key=score_res)
    return matches[0]


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
                for member in tar.getmembers():
                    member_path = extract_dir / member.name
                    if not is_safe_path(extract_dir, member_path):
                        print(f"[Security Guard] Blocked unsafe archive path: {member.name}")
                        return False
                tar.extractall(path=str(extract_dir))
                return True
        except Exception as exc:
            print(f"[Extract Error] tarfile extraction failed: {exc}")
            return False

    elif zipfile.is_zipfile(str(archive_path)):
        try:
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                for member in zf.infolist():
                    member_path = extract_dir / member.filename
                    if not is_safe_path(extract_dir, member_path):
                        print(f"[Security Guard] Blocked unsafe archive path: {member.filename}")
                        return False
                zf.extractall(path=str(extract_dir))
                return True
        except Exception as exc:
            print(f"[Extract Error] zipfile extraction failed: {exc}")
            return False

    else:
        print(f"[Extract Error] File is neither valid tar nor zip: {archive_path}")
        return False


def download_file(
    session: requests.Session,
    url: str,
    dest_path: Path,
) -> Tuple[bool, str]:
    """流式下载大文件并校验 Content-Type 与大小。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download = dest_path.with_suffix(".part")

    try:
        with session.get(url, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                return False, f"HTTP Error {resp.status_code}"

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                return False, "Server returned HTML instead of archive (likely session expired or unauthorized)"

            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()

            with open(temp_download, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if total_size > 0 and downloaded < total_size:
                if temp_download.exists():
                    temp_download.unlink()
                return False, f"Incomplete download: {downloaded}/{total_size} bytes"

            # 校验是否为合法 archive
            if not (tarfile.is_tarfile(str(temp_download)) or zipfile.is_zipfile(str(temp_download))):
                # 检查文件开头是否为 HTML 错误页面
                with open(temp_download, "rb") as f:
                    head = f.read(512)
                if b"<html" in head.lower() or b"<!doctype" in head.lower():
                    if temp_download.exists():
                        temp_download.unlink()
                    return False, "Downloaded file contains HTML content, archive corrupted"

            if dest_path.exists():
                dest_path.unlink()
            temp_download.rename(dest_path)
            dur = time.time() - start_time
            print(f"  [Download Complete] {dest_path.name} ({downloaded / (1024*1024):.1f} MB in {dur:.1f}s)")
            return True, "SUCCESS"

    except Exception as exc:
        if temp_download.exists():
            temp_download.unlink()
        return False, f"Download exception: {type(exc).__name__}: {str(exc)}"


def generate_manual_download_manifest(
    required_subdatasets: List[str],
    feasibility_manifest: List[dict],
    reason: str,
    output_cache_dir: Path,
):
    """生成详细的 AMASS 手动下载指引文档与清单。"""
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "reason": reason,
        "amass_website": AMASS_BASE_URL,
        "amass_download_page": AMASS_DOWNLOAD_URL,
        "target_extraction_dir": "datasets/amass/",
        "required_subdatasets": required_subdatasets,
        "feasibility_sequences": [
            {
                "target_class": item["target_class"],
                "babel_sid": item["babel_sid"],
                "proc_label": item["proc_label"],
                "feat_p": item["feat_p"],
                "amass_dataset": item["amass_dataset"],
                "segment_duration": item["segment_duration"],
            }
            for item in feasibility_manifest
        ],
    }

    json_path = output_cache_dir / "manual_download_required.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    md_path = output_cache_dir / "manual_download_required.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AMASS Manual Download Instructions\n\n")
        f.write(f"> **Notice**: Automated download could not be completed. Reason: `{reason}`.\n")
        f.write("> Please follow the instructions below to manually obtain the required AMASS motion data.\n\n")
        f.write("---\n\n")
        f.write("## 1. Required AMASS Subdatasets (Feasibility Set)\n\n")
        f.write("The current v7.0 motion feasibility set requires the following AMASS subdatasets:\n\n")
        for ds in sorted(required_subdatasets):
            seqs = [item for item in feasibility_manifest if item["amass_dataset"] == ds]
            f.write(f"- **`{ds}`** ({len(seqs)} sequences needed)\n")
            for s in seqs:
                f.write(f"  - `[{s['target_class']}]` {s['proc_label']} (SID: {s['babel_sid']}, `{s['feat_p']}`)\n")
        f.write("\n---\n\n")
        f.write("## 2. Step-by-Step Manual Download Guide\n\n")
        f.write("1. Log in to official AMASS website: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)\n")
        f.write("2. Navigate to the **Downloads** section (SMPL+H GMM or SMPL-X format)\n")
        f.write("3. Download the archive files corresponding to the required subdatasets above (e.g. `CMU.tar.bz2`, `KIT.tar.bz2`, `Eyes_Japan_Dataset.tar.bz2`, `BioMotionLab_NTroje.tar.bz2`, `EKUT.tar.bz2`)\n")
        f.write("4. Extract the downloaded archives into the ActiveView data directory:\n")
        f.write("   ```bash\n")
        f.write("   # Target destination directory:\n")
        f.write("   mkdir -p /home/zxf/WorkSpace/code/data/ActiveView/datasets/amass/\n")
        f.write("   tar -xvjf <SUBDATASET>.tar.bz2 -C /home/zxf/WorkSpace/code/data/ActiveView/datasets/amass/\n")
        f.write("   ```\n")
        f.write("5. Run the indexing script to build the local file mapping:\n")
        f.write("   ```bash\n")
        f.write("   python -m tools.motion_assets.index_amass_files\n")
        f.write("   ```\n")

    print(f"[Manual Download Manifest] Generated: {json_path}")
    print(f"[Manual Download Manifest] Generated: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="AMASS Dataset Downloader & Integrity Verifier")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run dry run only (login and resource mapping, no large download)")
    parser.add_argument("--execute", action="store_true", help="Execute actual download and extraction")
    parser.add_argument("--manifest", type=str, default=None, help="Path to feasibility_manifest.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output extract directory for AMASS")
    parser.add_argument("--tmp-dir", type=str, default=None, help="Temporary download directory")
    args = parser.parse_args()

    is_dry_run = not args.execute

    cache_dir = get_cache_dir("babel_selection")
    amass_cache_dir = get_cache_dir("amass_download")
    amass_out_dir = Path(args.output_dir) if args.output_dir else get_amass_dir()
    tmp_dl_dir = Path(args.tmp_dir) if args.tmp_dir else get_tmp_dir("downloads")

    manifest_path = Path(args.manifest) if args.manifest else cache_dir / "feasibility_manifest.json"
    if not manifest_path.exists():
        print(f"[Error] Feasibility manifest not found at {manifest_path}. Please run select_babel_elderly_actions first.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        feasibility_manifest = json.load(f)

    required_subdatasets = sorted(list(set(item["amass_dataset"] for item in feasibility_manifest)))
    print(f"[AMASS Downloader] Feasibility subdatasets required: {required_subdatasets}")

    email, password = get_credentials()
    has_email = "yes" if bool(email) else "no"
    has_pass = "yes" if bool(password) else "no"
    print(f"[AMASS Downloader] AMASS email configured: {has_email}")
    print(f"[AMASS Downloader] AMASS password configured: {has_pass}")

    if not email or not password:
        print("[AMASS Downloader] AMASS credentials are not available in environment variables.")
        print("[AMASS Downloader] Please export AMASS_EMAIL and AMASS_PASSWORD to enable automated download.")
        generate_manual_download_manifest(
            required_subdatasets,
            feasibility_manifest,
            "credentials_missing",
            amass_cache_dir,
        )
        return

    session = create_session()
    print("[AMASS Downloader] Attempting authentication with AMASS portal...")
    login_ok, login_msg = amass_login(session, email, password)

    if not login_ok:
        print(f"[AMASS Downloader] Authentication failed: {login_msg}")
        generate_manual_download_manifest(
            required_subdatasets,
            feasibility_manifest,
            f"login_failed: {login_msg}",
            amass_cache_dir,
        )
        return

    print("[AMASS Downloader] Authentication successful (LOGIN_OK)")
    print("[AMASS Downloader] Fetching and parsing Downloads catalog...")
    resources = parse_downloadable_resources(session)
    print(f"[AMASS Downloader] Discovered {len(resources)} downloadable resources on portal")

    planned_downloads = {}
    missing_subdatasets = []

    for subds in required_subdatasets:
        matched = match_subdataset_to_resource(subds, resources)
        if matched:
            planned_downloads[subds] = matched
        else:
            missing_subdatasets.append(subds)

    print("\n[AMASS Downloader] Resource Mapping Plan:")
    for subds, res in planned_downloads.items():
        print(f"  - {subds.ljust(20)} -> {res['filename']} ({res['url']})")
    if missing_subdatasets:
        print(f"  [Notice] Subdatasets without direct automatic match: {missing_subdatasets}")

    if is_dry_run:
        print("\n[AMASS Downloader] DRY-RUN COMPLETE: Login verified, resource mapping planned.")
        print("[AMASS Downloader] To execute downloads, pass --execute flag.")
        return

    # 3. 正式下载执行
    print("\n[AMASS Downloader] Starting download and extraction...")
    download_results = {}

    for subds, res in planned_downloads.items():
        url = res["url"]
        fname = res["filename"]
        target_archive = tmp_dl_dir / fname

        print(f"\n[Downloading] {subds} from {url}...")
        ok, msg = download_file(session, url, target_archive)
        if not ok:
            print(f"[Failed] {subds}: {msg}")
            download_results[subds] = {"status": "download_failed", "error": msg}
            continue

        print(f"[Extracting] {fname} to {amass_out_dir}...")
        ext_ok = safe_extract_archive(target_archive, amass_out_dir)
        download_results[subds] = {
            "status": "success" if ext_ok else "extract_failed",
            "archive_path": str(target_archive),
            "error": None if ext_ok else "extraction_failed",
        }

    # 结果持久化
    status_log = amass_cache_dir / "download_status.json"
    with open(status_log, "w", encoding="utf-8") as f:
        json.dump(download_results, f, indent=2, ensure_ascii=False)
    print(f"\n[AMASS Downloader] Download status log saved to {status_log}")


if __name__ == "__main__":
    main()
