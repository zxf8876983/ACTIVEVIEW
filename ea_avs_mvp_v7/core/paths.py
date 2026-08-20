"""
统一数据路径管理器 —— paths.py
==============================

功能：
    1. 解析 Git 源码仓库根目录与运行时数据根目录；
    2. 支持 ACTIVEVIEW_DATA_ROOT 环境变量动态覆盖；
    3. 提供 POSIX 规范的相对数据路径与绝对路径相互转换；
    4. 严禁将开发机绝对路径硬编码进源码或输出。
"""

import os
from pathlib import Path
from typing import Optional, Union

ENV_ACTIVEVIEW_DATA_ROOT = "ACTIVEVIEW_DATA_ROOT"


def get_repo_root() -> Path:
    """获取 ActiveView 源码仓库根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """获取 ActiveView 运行时数据根目录 (默认 ../../data/ActiveView)。"""
    env_path = os.environ.get(ENV_ACTIVEVIEW_DATA_ROOT)
    if env_path:
        root = Path(env_path).resolve()
    else:
        root = (get_repo_root().parent.parent / "data" / "ActiveView").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_assets_dir(subpath: str = "") -> Path:
    """获取 assets 外部数据资产目录。"""
    p = get_data_root() / "assets" / subpath
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_runs_dir(subpath: str = "") -> Path:
    """获取 runs 实验运行输出目录。"""
    p = get_data_root() / "runs" / subpath
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_datasets_dir(subpath: str = "") -> Path:
    """获取 datasets 数据集目录。"""
    p = get_data_root() / "datasets" / subpath
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_relative_data_path(abs_path: Union[str, Path]) -> str:
    """将绝对路径转换为相对于 data_root 的 POSIX 相对路径。"""
    p = Path(abs_path).resolve()
    root = get_data_root().resolve()
    try:
        rel = p.relative_to(root)
        return rel.as_posix()
    except ValueError:
        return p.as_posix()


def from_relative_data_path(rel_path: str) -> Path:
    """将相对于 data_root 的 POSIX 相对路径还原为绝对 Path。"""
    p = Path(rel_path)
    if p.is_absolute():
        return p.resolve()
    return (get_data_root() / p).resolve()
