"""
ActiveView 统一数据路径管理工具 —— data_paths.py
=================================================

功能：
    统一解析 ActiveView 项目数据根目录。
    支持：
        1. 环境变量 ACTIVEVIEW_DATA_ROOT 优先覆盖；
        2. 默认基于 Git 仓库根目录的相对路径 ../../data/ActiveView 解析。
    杜绝在代码或配置中硬编码机器特定的绝对路径。
"""

import os
from pathlib import Path
from typing import Union, Optional


def get_repo_root() -> Path:
    """获取 ActiveView 源码仓库根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """获取 ActiveView 外部运行时数据根目录。

    优先级：
        1. 环境变量 ACTIVEVIEW_DATA_ROOT（若设置且非空）
        2. 相对路径 (repo_root / "../../data/ActiveView").resolve()
    """
    env_root = os.environ.get("ACTIVEVIEW_DATA_ROOT")
    if env_root and env_root.strip():
        return Path(env_root.strip()).resolve()

    default_root = get_repo_root() / ".." / ".." / "data" / "ActiveView"
    return default_root.resolve()


def get_babel_dir(release_name: str = "babel_v1.0_release") -> Path:
    """获取 BABEL annotation 目录。"""
    return get_data_root() / "datasets" / "babel" / release_name


def get_amass_dir() -> Path:
    """获取 AMASS 数据集根目录。"""
    return get_data_root() / "datasets" / "amass"


def get_cache_dir(sub_dir: Optional[str] = None) -> Path:
    """获取缓存目录。"""
    base = get_data_root() / "cache"
    if sub_dir:
        base = base / sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_assets_dir(sub_dir: Optional[str] = None) -> Path:
    """获取资产目录。"""
    base = get_data_root() / "assets"
    if sub_dir:
        base = base / sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_logs_dir(sub_dir: Optional[str] = None) -> Path:
    """获取日志目录。"""
    base = get_data_root() / "logs"
    if sub_dir:
        base = base / sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_tmp_dir(sub_dir: Optional[str] = None) -> Path:
    """获取临时文件目录。"""
    base = get_data_root() / "tmp"
    if sub_dir:
        base = base / sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def to_relative_data_path(path: Union[str, Path]) -> str:
    """将绝对路径转换为相对于 data_root 的相对 POSIX 路径。"""
    p = Path(path).resolve()
    root = get_data_root()
    try:
        rel = p.relative_to(root)
        return rel.as_posix()
    except ValueError:
        return p.as_posix()


def from_relative_data_path(rel_path: str) -> Path:
    """将相对于 data_root 的相对路径还原为绝对 Path。"""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return (get_data_root() / p).resolve()
