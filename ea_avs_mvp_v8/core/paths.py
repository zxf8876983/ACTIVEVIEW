"""
路径解析核心模块 —— paths.py
============================

职责：
    1. 统一解析 Git 源码根目录与外部数据根目录 (ACTIVEVIEW_DATA_ROOT)；
    2. 提供安全、统一的相对路径序列化与反序列化工具。
"""

from pathlib import Path
from typing import Union

# 复用 v7 路径工具
from ea_avs_mvp_v7.core.paths import (
    get_repo_root,
    get_data_root,
    to_relative_data_path,
    from_relative_data_path,
    get_assets_dir,
    get_runs_dir,
    get_datasets_dir,
)


def ensure_dir(p: Union[str, Path]) -> Path:
    """确保目录存在。"""
    path = Path(p).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "get_repo_root",
    "get_data_root",
    "to_relative_data_path",
    "from_relative_data_path",
    "get_assets_dir",
    "get_runs_dir",
    "get_datasets_dir",
    "ensure_dir",
]
