"""
v9.0 路径解析与数据目录管理 —— paths.py
=======================================
"""

import os
from pathlib import Path
from typing import Union


def get_repo_root() -> Path:
    """获取代码仓库根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """获取数据存储根目录。"""
    env_data_root = os.environ.get("ACTIVEVIEW_DATA_ROOT")
    if env_data_root:
        p = Path(env_data_root).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    default_data_root = (get_repo_root().parent.parent / "data" / "ActiveView").resolve()
    default_data_root.mkdir(parents=True, exist_ok=True)
    return default_data_root


def to_relative_data_path(abs_path: Union[str, Path]) -> str:
    """将绝对路径转换为相对于 data 根目录的相对路径。"""
    data_root = get_data_root()
    p = Path(abs_path).resolve()
    try:
        return str(p.relative_to(data_root))
    except ValueError:
        return str(p)
