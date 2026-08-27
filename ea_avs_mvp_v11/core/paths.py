"""Canonical source and runtime paths for the v11 selected16 pipeline.

职责：
    1. 解析 ACTIVEVIEW Git 源码仓库根目录；
    2. 解析默认运行时数据目录 (/home/zxf/WorkSpace/code/data/ActiveView/ 或 ACTIVEVIEW_DATA_ROOT 覆盖)；
"""

import os
from pathlib import Path


def get_repo_root() -> Path:
    """获取 ACTIVEVIEW 源码仓库根目录。"""
    # ea_avs_mvp_v11/core/paths.py -> repo_root
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """
    获取 ACTIVEVIEW 运行时数据根目录。
    支持 ACTIVEVIEW_DATA_ROOT 环境变量覆盖；默认相对解析为 ../../data/ActiveView/。
    """
    env_root = os.environ.get("ACTIVEVIEW_DATA_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    default_p = (get_repo_root().parent.parent / "data" / "ActiveView").resolve()
    default_p.mkdir(parents=True, exist_ok=True)
    return default_p


def get_habitat_data_root() -> Path:
    """Return the fixed local Habitat asset root without scanning other disks.

    ``ACTIVEVIEW_HABITAT_DATA_ROOT`` can override the default sibling path
    ``../robot/DATA`` for another machine.  The default is resolved relative
    to this repository, so it does not embed a developer-specific absolute
    path in the v11 code.
    """
    env_root = os.environ.get("ACTIVEVIEW_HABITAT_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (get_repo_root().parent / "robot" / "DATA").resolve()


def get_humanoid_asset_root(model_name: str = "male_0") -> Path:
    """Return the project-local copy of a Habitat humanoid model."""
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be a non-empty directory name")
    return get_data_root() / "assets" / "habitat_humanoids" / model_name


def get_humanoid_urdf_path(model_name: str = "male_0") -> Path:
    """Return the URDF path for a project-local Habitat humanoid model."""
    return get_humanoid_asset_root(model_name) / f"{model_name}.urdf"
