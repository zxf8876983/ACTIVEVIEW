"""文件用途：解析 AMASS/BABEL motion 资产、缓存和日志的运行时路径。

主要输入：ACTIVEVIEW_DATA_ROOT 环境变量和可选的子目录名。
主要输出：规范化的 motion 资产路径与相对数据路径。
项目角色：data.motion 的路径辅助层，复用 core.paths 的数据根目录定义。
"""

from __future__ import annotations

from pathlib import Path

from activeview.core.paths import get_data_root


def get_babel_dir(release_name: str = "babel_v1.0_release") -> Path:
    """Return the BABEL annotation directory."""
    return get_data_root() / "datasets" / "babel" / release_name


def get_amass_dir() -> Path:
    """Return the AMASS dataset root."""
    return get_data_root() / "datasets" / "amass"


def get_cache_dir(sub_dir: str | None = None) -> Path:
    """Return and create a motion cache directory."""
    base = get_data_root() / "cache"
    if sub_dir:
        base /= sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_assets_dir(sub_dir: str | None = None) -> Path:
    """Return and create a motion assets directory."""
    base = get_data_root() / "assets"
    if sub_dir:
        base /= sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_logs_dir(sub_dir: str | None = None) -> Path:
    """Return and create a logs directory."""
    base = get_data_root() / "logs"
    if sub_dir:
        base /= sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_tmp_dir(sub_dir: str | None = None) -> Path:
    """Return and create a temporary runtime directory."""
    base = get_data_root() / "tmp"
    if sub_dir:
        base /= sub_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def to_relative_data_path(path: str | Path) -> str:
    """Convert a path to a POSIX path relative to the runtime data root."""
    value = Path(path).resolve()
    try:
        return value.relative_to(get_data_root()).as_posix()
    except ValueError:
        return value.as_posix()


def from_relative_data_path(rel_path: str) -> Path:
    """Resolve a runtime-data-relative path."""
    value = Path(rel_path)
    return value if value.is_absolute() else (get_data_root() / value).resolve()
