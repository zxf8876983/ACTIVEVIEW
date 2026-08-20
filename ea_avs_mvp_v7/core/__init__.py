"""
EA-AVS-MVP v7.0 核心基础模块
"""

from .paths import (
    get_repo_root,
    get_data_root,
    get_assets_dir,
    get_runs_dir,
    get_datasets_dir,
    to_relative_data_path,
    from_relative_data_path,
)
from .episode import Episode, EpisodeFrame
from .config import V7Config, load_v7_config

__all__ = [
    "get_repo_root",
    "get_data_root",
    "get_assets_dir",
    "get_runs_dir",
    "get_datasets_dir",
    "to_relative_data_path",
    "from_relative_data_path",
    "Episode",
    "EpisodeFrame",
    "V7Config",
    "load_v7_config",
]
