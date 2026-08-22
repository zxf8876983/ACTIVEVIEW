"""
v10 配置加载器 —— config.py
===========================

职责：
    1. 加载并解析 v10.0 YAML 配置文件；
    2. 支持默认参数回退与环境参数动态注入；
    3. 解析室内场景相对路径与数据路径。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from .paths import get_data_root, get_repo_root


def get_default_config_path() -> Path:
    """获取 v10 默认配置文件路径。"""
    return Path(__file__).resolve().parent.parent / "configs" / "default_v10_config.yaml"


class V10Config:
    """v10.0 配置项包装容器。"""

    def __init__(self, cfg_dict: Dict[str, Any]):
        self._raw = cfg_dict
        self.version = cfg_dict.get("version", "10.0.0")
        self.scene = cfg_dict.get("scene", {})
        self.camera = cfg_dict.get("camera", {})
        self.human = cfg_dict.get("human", {})
        self.motion = cfg_dict.get("motion", {})
        self.viewpoint = cfg_dict.get("viewpoint", {})
        self.dataset = cfg_dict.get("dataset_generation", {})

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


def load_v10_config(config_path: Optional[Union[str, Path]] = None) -> V10Config:
    """加载 v10.0 配置。"""
    p = Path(config_path) if config_path else get_default_config_path()
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    # 规范解析 scene_path
    if "scene" in raw_dict and "scene_path" in raw_dict["scene"]:
        s_path = Path(raw_dict["scene"]["scene_path"])
        if not s_path.is_absolute():
            resolved = get_repo_root().parent / s_path
            raw_dict["scene"]["scene_path"] = str(resolved)

    return V10Config(raw_dict)
