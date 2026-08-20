"""
统一配置加载器 —— config.py
============================

功能：
    1. 自动聚合并加载 configs/ 下的 habitat, humanoid, motion, sensor 子配置；
    2. 提供 v7_demo.yaml 演示配置加载；
    3. 提供类型化访问接口与字典转换；
    4. 支持动态路径覆盖与安全默认值。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .paths import get_repo_root


@dataclass
class V7Config:
    """EA-AVS-MVP v7.0 统一配置容器。"""
    habitat: Dict[str, Any] = field(default_factory=dict)
    humanoid: Dict[str, Any] = field(default_factory=dict)
    motion: Dict[str, Any] = field(default_factory=dict)
    sensor: Dict[str, Any] = field(default_factory=dict)
    demo: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "habitat": self.habitat,
            "humanoid": self.humanoid,
            "motion": self.motion,
            "sensor": self.sensor,
            "demo": self.demo,
        }


def load_v7_config(
    configs_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> V7Config:
    """从 configs 目录加载全部 yaml 配置文件并返回 V7Config。"""
    base_dir = Path(configs_dir) if configs_dir else get_repo_root() / "ea_avs_mvp_v7" / "configs"
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parent.parent / "configs"

    def _read_yaml(filenames: list) -> Dict[str, Any]:
        for fn in filenames:
            p = base_dir / fn
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        return {}

    habitat_cfg = _read_yaml(["habitat_config.yaml", "habitat.yaml"])
    humanoid_cfg = _read_yaml(["humanoid.yaml", "humanoid_config.yaml"])
    motion_cfg = _read_yaml(["motion_config.yaml", "motion.yaml"])
    sensor_cfg = _read_yaml(["sensor_config.yaml", "sensor.yaml"])
    demo_cfg = _read_yaml(["v7_demo.yaml", "demo.yaml"])

    cfg = V7Config(
        habitat=habitat_cfg,
        humanoid=humanoid_cfg,
        motion=motion_cfg,
        sensor=sensor_cfg,
        demo=demo_cfg,
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k) and isinstance(v, dict):
                getattr(cfg, k).update(v)

    return cfg


def load_demo_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """加载 v7_demo.yaml 专用配置。"""
    if config_path:
        p = Path(config_path)
    else:
        p = get_repo_root() / "ea_avs_mvp_v7" / "configs" / "v7_demo.yaml"
        if not p.exists():
            p = Path(__file__).resolve().parent.parent / "configs" / "v7_demo.yaml"

    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
