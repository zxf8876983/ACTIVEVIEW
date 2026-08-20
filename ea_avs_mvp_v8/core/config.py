"""
v8 统一配置加载器 —— config.py
==============================

功能：
    1. 加载 configs/ 下的 v8_demo.yaml, viewpoint.yaml, humanoid.yaml 等配置；
    2. 支持动态路径覆盖与安全默认值。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .paths import get_repo_root


@dataclass
class V8Config:
    """EA-AVS-MVP v8.0 统一配置容器。"""
    scene: Dict[str, Any] = field(default_factory=dict)
    human: Dict[str, Any] = field(default_factory=dict)
    robot: Dict[str, Any] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    viewpoint: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "human": self.human,
            "robot": self.robot,
            "camera": self.camera,
            "viewpoint": self.viewpoint,
            "simulation": self.simulation,
        }


def load_v8_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> V8Config:
    """加载 v8 配置容器。"""
    base_dir = get_repo_root() / "ea_avs_mvp_v8" / "configs"
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parent.parent / "configs"

    if config_path:
        demo_p = Path(config_path)
    else:
        demo_p = base_dir / "v8_demo.yaml"

    demo_cfg = {}
    if demo_p.exists():
        with open(demo_p, "r", encoding="utf-8") as f:
            demo_cfg = yaml.safe_load(f) or {}

    vp_p = base_dir / "viewpoint.yaml"
    vp_cfg = {}
    if vp_p.exists():
        with open(vp_p, "r", encoding="utf-8") as f:
            vp_cfg = yaml.safe_load(f) or {}

    human_p = base_dir / "humanoid.yaml"
    human_cfg = {}
    if human_p.exists():
        with open(human_p, "r", encoding="utf-8") as f:
            human_cfg = yaml.safe_load(f) or {}

    merged_human = {**human_cfg, **demo_cfg.get("human", {})}
    merged_vp = {**vp_cfg, **demo_cfg.get("viewpoint", {})}

    cam_raw = demo_cfg.get("camera", {})
    normalized_cam = {
        "width": int(cam_raw.get("width", 640)),
        "height": int(cam_raw.get("height", cam_raw.get("height_px", 480))),
        "hfov_deg": float(cam_raw.get("hfov_deg", cam_raw.get("hfov", 90.0))),
        "camera_height": float(cam_raw.get("camera_height", 1.2)),
        "clip_near": float(cam_raw.get("clip_near", 0.01)),
        "clip_far": float(cam_raw.get("clip_far", 10.0)),
    }

    cfg = V8Config(
        scene=demo_cfg.get("scene", {}),
        human=merged_human,
        robot=demo_cfg.get("robot", {}),
        camera=normalized_cam,
        viewpoint=merged_vp,
        simulation=demo_cfg.get("simulation", {}),
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k) and isinstance(v, dict):
                getattr(cfg, k).update(v)

    return cfg
