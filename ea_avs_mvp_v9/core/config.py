"""
v9.0 统一配置加载器 —— config.py
================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .paths import get_repo_root


@dataclass
class V9Config:
    """EA-AVS-MVP v9.0 统一配置容器。"""
    scene: Dict[str, Any] = field(default_factory=dict)
    human: Dict[str, Any] = field(default_factory=dict)
    robot: Dict[str, Any] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    viewpoint: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    action_weights: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "human": self.human,
            "robot": self.robot,
            "camera": self.camera,
            "viewpoint": self.viewpoint,
            "scoring": self.scoring,
            "action_weights": self.action_weights,
            "simulation": self.simulation,
        }


def load_v9_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> V9Config:
    """加载 v9 配置容器。"""
    base_dir = get_repo_root() / "ea_avs_mvp_v9" / "configs"
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parent.parent / "configs"

    if config_path:
        demo_p = Path(config_path)
    else:
        demo_p = base_dir / "v9_demo.yaml"

    demo_cfg = {}
    if demo_p.exists():
        with open(demo_p, "r", encoding="utf-8") as f:
            demo_cfg = yaml.safe_load(f) or {}

    act_p = base_dir / "action_weights.yaml"
    act_cfg = {}
    if act_p.exists():
        with open(act_p, "r", encoding="utf-8") as f:
            act_cfg = yaml.safe_load(f) or {}

    merged_scoring = {
        "w_geometry": 0.60,
        "w_action": 0.40,
        "evaluation_mode": "oracle",
        "pose_source": "oracle",
        **demo_cfg.get("scoring", {}),
    }

    cam_raw = demo_cfg.get("camera", {})
    normalized_cam = {
        "width": int(cam_raw.get("width", 640)),
        "height": int(cam_raw.get("height", cam_raw.get("height_px", 480))),
        "hfov_deg": float(cam_raw.get("hfov_deg", cam_raw.get("hfov", 90.0))),
        "camera_height": float(cam_raw.get("camera_height", 1.2)),
        "clip_near": float(cam_raw.get("clip_near", 0.01)),
        "clip_far": float(cam_raw.get("clip_far", 10.0)),
    }

    cfg = V9Config(
        scene=demo_cfg.get("scene", {}),
        human=demo_cfg.get("human", {}),
        robot=demo_cfg.get("robot", {}),
        camera=normalized_cam,
        viewpoint=demo_cfg.get("viewpoint", {}),
        scoring=merged_scoring,
        action_weights=act_cfg.get("actions", {}),
        simulation=demo_cfg.get("simulation", {}),
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k) and isinstance(v, dict):
                getattr(cfg, k).update(v)

    return cfg
