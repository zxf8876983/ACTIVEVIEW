"""
动作编码器与配置映射器 —— action_encoder.py
=========================================

职责：
    1. 接收离散动作标签 (action label)；
    2. 从配置文件 (configs/action_prior.yaml) 读取关键解剖部位 (critical_regions) 与观测先验 (preferred angles, optimal distance)；
    3. 生成标准化的 ActionEmbedding 结构；
    4. 严格禁止在 Python 代码中硬编码动作先验规则。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ea_avs_mvp_v9.core.paths import get_repo_root
from ea_avs_mvp_v9.core.types import ActionClass, ActionEmbedding
from .action_types import normalize_action_label

logger = logging.getLogger(__name__)

# 预设标准类别顺序 (保证 One-hot 向量的一致性)
ALL_ACTION_CLASSES = [
    ActionClass.FALL,
    ActionClass.SITTING,
    ActionClass.STANDING,
    ActionClass.BENDING,
    ActionClass.REACHING,
]


def load_action_prior_yaml(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """从 YAML 配置文件中读取动作先验知识库。"""
    if config_path:
        p = Path(config_path)
    else:
        base_dir = get_repo_root() / "ea_avs_mvp_v9" / "configs"
        if not base_dir.exists():
            base_dir = Path(__file__).resolve().parent.parent / "configs"

        p = base_dir / "action_prior.yaml"
        if not p.exists():
            p = base_dir / "action_weights.yaml"

    if not p.exists():
        raise FileNotFoundError(f"Action prior configuration file not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("actions", {})


class ActionEncoder:
    """解耦的动作编码器，完全由 YAML 配置文件驱动。"""

    def __init__(
        self,
        action_prior_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        if action_prior_config is not None:
            self.action_priors = action_prior_config
        else:
            self.action_priors = load_action_prior_yaml(config_path)

    def encode(self, action_label: Union[str, ActionClass]) -> ActionEmbedding:
        """将动作标签编码为 ActionEmbedding。"""
        if isinstance(action_label, ActionClass):
            act_class = action_label
        else:
            act_class = normalize_action_label(action_label)

        # 1. 生成 One-hot 向量
        one_hot = [0.0] * len(ALL_ACTION_CLASSES)
        if act_class in ALL_ACTION_CLASSES:
            idx = ALL_ACTION_CLASSES.index(act_class)
            one_hot[idx] = 1.0

        # 2. 从配置读取动作先验
        cfg_prior = self.action_priors.get(act_class.value)
        if not cfg_prior:
            # 兼容别名或回退
            cfg_prior = self.action_priors.get(
                "standing",
                {
                    "critical_regions": ["torso", "head", "upper_body", "lower_body"],
                    "preferred_angle_range": [0.0, 30.0],
                    "optimal_distance": 2.0,
                    "region_weights": {"torso": 0.3, "head": 0.3, "upper_body": 0.2, "lower_body": 0.2},
                    "aspect_weight": 0.20,
                    "distance_weight": 0.15,
                }
            )

        critical_regions = list(cfg_prior.get("critical_regions", ["torso"]))
        preferred_angles = [float(x) for x in cfg_prior.get("preferred_angle_range", [0.0, 45.0])]
        optimal_dist = float(cfg_prior.get("optimal_distance", 2.0))
        region_weights = {k: float(v) for k, v in cfg_prior.get("region_weights", {}).items()}
        aspect_weight = float(cfg_prior.get("aspect_weight", 0.25))
        dist_weight = float(cfg_prior.get("distance_weight", 0.15))

        return ActionEmbedding(
            action_name=act_class.value,
            action_class=act_class,
            vector=one_hot,
            critical_regions=critical_regions,
            preferred_angle_range=preferred_angles,
            optimal_distance=optimal_dist,
            region_weights=region_weights,
            aspect_weight=aspect_weight,
            distance_weight=dist_weight,
            metadata={"raw_label": str(action_label)},
        )
