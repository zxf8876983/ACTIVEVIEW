"""
动作编码器与先验特征提取器 —— action_encoder.py
=============================================

职责：
    1. 将离散动作标签转换为 One-hot 向量与语义表示；
    2. 加载各动作针对老人监护任务的关键身体部位 (critical_regions) 与观测先验 (preferred angles, optimal distance)；
    3. 输出标准化 ActionEmbedding 结构。
"""

import logging
from typing import Any, Dict, List, Optional, Union
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

DEFAULT_ACTION_PRIORS: Dict[ActionClass, Dict[str, Any]] = {
    ActionClass.FALL: {
        "critical_regions": ["torso", "pelvis", "head", "lower_body"],
        "preferred_angle_range": [20.0, 75.0],
        "optimal_distance": 2.5,
        "region_weights": {"torso": 0.30, "pelvis": 0.30, "head": 0.20, "lower_body": 0.20},
        "aspect_weight": 0.25,
        "distance_weight": 0.20,
    },
    ActionClass.SITTING: {
        "critical_regions": ["lower_body", "pelvis", "torso"],
        "preferred_angle_range": [40.0, 90.0],
        "optimal_distance": 2.0,
        "region_weights": {"lower_body": 0.45, "pelvis": 0.30, "torso": 0.25, "head": 0.00},
        "aspect_weight": 0.25,
        "distance_weight": 0.15,
    },
    ActionClass.STANDING: {
        "critical_regions": ["torso", "head", "upper_body", "lower_body"],
        "preferred_angle_range": [0.0, 30.0],
        "optimal_distance": 2.0,
        "region_weights": {"torso": 0.30, "head": 0.30, "upper_body": 0.20, "lower_body": 0.20},
        "aspect_weight": 0.20,
        "distance_weight": 0.15,
    },
    ActionClass.BENDING: {
        "critical_regions": ["torso", "pelvis", "head"],
        "preferred_angle_range": [45.0, 90.0],
        "optimal_distance": 2.2,
        "region_weights": {"torso": 0.45, "pelvis": 0.35, "head": 0.20, "lower_body": 0.00},
        "aspect_weight": 0.30,
        "distance_weight": 0.15,
    },
    ActionClass.REACHING: {
        "critical_regions": ["upper_body", "torso", "head"],
        "preferred_angle_range": [15.0, 60.0],
        "optimal_distance": 1.8,
        "region_weights": {"upper_body": 0.50, "torso": 0.30, "head": 0.20, "lower_body": 0.00},
        "aspect_weight": 0.25,
        "distance_weight": 0.15,
    },
}


class ActionEncoder:
    """动作编码与先验映射器。"""

    def __init__(self, action_weights_config: Optional[Dict[str, Any]] = None):
        self.action_weights = action_weights_config or {}

    def encode(self, action_label: Union[str, ActionClass]) -> ActionEmbedding:
        """将动作标签编码为 ActionEmbedding。"""
        if isinstance(action_label, ActionClass):
            act_class = action_label
        else:
            act_class = normalize_action_label(action_label)

        # 1. 生成 One-hot 向量
        one_hot = [0.0] * len(ALL_ACTION_CLASSES)
        idx = ALL_ACTION_CLASSES.index(act_class)
        one_hot[idx] = 1.0

        # 2. 读取配置中的先验或使用默认先验
        cfg_prior = self.action_weights.get(act_class.value, {})
        default_prior = DEFAULT_ACTION_PRIORS[act_class]

        critical_regions = cfg_prior.get("critical_regions", default_prior["critical_regions"])
        preferred_angles = cfg_prior.get("preferred_angle_range", default_prior["preferred_angle_range"])
        optimal_dist = float(cfg_prior.get("optimal_distance", default_prior["optimal_distance"]))
        region_weights = cfg_prior.get("region_weights", default_prior["region_weights"])
        aspect_weight = float(cfg_prior.get("aspect_weight", default_prior["aspect_weight"]))
        dist_weight = float(cfg_prior.get("distance_weight", default_prior["distance_weight"]))

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
