"""
动作分类与标准化别名映射 —— action_types.py
========================================
"""

from typing import Dict
from ea_avs_mvp_v9.core.types import ActionClass

ACTION_ALIASES: Dict[str, ActionClass] = {
    # Fall aliases
    "fall": ActionClass.FALL,
    "fall_related": ActionClass.FALL,
    "fall to the ground": ActionClass.FALL,
    "falling": ActionClass.FALL,
    "collapse": ActionClass.FALL,
    "trip": ActionClass.FALL,
    "slip": ActionClass.FALL,

    # Sitting aliases
    "sitting": ActionClass.SITTING,
    "sit": ActionClass.SITTING,
    "sit down": ActionClass.SITTING,
    "sit on chair": ActionClass.SITTING,
    "seated": ActionClass.SITTING,

    # Standing aliases
    "standing": ActionClass.STANDING,
    "stand": ActionClass.STANDING,
    "stand up": ActionClass.STANDING,
    "idle": ActionClass.STANDING,
    "upright": ActionClass.STANDING,

    # Bending aliases
    "bending": ActionClass.BENDING,
    "bend": ActionClass.BENDING,
    "bend down": ActionClass.BENDING,
    "stoop": ActionClass.BENDING,
    "crouch": ActionClass.BENDING,

    # Reaching aliases
    "reaching": ActionClass.REACHING,
    "reach": ActionClass.REACHING,
    "reach for": ActionClass.REACHING,
    "grasp": ActionClass.REACHING,
    "stretch": ActionClass.REACHING,
}


def normalize_action_label(label: str) -> ActionClass:
    """将任意输入的动作标签字符串标准化为系统支持的 ActionClass。"""
    clean_str = str(label).lower().strip().replace("_", " ")
    for alias, act_class in ACTION_ALIASES.items():
        if alias in clean_str or clean_str in alias:
            return act_class

    # 默认回退
    return ActionClass.STANDING
