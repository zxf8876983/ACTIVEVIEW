"""
动作关键部位权重模块 —— action_part_weights.py
=================================================

功能：
    根据姿态类型（pose_type）返回对应的动作关键部位权重。
"""

from typing import Dict


def get_action_part_weights(pose_type: str, config: dict) -> Dict[str, float]:
    """根据姿态类型获取动作关键部位权重。"""
    action_weights = config.get("action_part_weights", {})
    if pose_type in action_weights:
        weights = dict(action_weights[pose_type])
    else:
        weights = dict(config.get("score_weights", {}))

    default_weights = {"torso": 0.40, "lower_body": 0.40, "head": 0.10, "arms": 0.10}
    for key in default_weights:
        if key not in weights:
            weights[key] = default_weights[key]

    total = sum(weights.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        for key in weights:
            weights[key] /= total

    return weights
