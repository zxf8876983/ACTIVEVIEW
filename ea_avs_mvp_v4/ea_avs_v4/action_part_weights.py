"""
动作关键部位权重模块 —— action_part_weights.py
=================================================

功能：
    根据姿态类型（pose_type）返回对应的动作关键部位权重。
    从 v3.0 迁移，v4.0 保持不变。

v3.0/v4.0 核心设计：
    - 不同动作/姿态有不同的关键部位权重
    - 例如：判断"坐姿"时，下肢（膝盖角度）比头部更重要
    - 判断"跌倒"时，躯干姿态和下肢展开角度是关键

权重设计原则：
    - standing:   躯干和下肢均衡（各 0.35），头部（0.20），手臂（0.10）
    - sitting:    下肢最重要（0.45），躯干次之（0.40）
    - lying_fallen: 躯干最重要（0.40），下肢次之（0.35），头部（0.20）
    - bending:    躯干最重要（0.45），下肢次之（0.35）
"""

from typing import Dict


def get_action_part_weights(pose_type: str, config: dict) -> Dict[str, float]:
    """根据姿态类型获取动作关键部位权重。

    参数：
        pose_type: 姿态类型，如 "standing"/"sitting"/"lying_fallen"/"bending"。
        config: 配置字典，需要 action_part_weights 配置段。

    返回：
        权重字典，如 {"torso": 0.40, "lower_body": 0.45, "head": 0.10, "arms": 0.05}
        权重和接近 1.0。

    优先级：
        1. 优先读取 config["action_part_weights"][pose_type]
        2. 如果找不到，退回 config["score_weights"]
        3. 如果还找不到，返回默认权重
    """
    # 尝试从 action_part_weights 获取该姿态的专用权重
    action_weights = config.get("action_part_weights", {})
    if pose_type in action_weights:
        weights = dict(action_weights[pose_type])
    else:
        # 退回 score_weights（通用权重）
        weights = dict(config.get("score_weights", {}))

    # 确保四个组都有值
    default_weights = {"torso": 0.40, "lower_body": 0.40, "head": 0.10, "arms": 0.10}
    for key in default_weights:
        if key not in weights:
            weights[key] = default_weights[key]

    # 归一化：确保权重和为 1.0
    total = sum(weights.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        for key in weights:
            weights[key] /= total

    return weights