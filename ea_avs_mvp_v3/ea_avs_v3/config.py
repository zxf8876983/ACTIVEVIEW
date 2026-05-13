"""
配置文件加载模块 —— config.py
===============================

功能：
    读取 YAML 配置文件，验证 v3.0 新增的必要字段。

v3.0 新增必要字段：
    - human.pose_types（支持的姿态类型列表）
    - action_part_weights（各姿态的动作关键部位权重）
    - orientation_score.preferred_angle_deg（朝向偏好角度）
    - predictive_score.w_action_part_pred（动作关键部位预测权重）
    - true_score.w_action_part_true（动作关键部位真实权重）
"""

import os
import yaml

# v3.0 必要字段 —— 相比 v2.0 新增了 human.pose_types、action_part_weights 等
REQUIRED_FIELDS = [
    "habitat.scene_path",
    "camera.width",
    "camera.height",
    "human.pose_types",
    "candidate_sampling.radii",
    "candidate_sampling.angles_deg",
    "action_part_weights",
    "orientation_score.preferred_angle_deg",
    "predictive_score.w_action_part_pred",
    "true_score.w_action_part_true",
]


def _check_field(config: dict, dotted_path: str) -> bool:
    """检查嵌套字典中是否存在指定路径的字段。"""
    keys = dotted_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def load_config(path: str) -> dict:
    """加载 YAML 配置文件。

    参数：
        path: YAML 配置文件路径。

    返回：
        包含所有配置项的嵌套字典。

    抛出异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置文件为空或缺少必要字段。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件未找到: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空: {path}")

    missing = [field for field in REQUIRED_FIELDS if not _check_field(config, field)]
    if missing:
        raise ValueError(
            f"缺少必要的配置字段: {', '.join(missing)}"
        )

    return config
