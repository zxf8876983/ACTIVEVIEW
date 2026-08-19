"""
配置文件加载模块 —— config.py
===============================

功能：
    读取 YAML 配置文件，验证 v6.0 必要字段。
"""

import os
import yaml

REQUIRED_FIELDS = [
    "habitat.scene_path",
    "camera.width",
    "camera.height",
    "candidate_sampling.radii",
    "candidate_sampling.angles_deg",
    "action_part_weights",
    "orientation_score.preferred_angle_deg",
    "predictive_score.w_action_occ_pred",
    "true_score.w_action_occ_true",
    "occlusion.enabled",
    "humanoid.enabled",
    "perception.pose_backend",
    "human_state_estimation.min_2d_keypoints",
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

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空: {path}")

    missing = [field for field in REQUIRED_FIELDS if not _check_field(config, field)]
    if missing:
        raise ValueError(f"缺少必要的配置字段: {', '.join(missing)}")

    return config
