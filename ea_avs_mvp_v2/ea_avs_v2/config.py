"""
配置文件加载模块 —— config.py
===============================

功能：
    读取 YAML 配置文件，验证必要字段，返回配置字典。

必要字段列表（v2.0 新增）：
    - habitat.scene_path（场景文件路径）
    - camera.width（图像宽度）
    - camera.height（图像高度）
    - candidate_sampling.radii（候选采样半径）
    - candidate_sampling.angles_deg（候选采样角度）
    - predictive_score.w_kp_pred（预测评分 S_kp 权重）
    - true_score.w_kp_true（真实评分 S_kp 权重）
"""

import os
import yaml

# 必要字段列表 —— v2.0 新增了 predictive_score 和 true_score 的验证
REQUIRED_FIELDS = [
    "habitat.scene_path",
    "camera.width",
    "camera.height",
    "candidate_sampling.radii",
    "candidate_sampling.angles_deg",
    "predictive_score.w_kp_pred",
    "true_score.w_kp_true",
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
