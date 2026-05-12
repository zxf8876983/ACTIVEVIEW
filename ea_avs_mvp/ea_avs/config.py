"""
配置文件加载模块 —— config.py
===============================

功能：
    读取 YAML 格式的配置文件，验证必要字段是否存在，返回 Python 字典。

本模块是整个实验的配置入口，所有可调节参数（场景路径、相机参数、采样半径、
评分权重等）都通过 YAML 配置文件传入，保证实验可复现和参数可配置。

严格遵守「不做实验逻辑」的原则 —— 本模块只负责配置的读取和验证。
"""

import os
import yaml

# 必要字段列表 —— 这些字段在配置文件中必须存在，否则实验无法进行
# 使用点号分隔表示嵌套字典的路径，例如 "habitat.scene_path" 表示
# config["habitat"]["scene_path"]
REQUIRED_FIELDS = [
    "habitat.scene_path",           # Habitat 场景文件路径（.glb）
    "camera.width",                  # 渲染图像宽度（像素）
    "camera.height",                 # 渲染图像高度（像素）
    "candidate_sampling.radii",      # 候选视角采样半径列表（米）
    "candidate_sampling.angles_deg", # 候选视角采样角度列表（度）
]


def _check_field(config: dict, dotted_path: str) -> bool:
    """
    检查嵌套字典中是否存在指定路径的字段。

    参数：
        config: 配置字典（可能有多层嵌套）
        dotted_path: 点号分隔的字段路径，如 "habitat.scene_path"

    返回：
        True 表示字段存在，False 表示不存在

    实现说明：
        逐层向下遍历字典，如果中间某层不是字典或键不存在则返回 False。
        例如 "habitat.scene_path" 会先找 config["habitat"]，再找其中的 "scene_path"。
    """
    keys = dotted_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def load_config(path: str) -> dict:
    """
    加载 YAML 配置文件并返回配置字典。

    参数：
        path: YAML 配置文件的路径（字符串）。

    返回：
        包含所有配置项的嵌套字典。

    抛出异常：
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 配置文件为空或缺少必要字段时抛出。

    使用示例：
        >>> config = load_config("configs/mvp_visibility.yaml")
        >>> print(config["camera"]["width"])
        640

    实现要求：
        - 使用 yaml.safe_load 加载（防止 YAML 注入攻击）
        - 检查配置文件是否存在于磁盘上
        - 检查所有必要字段是否都存在
        - 不要在本函数内初始化 Habitat 或执行任何实验逻辑
    """
    # ---------- 第一步：检查文件是否存在 ----------
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件未找到: {path}")

    # ---------- 第二步：读取并解析 YAML ----------
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # ---------- 第三步：检查配置文件是否为空 ----------
    if config is None:
        raise ValueError(f"配置文件为空: {path}")

    # ---------- 第四步：检查所有必要字段 ----------
    missing = [field for field in REQUIRED_FIELDS if not _check_field(config, field)]
    if missing:
        raise ValueError(
            f"缺少必要的配置字段: {', '.join(missing)}"
        )

    return config
