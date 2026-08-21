"""
v10 运行时路径管理与数据边界 —— paths.py
======================================

职责：
    1. 解析 ACTIVEVIEW Git 源码仓库根目录；
    2. 解析默认运行时数据目录 (/home/zxf/WorkSpace/code/data/ActiveView/ 或 ACTIVEVIEW_DATA_ROOT 覆盖)；
    3. 规范定义 v10.0 数据集目录结构：
       datasets/v10/
       ├── raw/ (rgb, depth, camera_pose, scene_meta)
       ├── ground_truth/ (skeleton, action)
       └── metadata/ (samples.json, dataset_manifest.json)
"""

import os
from pathlib import Path
from typing import Dict, Union


def get_repo_root() -> Path:
    """获取 ACTIVEVIEW 源码仓库根目录。"""
    # ea_avs_mvp_v10/core/paths.py -> repo_root
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """
    获取 ACTIVEVIEW 运行时数据根目录。
    支持 ACTIVEVIEW_DATA_ROOT 环境变量覆盖；默认相对解析为 ../../data/ActiveView/。
    """
    env_root = os.environ.get("ACTIVEVIEW_DATA_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    default_p = (get_repo_root().parent.parent / "data" / "ActiveView").resolve()
    default_p.mkdir(parents=True, exist_ok=True)
    return default_p


def get_v10_dataset_root() -> Path:
    """获取 v10.0 数据集根目录 (datasets/v10)。"""
    root = get_data_root() / "datasets" / "v10"
    root.mkdir(parents=True, exist_ok=True)
    return root


def init_v10_dataset_dirs(base_dir: Union[str, Path, None] = None) -> Dict[str, Path]:
    """
    初始化并返回 v10.0 数据集标准子目录字典。
    
    结构：
        datasets/v10/
        ├── raw/
        │   ├── rgb/
        │   ├── depth/
        │   ├── camera_pose/
        │   └── scene_meta/
        ├── ground_truth/
        │   ├── skeleton/
        │   └── action/
        └── metadata/
    """
    root = Path(base_dir) if base_dir else get_v10_dataset_root()

    dirs = {
        "root": root,
        "raw": root / "raw",
        "raw_rgb": root / "raw" / "rgb",
        "raw_depth": root / "raw" / "depth",
        "raw_camera_pose": root / "raw" / "camera_pose",
        "raw_scene_meta": root / "raw" / "scene_meta",
        "gt": root / "ground_truth",
        "gt_skeleton": root / "ground_truth" / "skeleton",
        "gt_action": root / "ground_truth" / "action",
        "metadata": root / "metadata",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs
