"""
Humanoid 资源模块 —— humanoid_assets.py
=========================================

功能：
    只负责 Humanoid 资源路径发现与完整性检查。
    不在此文件执行任何 Habitat 仿真逻辑。

本机官方 habitat_humanoids 资源目录结构（v5.0 默认使用 neutral_0）：
    <assets_root>/
    ├── neutral_0/
    │   ├── neutral_0.ao_config.json
    │   ├── neutral_0.glb
    │   ├── neutral_0.urdf
    │   └── neutral_0_motion_data_smplx.pkl
    ├── standing_pose_smplx.pkl
    ├── walking_motion_processed_smplx.pkl
    └── walk_motion/
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HumanoidAssetBundle:
    """Humanoid 资源包（文件路径集合）。"""
    avatar_name: str
    root_dir: str
    ao_config_path: str
    glb_path: str
    urdf_path: str
    motion_data_path: str
    # 可选：官方提供的独立 standing/walking 资源（不代表单个 avatar）
    standing_pose_path: Optional[str] = None
    walking_motion_path: Optional[str] = None
    base_height_offset: float = 0.0


def resolve_humanoid_assets(config: dict) -> HumanoidAssetBundle:
    """根据配置定位 Humanoid 资源路径。

    参数：
        config: 配置字典，需要 humanoid 配置段：
            - assets_root: habitat_humanoids 资源根目录
            - avatar_name: avatar 名称，如 "neutral_0"

    返回：
        HumanoidAssetBundle。

    抛出异常：
        FileNotFoundError: assets_root 不存在。
    """
    humanoid_cfg = config["humanoid"]
    assets_root = humanoid_cfg["assets_root"]
    avatar_name = humanoid_cfg["avatar_name"]

    if not os.path.isdir(assets_root):
        raise FileNotFoundError(
            f"Humanoid assets root 不存在: {assets_root}\n"
            f"请下载 ai-habitat/habitat_humanoids 并设置 humanoid.assets_root。"
        )

    root_dir = os.path.join(assets_root, avatar_name)

    bundle = HumanoidAssetBundle(
        avatar_name=avatar_name,
        root_dir=root_dir,
        ao_config_path=os.path.join(root_dir, f"{avatar_name}.ao_config.json"),
        glb_path=os.path.join(root_dir, f"{avatar_name}.glb"),
        urdf_path=os.path.join(root_dir, f"{avatar_name}.urdf"),
        motion_data_path=os.path.join(
            root_dir, f"{avatar_name}_motion_data_smplx.pkl"),
        standing_pose_path=os.path.join(assets_root, "standing_pose_smplx.pkl"),
        walking_motion_path=os.path.join(
            assets_root, "walking_motion_processed_smplx.pkl"),
        base_height_offset=float(humanoid_cfg.get("base_height_offset", 0.0)),
    )
    return bundle


def validate_humanoid_assets(bundle: HumanoidAssetBundle) -> None:
    """验证 Humanoid 资源完整性。

    至少检查：
        - AO config 是否存在
        - GLB 是否存在
        - URDF 是否存在
        - motion PKL 是否存在

    参数：
        bundle: HumanoidAssetBundle。

    抛出异常：
        FileNotFoundError: 任一必需资源缺失，附带清晰的下载提示。
    """
    missing = []
    required = {
        "AO config": bundle.ao_config_path,
        "GLB": bundle.glb_path,
        "URDF": bundle.urdf_path,
        "motion PKL": bundle.motion_data_path,
    }
    for name, path in required.items():
        if not os.path.isfile(path):
            missing.append(f"{name}: {path}")

    if missing:
        detail = "\n".join(missing)
        raise FileNotFoundError(
            f"Humanoid assets missing for {bundle.avatar_name}:\n{detail}\n"
            f"Please download ai-habitat/habitat_humanoids first and verify "
            f"humanoid.assets_root / humanoid.avatar_name."
        )