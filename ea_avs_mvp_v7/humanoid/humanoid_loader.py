"""
Humanoid 资产加载器 —— humanoid_loader.py
=========================================

功能：
    1. 解析与校验 Humanoid URDF 与默认动作数据资产；
    2. 支持基于相对路径或环境变量定位 Habitat Humanoid 资产；
    3. 封装 HumanoidAssetBundle。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from tools.motion_assets.data_paths import get_repo_root, get_data_root

CANDIDATE_HUMANOID_PATHS = [
    Path("/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids"),
    get_data_root() / "assets" / "habitat_humanoids",
    get_repo_root().parent / "robot" / "habitat-lab" / "data" / "versioned_data" / "habitat_humanoids",
]


@dataclass
class HumanoidAssetBundle:
    """Humanoid 资产包描述。"""
    avatar_name: str
    urdf_path: str
    motion_data_path: str
    skin_gltf_path: Optional[str] = None
    gender: str = "neutral"


def resolve_humanoid_assets(
    config: Optional[Dict] = None,
    avatar_name: str = "neutral_0",
) -> HumanoidAssetBundle:
    """解析并定位 Humanoid 资产路径。"""
    cfg_root = None
    if config and "humanoid" in config:
        cfg_root = config["humanoid"].get("assets_root")
        avatar_name = config["humanoid"].get("avatar_name", avatar_name)

    search_roots = []
    if cfg_root:
        p = Path(cfg_root)
        if not p.is_absolute():
            search_roots.append(get_repo_root().parent / p)
            search_roots.append(get_data_root() / p)
        else:
            search_roots.append(p)

    search_roots.extend(CANDIDATE_HUMANOID_PATHS)

    found_root = None
    for r in search_roots:
        if r.exists() and (r / avatar_name).exists():
            found_root = r
            break

    if not found_root:
        raise FileNotFoundError(
            f"Could not find Humanoid '{avatar_name}' in search paths: {search_roots}"
        )

    urdf_p = found_root / avatar_name / f"{avatar_name}.urdf"
    if not urdf_p.exists():
        raise FileNotFoundError(f"Humanoid URDF not found: {urdf_p}")

    motion_p = found_root / "walking_motion_processed_smplx.pkl"
    if not motion_p.exists():
        motion_p = found_root / avatar_name / "motion_data.pkl"

    return HumanoidAssetBundle(
        avatar_name=avatar_name,
        urdf_path=str(urdf_p.resolve()),
        motion_data_path=str(motion_p.resolve()) if motion_p.exists() else str(urdf_p.parent),
    )


def validate_humanoid_assets(bundle: HumanoidAssetBundle) -> bool:
    """验证资产包文件完整性。"""
    if not Path(bundle.urdf_path).exists():
        raise FileNotFoundError(f"URDF path does not exist: {bundle.urdf_path}")
    return True
