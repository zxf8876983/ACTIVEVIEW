"""
场景资产与 NavMesh 管理器 —— scene_manager.py
============================================

功能：
    1. 解析 Habitat 场景 .glb 与对应 .navmesh 资产；
    2. 支持基于相对数据路径与备选路径自动定位。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ea_avs_mvp_v7.core.paths import get_repo_root, get_data_root


def resolve_scene_path(scene_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Tuple[Path, Optional[Path]]:
    """解析场景 glb 路径与 navmesh 路径。"""
    candidate_roots = [
        get_repo_root().parent / "robot" / "habitat-sim" / "data" / "versioned_data" / "habitat_test_scenes",
        get_data_root() / "assets" / "scenes",
    ]

    target_str = None
    if isinstance(scene_config_path, dict):
        target_str = scene_config_path.get("scene_path") or scene_config_path.get("scene_id")
    elif scene_config_path is not None:
        target_str = str(scene_config_path)

    found_scene = None
    if target_str:
        p = Path(target_str)
        if p.is_absolute() and p.exists():
            found_scene = p
        elif (get_repo_root().parent / p).exists():
            found_scene = get_repo_root().parent / p
        elif (get_data_root() / p).exists():
            found_scene = get_data_root() / p

    if not found_scene:
        for r in candidate_roots:
            cand = r / "apartment_1.glb"
            if cand.exists():
                found_scene = cand
                break

    if not found_scene or not found_scene.exists():
        raise FileNotFoundError(f"Habitat scene glb not found. Checked candidate roots: {candidate_roots}")

    navmesh_path = found_scene.with_suffix(".navmesh")
    navmesh_resolved = navmesh_path.resolve() if navmesh_path.exists() else None

    return found_scene.resolve(), navmesh_resolved


class SceneManager:
    """场景资产管理器。"""

    def __init__(self, scene_path: Optional[str] = None):
        self.scene_path, self.navmesh_path = resolve_scene_path(scene_path)

    @property
    def scene_id(self) -> str:
        return self.scene_path.stem
