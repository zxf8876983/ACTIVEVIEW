"""
多场景资产与环境管理器 —— active_view/scene_manager.py (v11.3)
===========================================================

职责：
    1. 统一管理多个 Habitat 室内物理场景资产 (.glb, .navmesh, .scene_dataset.json)；
    2. 支持多目录索引与智能解析 (habitat_test_scenes, hssd-scenes, assets/scenes)；
    3. 提供场景边界、可行导航区域与地面标高元数据提取接口。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ea_avs_mvp_v11.core.paths import get_data_root, get_repo_root

logger = logging.getLogger("scene_manager")


@dataclass
class SceneMetadata:
    """场景元数据结构。"""
    scene_id: str
    glb_path: Path
    navmesh_path: Optional[Path] = None
    floor_height: float = 0.0
    ceiling_height: float = 2.5
    bounds_min: List[float] = field(default_factory=lambda: [-5.0, -0.1, -5.0])
    bounds_max: List[float] = field(default_factory=lambda: [5.0, 3.0, 5.0])
    is_valid: bool = True
    description: str = ""


# 预定义场景物理特性注册表
KNOWN_SCENE_PROFILES: Dict[str, Dict[str, Any]] = {
    "apartment_1": {
        "floor_height": -1.60,
        "ceiling_height": 1.14,
        "bounds_min": [-4.5, -1.60, -4.5],
        "bounds_max": [4.5, 1.14, 4.5],
        "description": "Standard Multi-Room Indoor Apartment with calibrated floor Y=-1.60m",
    },
    "skokloster-castle": {
        "floor_height": 0.0,
        "ceiling_height": 3.2,
        "bounds_min": [-8.0, 0.0, -8.0],
        "bounds_max": [8.0, 3.2, 8.0],
        "description": "Spacious Historic Castle Interior with stone obstacles",
    },
    "van-gogh-room": {
        "floor_height": 0.0,
        "ceiling_height": 2.6,
        "bounds_min": [-3.5, 0.0, -3.5],
        "bounds_max": [3.5, 2.6, 3.5],
        "description": "Compact Bedroom Interior with furniture obstacles",
    },
    "hssd_interior_01": {
        "floor_height": 0.0,
        "ceiling_height": 2.8,
        "bounds_min": [-6.0, 0.0, -6.0],
        "bounds_max": [6.0, 2.8, 6.0],
        "description": "HSSD High-Fidelity Synthetic Residential Interior",
    },
}


class SceneManager:
    """多场景管理器。"""

    def __init__(self, scene_search_dirs: Optional[List[Union[str, Path]]] = None):
        self.repo_root = get_repo_root()
        self.data_root = get_data_root()

        self.search_dirs: List[Path] = []
        if scene_search_dirs:
            self.search_dirs.extend([Path(d) for d in scene_search_dirs])

        # 默认搜索路径
        default_candidate_roots = [
            self.repo_root.parent / "robot" / "habitat-sim" / "data" / "versioned_data" / "habitat_test_scenes",
            Path("/home/zxf/MG08/hssd-scenes/scenes"),
            self.data_root / "assets" / "scenes",
        ]
        for r in default_candidate_roots:
            if r.exists() and r not in self.search_dirs:
                self.search_dirs.append(r)

        self.discovered_scenes: Dict[str, SceneMetadata] = {}
        self.discover_scenes()

    def discover_scenes(self) -> Dict[str, SceneMetadata]:
        """扫描所有可用场景文件并构建索引。"""
        self.discovered_scenes.clear()

        for s_dir in self.search_dirs:
            if not s_dir.exists():
                continue
            for glb in s_dir.glob("*.glb"):
                scene_name = glb.stem
                navmesh = glb.with_suffix(".navmesh")
                navmesh_resolved = navmesh if navmesh.exists() else None

                profile = KNOWN_SCENE_PROFILES.get(scene_name, {
                    "floor_height": 0.0,
                    "ceiling_height": 2.6,
                    "bounds_min": [-5.0, 0.0, -5.0],
                    "bounds_max": [5.0, 2.6, 5.0],
                    "description": f"Habitat Scene: {scene_name}",
                })

                meta = SceneMetadata(
                    scene_id=scene_name,
                    glb_path=glb.resolve(),
                    navmesh_path=navmesh_resolved.resolve() if navmesh_resolved else None,
                    floor_height=profile["floor_height"],
                    ceiling_height=profile["ceiling_height"],
                    bounds_min=profile["bounds_min"],
                    bounds_max=profile["bounds_max"],
                    description=profile["description"],
                )
                self.discovered_scenes[scene_name] = meta

        logger.info("SceneManager discovered %d scenes across %d search dirs.",
                    len(self.discovered_scenes), len(self.search_dirs))
        return self.discovered_scenes

    def get_scene(self, scene_id: str) -> SceneMetadata:
        """获取指定场景元数据。"""
        if scene_id not in self.discovered_scenes:
            raise KeyError(f"Scene '{scene_id}' not found. Available scenes: {list(self.discovered_scenes.keys())}")
        return self.discovered_scenes[scene_id]

    def list_scene_ids(self) -> List[str]:
        """返回所有已发现场景 ID 列表。"""
        return list(self.discovered_scenes.keys())

    def get_primary_scenes(self, count: int = 3) -> List[SceneMetadata]:
        """
        获取主要多场景集合（优先选择包含已校准 NavMesh 的代表性室内场景）。
        """
        # 优先返回官方 Habitat 室内场景与 HSSD 场景
        priority_keys = ["apartment_1", "skokloster-castle", "van-gogh-room"]
        selected: List[SceneMetadata] = []

        for k in priority_keys:
            if k in self.discovered_scenes:
                selected.append(self.discovered_scenes[k])

        for k, meta in self.discovered_scenes.items():
            if len(selected) >= count:
                break
            if meta not in selected:
                selected.append(meta)

        if not selected:
            # 构造默认 fallback
            fallback_meta = SceneMetadata(
                scene_id="apartment_1",
                glb_path=Path("apartment_1.glb"),
                floor_height=-1.60,
                ceiling_height=1.14,
            )
            selected.append(fallback_meta)

        return selected[:count]
