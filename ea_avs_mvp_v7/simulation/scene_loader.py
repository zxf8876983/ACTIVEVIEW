"""
Habitat 场景与模拟器初始化器 —— scene_loader.py
=============================================

功能：
    1. 解析场景资产（如 apartment_1.glb）与 NavMesh；
    2. 配置物理引擎与传感器挂载；
    3. 启动并返回 Habitat-Sim Simulator 实例。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

from tools.motion_assets.data_paths import get_repo_root, get_data_root

logger = logging.getLogger(__name__)

CANDIDATE_SCENE_PATHS = [
    Path("/home/zxf/WorkSpace/code/code/robot/habitat-sim/data/versioned_data/habitat_test_scenes/apartment_1.glb"),
    get_data_root() / "assets" / "scenes" / "apartment_1.glb",
    get_repo_root().parent / "robot" / "habitat-sim" / "data" / "versioned_data" / "habitat_test_scenes" / "apartment_1.glb",
]


class HabitatSceneLoader:
    """Habitat 仿真场景加载器。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.habitat_cfg = config.get("habitat", {})
        self.camera_cfg = config.get("camera", {})

    def resolve_scene_path(self) -> Tuple[Path, Optional[Path]]:
        """解析场景 glb 与 navmesh 文件路径。"""
        cfg_path = self.habitat_cfg.get("scene_path")
        scene_p = None
        if cfg_path:
            p = Path(cfg_path)
            if p.is_absolute() and p.exists():
                scene_p = p
            elif (get_repo_root().parent / p).exists():
                scene_p = get_repo_root().parent / p
            elif (get_data_root() / p).exists():
                scene_p = get_data_root() / p

        if not scene_p:
            for cand in CANDIDATE_SCENE_PATHS:
                if cand.exists():
                    scene_p = cand
                    break

        if not scene_p or not scene_p.exists():
            raise FileNotFoundError(f"Habitat scene glb not found. Checked: {CANDIDATE_SCENE_PATHS}")

        # Navmesh 自动匹配
        navmesh_p = scene_p.with_suffix(".navmesh")
        if not navmesh_p.exists():
            navmesh_p = None

        return scene_p.resolve(), navmesh_p.resolve() if navmesh_p else None

    def create_simulator(self) -> Any:
        """配置并创建 Habitat-Sim Simulator。"""
        if habitat_sim is None:
            raise RuntimeError("habitat_sim is unavailable. Please run in habitat conda env.")

        scene_path, navmesh_path = self.resolve_scene_path()

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        sim_cfg.enable_physics = self.habitat_cfg.get("enable_physics", True)

        agent_cfg = habitat_sim.AgentConfiguration()

        cam_w = self.camera_cfg.get("width", 640)
        cam_h = self.camera_cfg.get("height", 480)
        cam_hfov = self.camera_cfg.get("hfov_deg", 90)
        cam_height = self.camera_cfg.get("camera_height", 1.2)
        clip_near = self.camera_cfg.get("clip_near", 0.01)
        clip_far = self.camera_cfg.get("clip_far", 10.0)

        # 1. RGB 相机
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [cam_h, cam_w]
        rgb_spec.position = [0.0, cam_height, 0.0]
        rgb_spec.hfov = cam_hfov

        # 2. Depth 相机
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [cam_h, cam_w]
        depth_spec.position = [0.0, cam_height, 0.0]
        depth_spec.hfov = cam_hfov
        depth_spec.clip_near = clip_near
        depth_spec.clip_far = clip_far

        sensor_specs = [rgb_spec, depth_spec]

        # 3. 语义相机 (可选)
        if self.config.get("humanoid", {}).get("semantic_enabled", False):
            sem_spec = habitat_sim.CameraSensorSpec()
            sem_spec.uuid = "semantic_sensor"
            sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_spec.resolution = [cam_h, cam_w]
            sem_spec.position = [0.0, cam_height, 0.0]
            sem_spec.hfov = cam_hfov
            sensor_specs.append(sem_spec)

        agent_cfg.sensor_specifications = sensor_specs
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        sim = habitat_sim.Simulator(cfg)

        if navmesh_path and navmesh_path.exists():
            sim.pathfinder.load_nav_mesh(str(navmesh_path))

        seed = self.config.get("project", {}).get("seed", 42)
        sim.pathfinder.seed(seed)

        logger.info("Habitat Simulator initialized with scene: %s", scene_path)
        return sim
