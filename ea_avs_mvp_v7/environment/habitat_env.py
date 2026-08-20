"""
Habitat 仿真环境封装 —— habitat_env.py
=====================================

职责：
    1. 初始化 Habitat-Sim Simulator 实例；
    2. 加载室内静态场景与 NavMesh；
    3. 挂载机器人 RGB-D 传感器组并管理仿真步进 (step) 与生命周期；
    4. 提供安全重置 (reset) 与资源释放 (close) 接口。

边界约束：
    - 严禁在此实现 SLAM、建图、路径规划或避障算法。
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

from .scene_manager import SceneManager

logger = logging.getLogger(__name__)


class HabitatEnv:
    """Habitat 仿真环境统一封装。"""

    def __init__(
        self,
        habitat_cfg: Optional[Dict[str, Any]] = None,
        sensor_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.habitat_cfg = habitat_cfg or {}
        self.sensor_cfg = sensor_cfg or {}
        self.scene_mgr = SceneManager(self.habitat_cfg.get("scene_path"))
        self._sim: Optional[Any] = None

    @property
    def sim(self) -> Any:
        if self._sim is None:
            raise RuntimeError("HabitatEnv is not started. Call start() or reset() first.")
        return self._sim

    @property
    def scene_id(self) -> str:
        return self.scene_mgr.scene_id

    def start(self) -> Any:
        """启动 Habitat-Sim Simulator。"""
        if habitat_sim is None:
            raise RuntimeError("habitat_sim is unavailable. Please run in habitat conda environment.")

        if self._sim is not None:
            return self._sim

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self.scene_mgr.scene_path)
        sim_cfg.enable_physics = self.habitat_cfg.get("enable_physics", True)

        agent_cfg = habitat_sim.AgentConfiguration()

        cam_w = int(self.sensor_cfg.get("width", 640))
        cam_h = int(self.sensor_cfg.get("height", 480))
        cam_hfov = float(self.sensor_cfg.get("hfov_deg", 90.0))
        cam_height = float(self.sensor_cfg.get("camera_height", 1.2))
        clip_near = float(self.sensor_cfg.get("clip_near", 0.01))
        clip_far = float(self.sensor_cfg.get("clip_far", 10.0))

        # 1. RGB 相机 SensorSpec
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [cam_h, cam_w]
        rgb_spec.position = [0.0, cam_height, 0.0]
        rgb_spec.hfov = cam_hfov

        # 2. Depth 相机 SensorSpec
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [cam_h, cam_w]
        depth_spec.position = [0.0, cam_height, 0.0]
        depth_spec.hfov = cam_hfov
        depth_spec.clip_near = clip_near
        depth_spec.clip_far = clip_far

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        self._sim = habitat_sim.Simulator(cfg)

        if self.scene_mgr.navmesh_path and self.scene_mgr.navmesh_path.exists():
            self._sim.pathfinder.load_nav_mesh(str(self.scene_mgr.navmesh_path))

        seed = int(self.habitat_cfg.get("seed", 42))
        self._sim.pathfinder.seed(seed)

        logger.info("HabitatEnv initialized with scene: %s", self.scene_mgr.scene_path)
        return self._sim

    def step(self, time_step: float = 1.0 / 30.0) -> None:
        """物理世界步进。"""
        if self._sim is not None:
            self._sim.step_world(time_step)

    def reset(self) -> Any:
        """重置环境。"""
        if self._sim is None:
            return self.start()
        self._sim.reset()
        return self._sim

    def close(self) -> None:
        """关闭模拟器并释放资源。"""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            logger.info("HabitatEnv simulator closed.")
