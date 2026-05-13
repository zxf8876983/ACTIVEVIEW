"""
Habitat 模拟器封装模块 —— habitat_runner.py
=============================================

功能：
    封装 Habitat-Sim 模拟器的初始化与核心操作。
    从 v2.0 迁移，保留相同接口。

约束：
    render_at() 只能在策略选择后调用，不能用于候选点评分。
"""

from typing import Optional
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from .geometry import yaw_to_forward


class HabitatRunner:
    """Habitat-Sim 运行器。"""

    def __init__(self, config: dict):
        self.config = config
        self._sim: Optional[habitat_sim.Simulator] = None
        self._init_simulator()

    def _init_simulator(self):
        habitat_cfg = self.config["habitat"]
        camera_cfg = self.config["camera"]
        scene_path = habitat_cfg["scene_path"]

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False

        agent_cfg = habitat_sim.AgentConfiguration()

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        rgb_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
        rgb_spec.hfov = camera_cfg["hfov_deg"]

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        depth_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
        depth_spec.hfov = camera_cfg["hfov_deg"]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        self._sim = habitat_sim.Simulator(cfg)

        navmesh_path = habitat_cfg.get("navmesh_path")
        if navmesh_path:
            self._sim.pathfinder.load_nav_mesh(navmesh_path)

    @property
    def sim(self) -> habitat_sim.Simulator:
        if self._sim is None:
            raise RuntimeError("模拟器未初始化")
        return self._sim

    @property
    def pathfinder(self):
        return self.sim.pathfinder

    def sample_navigable_point(self) -> np.ndarray:
        pt = self.pathfinder.get_random_navigable_point()
        return np.array(pt, dtype=np.float32)

    def is_navigable(self, point: np.ndarray) -> bool:
        return bool(self.pathfinder.is_navigable(point))

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        snapped = self.pathfinder.snap_point(point)
        return np.array(snapped, dtype=np.float32)

    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if self.pathfinder.find_path(path):
            return float(path.geodesic_distance)
        return float("inf")

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        """在指定位姿渲染 RGB 和 Depth 图像。

        ⚠ 只能在策略选择后调用，不能用于候选点评分。
        """
        agent = self.sim.agents[0]
        state = agent.get_state()
        state.position = position
        state.rotation = quat_from_angle_axis(yaw, np.array([0, 1, 0]))
        agent.set_state(state)

        obs = self.sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        depth = obs.get("depth_sensor")

        return {
            "rgb": np.array(rgb, dtype=np.uint8) if rgb is not None else None,
            "depth": np.array(depth, dtype=np.float32) if depth is not None else None,
            "position": position.copy(),
            "yaw": yaw,
        }

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __del__(self):
        self.close()
