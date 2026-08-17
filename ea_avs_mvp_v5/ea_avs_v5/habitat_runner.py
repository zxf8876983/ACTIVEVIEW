"""
Habitat 模拟器封装模块 —— habitat_runner.py
=============================================

功能：
    封装 Habitat-Sim 模拟器的初始化与核心操作。
    从 v4.0 迁移，并为 v5.0 增加：
        1. 可选 semantic sensor（调试用，确认 Humanoid 像素）
        2. attach_humanoid_manager 关联（Humanoid 在 episode 生命周期持续存在）

约束：
    render_at() 只能在策略选择后调用，不能用于候选点评分。
"""

from typing import Dict, Optional

import numpy as np
import habitat_sim
import habitat_sim.geo as geo
from habitat_sim.utils.common import quat_from_angle_axis

from .geometry import yaw_to_forward


class HabitatRunner:
    """Habitat-Sim 运行器。"""

    def __init__(self, config: dict):
        self.config = config
        self._sim: Optional[habitat_sim.Simulator] = None
        self._humanoid_manager = None
        self._init_simulator()

    def _init_simulator(self):
        habitat_cfg = self.config["habitat"]
        camera_cfg = self.config["camera"]
        scene_path = habitat_cfg["scene_path"]

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = habitat_cfg.get("enable_physics", True)

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
        depth_spec.clip_near = camera_cfg.get("clip_near", 0.01)
        depth_spec.clip_far = camera_cfg.get("clip_far", 10.0)

        sensor_specs = [rgb_spec, depth_spec]

        # v5.0：可选 semantic sensor（调试用，确认 Humanoid 像素/面积）
        if self.config["humanoid"].get("semantic_enabled", False):
            sem_spec = habitat_sim.CameraSensorSpec()
            sem_spec.uuid = "semantic_sensor"
            sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
            sem_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
            sem_spec.hfov = camera_cfg["hfov_deg"]
            sensor_specs.append(sem_spec)

        agent_cfg.sensor_specifications = sensor_specs
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        self._sim = habitat_sim.Simulator(cfg)

        navmesh_path = habitat_cfg.get("navmesh_path")
        if navmesh_path:
            self._sim.pathfinder.load_nav_mesh(navmesh_path)

        seed = self.config["project"].get("seed", 42)
        self._sim.pathfinder.seed(seed)

    @property
    def sim(self) -> habitat_sim.Simulator:
        if self._sim is None:
            raise RuntimeError("模拟器未初始化")
        return self._sim

    @property
    def pathfinder(self):
        return self.sim.pathfinder

    # =====================================================================
    # v5.0：Humanoid 关联
    # =====================================================================

    def attach_humanoid_manager(self, manager):
        """关联 HumanoidManager。

        Humanoid 在 episode 生命周期内持续存在于场景，render_at 会自然渲染。
        """
        self._humanoid_manager = manager

    @property
    def humanoid_manager(self):
        return self._humanoid_manager

    # =====================================================================
    # 导航网格操作（与 v4.0 相同）
    # =====================================================================

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

    # =====================================================================
    # 渲染（与 v4.0 相同，仅可用于策略选择之后）
    # =====================================================================

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        """在指定位姿渲染 RGB 和 Depth（及可选 semantic）。

        ⚠ 只能在策略选择后调用，不能用于候选点评分。
        """
        agent = self.sim.agents[0]
        state = agent.get_state()
        state.position = position
        state.rotation = quat_from_angle_axis(yaw + np.pi, np.array([0, 1, 0]))
        agent.set_state(state)

        obs = self.sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        depth = obs.get("depth_sensor")
        sem = obs.get("semantic_sensor")

        return {
            "rgb": np.array(rgb, dtype=np.uint8) if rgb is not None else None,
            "depth": np.array(depth, dtype=np.float32) if depth is not None else None,
            "semantic": np.array(sem, dtype=np.int32) if sem is not None else None,
            "position": position.copy(),
            "yaw": yaw,
        }

    # =====================================================================
    # 统一射线接口（与 v4.0 相同）
    # =====================================================================

    def cast_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Dict[str, Optional[object]]:
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm < 1e-8:
            return {"has_hits": False, "hit_point": None,
                    "hit_distance": None, "hit_object_id": None}
        unit_dir = np.array(direction, dtype=np.float64) / dir_norm

        ray = geo.Ray(
            origin=np.array(origin, dtype=np.float64),
            direction=unit_dir,
        )
        results = self.sim.cast_ray(
            ray, max_distance=float(max_distance), buffer_distance=0.0
        )

        if not results.has_hits():
            return {"has_hits": False, "hit_point": None,
                    "hit_distance": None, "hit_object_id": None}

        best_hit = None
        best_dist = float("inf")
        for hit in results.hits:
            hit_point = np.array(hit.point, dtype=np.float64)
            dist = float(np.linalg.norm(hit_point - origin))
            if dist < best_dist:
                best_dist = dist
                best_hit = hit

        if best_hit is None:
            return {"has_hits": False, "hit_point": None,
                    "hit_distance": None, "hit_object_id": None}

        return {
            "has_hits": True,
            "hit_point": np.array(best_hit.point, dtype=np.float64),
            "hit_distance": best_dist,
            "hit_object_id": int(best_hit.object_id),
        }

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._humanoid_manager = None

    def __del__(self):
        self.close()