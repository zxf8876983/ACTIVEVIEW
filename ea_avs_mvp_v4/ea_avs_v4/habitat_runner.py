"""
Habitat 模拟器封装模块 —— habitat_runner.py
=============================================

功能：
    封装 Habitat-Sim 模拟器的初始化与核心操作。
    从 v3.0 迁移，并增加 v4.0 ray casting 统一接口。

v4.0 关键变化：
    1. enable_physics 默认开启 —— cast_ray 依赖碰撞场景，physics 关闭时无命中
    2. 新增 cast_ray() 统一封装，隔离 Habitat-Sim 不同版本 ray API 差异

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
        self._init_simulator()

    def _init_simulator(self):
        habitat_cfg = self.config["habitat"]
        camera_cfg = self.config["camera"]
        scene_path = habitat_cfg["scene_path"]

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        # v4.0：ray casting 必须在碰撞场景中进行，physics 默认开启
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
        # 显式设置近/远裁剪面，保证 depth 为公制真实距离（默认远裁剪面过小会截断深度）
        depth_spec.clip_near = camera_cfg.get("clip_near", 0.01)
        depth_spec.clip_far = camera_cfg.get("clip_far", 10.0)

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        self._sim = habitat_sim.Simulator(cfg)

        navmesh_path = habitat_cfg.get("navmesh_path")
        if navmesh_path:
            self._sim.pathfinder.load_nav_mesh(navmesh_path)

        # 可复现性：为 pathfinder 内部 RNG 播种，保证随机导航点采样可复现
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

    # =========================================================================
    # 导航网格操作（与 v3.0 相同）
    # =========================================================================

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

    # =========================================================================
    # 渲染（与 v3.0 相同，仅可用于策略选择之后）
    # =========================================================================

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        """在指定位姿渲染 RGB 和 Depth 图像。

        ⚠ 只能在策略选择后调用，不能用于候选点评分。

        v4.0 约定修正：
            Habitat-Sim 相机在 identity 旋转下朝向 -Z，而评分几何模型
            （geometry.yaw_to_forward）约定 yaw 朝向 +Z。为让渲染图像与
            几何评分模型一致，渲染时对 yaw 补 π 旋转（相机前向 = +Z 模型方向）。
            该修正已通过深度图 + 射线投影对比实证验证。
        """
        agent = self.sim.agents[0]
        state = agent.get_state()
        state.position = position
        state.rotation = quat_from_angle_axis(yaw + np.pi, np.array([0, 1, 0]))
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

    # =========================================================================
    # v4.0 新增：统一射线接口
    # =========================================================================

    def cast_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Dict[str, Optional[object]]:
        """向场景中发射一条射线，返回最近命中信息。

        ⚠ 隔离 Habitat-Sim 不同版本 ray API，其他模块不直接依赖版本细节。
        ⚠ physics 必须开启，否则射线无命中。

        参数：
            origin: 射线起点，shape=(3,)。
            direction: 射线方向（无需单位化，内部会归一化），shape=(3,)。
            max_distance: 最大射程（沿射线方向的距离，米）。

        返回：
            {
                "has_hits": bool,
                "hit_point": np.ndarray 或 None,
                "hit_distance": float 或 None（世界单位，米）,
                "hit_object_id": int 或 None,
            }
        """
        # 归一化方向向量，保证 hit_distance 为世界单位米
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm < 1e-8:
            return {"has_hits": False, "hit_point": None,
                    "hit_distance": None, "hit_object_id": None}
        unit_dir = np.array(direction, dtype=np.float64) / dir_norm

        ray = geo.Ray(
            origin=np.array(origin, dtype=np.float64),
            direction=unit_dir,
        )
        # buffer_distance=0.0 显式指定，避免依赖版本默认值；起点附近命中由
        # 调用方的 min_hit_distance 容差处理
        results = self.sim.cast_ray(
            ray, max_distance=float(max_distance), buffer_distance=0.0
        )

        if not results.has_hits():
            return {"has_hits": False, "hit_point": None,
                    "hit_distance": None, "hit_object_id": None}

        # 取最近命中（ray_distance 为沿射线的距离，此处以几何距离为准）
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

    def __del__(self):
        self.close()