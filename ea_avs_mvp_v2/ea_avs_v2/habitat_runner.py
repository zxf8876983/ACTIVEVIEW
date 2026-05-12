"""
Habitat 模拟器封装模块 —— habitat_runner.py
=============================================

功能：
    封装 Habitat-Sim 模拟器的初始化与核心操作。

重要约束（v2.0 核心规范）：
    - render_at() 只能在策略选择后调用，不能在候选点评分阶段使用
    - render_at() 的用途仅限于：保存可视化图像和计算 true_score
"""

from typing import Optional
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from .geometry import yaw_to_forward


class HabitatRunner:
    """Habitat-Sim 运行器，封装所有 Habitat 操作。"""

    def __init__(self, config: dict):
        """初始化 Habitat 模拟器。

        加载场景、导航网格，配置 RGB 和 Depth 传感器。

        参数：
            config: 配置字典（需要 habitat、camera 配置段）。
        """
        self.config = config
        self._sim: Optional[habitat_sim.Simulator] = None
        self._init_simulator()

    def _init_simulator(self):
        """内部方法：创建并配置 Habitat 模拟器。"""
        habitat_cfg = self.config["habitat"]
        camera_cfg = self.config["camera"]
        scene_path = habitat_cfg["scene_path"]

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False

        # Agent 配置
        agent_cfg = habitat_sim.AgentConfiguration()

        # RGB 传感器
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        rgb_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
        rgb_spec.hfov = camera_cfg["hfov_deg"]

        # Depth 传感器
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        depth_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
        depth_spec.hfov = camera_cfg["hfov_deg"]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        self._sim = habitat_sim.Simulator(cfg)

        # 加载导航网格
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
        """从导航网格随机采样一个可达点。"""
        pt = self.pathfinder.get_random_navigable_point()
        return np.array(pt, dtype=np.float32)

    def is_navigable(self, point: np.ndarray) -> bool:
        """判断点是否在导航网格上（机器人能否到达）。"""
        return bool(self.pathfinder.is_navigable(point))

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        """将点吸附到最近的导航网格可达点。"""
        snapped = self.pathfinder.snap_point(point)
        return np.array(snapped, dtype=np.float32)

    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """计算导航网格上两点之间的测地距离。

        返回 float("inf") 表示没有可达路径。
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if self.pathfinder.find_path(path):
            return float(path.geodesic_distance)
        return float("inf")

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        """在指定位姿渲染 RGB 和 Depth 图像。

        ⚠ v2.0 约束：此方法只能在策略选择后调用，不能在候选点评分阶段使用。

        参数：
            position: 机器人基座位置，shape=(3,)。
            yaw: 机器人朝向角（弧度）。

        返回：
            {"rgb": np.ndarray (H,W,4), "depth": np.ndarray (H,W), "position", "yaw"}
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
        """关闭模拟器，释放 GPU 资源。"""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __del__(self):
        self.close()
