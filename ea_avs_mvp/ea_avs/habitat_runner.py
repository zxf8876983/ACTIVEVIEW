"""
Habitat 模拟器封装模块 —— habitat_runner.py
=============================================

功能：
    封装 Habitat-Sim 模拟器的初始化和核心操作 API。

设计原则：
    MVP0.1 中约定「主脚本不直接调用 Habitat API」—— 所有与 Habitat 模拟器的
    交互都通过本模块完成。这样做的好处：
    1. 如果 Habitat API 版本升级，只需要修改这一个文件
    2. 便于后续切换到 Habitat-Lab 或其他模拟后端
    3. 方便单元测试时 mock

封装的主要功能：
    - 模拟器初始化（场景加载、navmesh 加载）
    - 导航网格查询（采样可达点、导航性检查、吸附到导航网格）
    - 测地距离计算（两点之间的最短路径距离）
    - 图像渲染（RGB + Depth）
"""

from typing import Optional
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from .geometry import yaw_to_forward


class HabitatRunner:
    """
    Habitat-Sim 运行器 —— 封装模拟器的所有操作。

    属性：
        config: dict —— 配置字典（包含 habitat、camera 等配置项）
        _sim: habitat_sim.Simulator —— Habitat 模拟器实例（内部使用）
    
    通过 property 暴露：
        sim: 模拟器实例
        pathfinder: 导航路径查找器（用于 navmesh 操作）
    """

    def __init__(self, config: dict):
        """
        初始化 Habitat 模拟器。

        初始化步骤：
            1. 从配置中读取场景路径（.glb 文件）和导航网格路径（.navmesh 文件）
            2. 配置相机参数（分辨率、视场角、安装高度）
            3. 创建 RGB 和 Depth 两个传感器
            4. 创建 habitat_sim.Simulator 实例
            5. 加载导航网格（如果指定了 navmesh_path）

        参数：
            config: 配置字典，必须包含以下键：
                - config["habitat"]["scene_path"] —— 场景文件路径
                - config["habitat"]["navmesh_path"] —— 导航网格路径（可选）
                - config["camera"]["width"] —— 图像宽度
                - config["camera"]["height"] —— 图像高度
                - config["camera"]["hfov_deg"] —— 水平视场角
                - config["camera"]["camera_height"] —— 相机安装高度
        """
        self.config = config
        self._sim: Optional[habitat_sim.Simulator] = None
        self._init_simulator()

    def _init_simulator(self):
        """
        创建并配置 Habitat 模拟器（内部方法）。

        Habitat-Sim API 说明：
            - SimulatorConfiguration: 设置场景 ID、物理引擎开关等
            - AgentConfiguration: 设置 agent（机器人）的传感器配置
            - CameraSensorSpec: 定义 RGB/Depth 相机的参数
            - Configuration: 组合 sim 配置和 agent 配置，传入 Simulator
        
        传感器配置：
            - color_sensor: RGB 彩色图像，4 通道（RGBA），uint8
            - depth_sensor: 深度图像，单通道，float32（单位：米）
            两个传感器共享相同的分辨率、位置和视场角。
        """
        habitat_cfg = self.config["habitat"]
        camera_cfg = self.config["camera"]
        scene_path = habitat_cfg["scene_path"]

        # ---------- Simulator 配置 ----------
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path          # 场景 .glb 文件路径
        sim_cfg.enable_physics = False         # MVP0.1 不需要物理引擎

        # ---------- Agent（机器人）配置 ----------
        agent_cfg = habitat_sim.AgentConfiguration()

        # --- RGB 彩色相机传感器 ---
        # uuid: "color_sensor" —— 通过这个名称获取图像数据
        # 输出格式：H x W x 4 (RGBA)，uint8，范围 [0, 255]
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        rgb_spec.position = [0.0, camera_cfg["camera_height"], 0.0]  # 相对于机器人基座
        rgb_spec.hfov = camera_cfg["hfov_deg"]

        # --- Depth 深度传感器 ---
        # uuid: "depth_sensor"
        # 输出格式：H x W，float32，单位：米
        # 值表示从相机到场景中物体表面的距离
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [camera_cfg["height"], camera_cfg["width"]]
        depth_spec.position = [0.0, camera_cfg["camera_height"], 0.0]
        depth_spec.hfov = camera_cfg["hfov_deg"]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        # ---------- 创建 Simulator ----------
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)

        # ---------- 加载导航网格 ----------
        navmesh_path = habitat_cfg.get("navmesh_path")
        if navmesh_path:
            self._sim.pathfinder.load_nav_mesh(navmesh_path)

    # =========================================================================
    # Properties（属性访问器）
    # =========================================================================

    @property
    def sim(self) -> habitat_sim.Simulator:
        """获取模拟器实例。"""
        if self._sim is None:
            raise RuntimeError("模拟器未初始化")
        return self._sim

    @property
    def pathfinder(self):
        """
        获取路径查找器（导航网格接口）。

        通过 pathfinder 可以：
            - get_random_navigable_point(): 随机采样可达点
            - is_navigable(point): 判断点是否可达
            - snap_point(point): 吸附到最近的导航网格
            - find_path(path): 计算两点之间的路径
        """
        return self.sim.pathfinder

    # =========================================================================
    # 导航网格操作
    # =========================================================================

    def sample_navigable_point(self) -> np.ndarray:
        """
        从导航网格中随机采样一个可达点。

        返回：
            shape=(3,) 的 numpy 数组，表示一个在导航网格上的随机位置。
            这个点一定在导航网格上（即机器人可以到达的位置）。

        用途：
            用于采样人体位置和机器人起始位置。
        """
        pt = self.pathfinder.get_random_navigable_point()
        return np.array(pt, dtype=np.float32)

    def is_navigable(self, point: np.ndarray) -> bool:
        """
        判断一个点是否在导航网格上（即机器人能否到达）。

        参数：
            point: 需要判断的点，shape=(3,)。

        返回：
            True 表示该点在导航网格上，机器人可以到达。
            False 表示该点不可达（可能在墙上、家具上或场景外）。
        """
        return bool(self.pathfinder.is_navigable(point))

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        """
        将一个点"吸附"到最近的导航网格点上。

        参数：
            point: 待吸附的点，shape=(3,)。

        返回：
            导航网格上最近的可达点。

        用途：
            在采样候选视角时，先计算理想的几何位置（围绕人体的圆上），
            然后用 snap_point 找到实际可达的最近位置。
            如果理想位置恰好在导航网格上，snap_point 返回原位置。
        """
        snapped = self.pathfinder.snap_point(point)
        return np.array(snapped, dtype=np.float32)

    # =========================================================================
    # 路径和距离计算
    # =========================================================================

    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """
        计算导航网格上两点之间的测地距离（沿 navigation mesh 的最短路径距离）。

        参数：
            start: 起点位置，shape=(3,)。
            goal: 终点位置，shape=(3,)。

        返回：
            测地距离（米）。如果两点之间没有有效路径，返回 float("inf")。

        注意：
            测地距离 ≠ 欧氏距离（直线距离）。测地距离是机器人实际
            需要行走的路径长度，考虑了墙壁、家具等障碍物。
            在室内场景中，测地距离通常大于欧氏距离。

        用途：
            用于评估机器人从当前位置移动到候选视角需要付出的运动代价。
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if self.pathfinder.find_path(path):
            return float(path.geodesic_distance)
        return float("inf")

    # =========================================================================
    # 图像渲染
    # =========================================================================

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        """
    在指定位置和朝向渲染 RGB 和 Depth 图像。

    参数：
        position: 机器人基座的三维位置，shape=(3,)。
        yaw: 机器人朝向角（弧度制）。

    返回：
        字典，包含以下字段：
        - rgb: shape=(H, W, 4) 的 uint8 数组（RGBA），范围 [0, 255]
               如果没有 RGB 传感器则返回 None
        - depth: shape=(H, W) 的 float32 数组，单位：米
                 如果没有 Depth 传感器则返回 None
        - position: 渲染时的机器人位置（备份）
        - yaw: 渲染时的机器人朝向（备份）

    实现步骤：
        1. 获取 agent（机器人）的当前状态
        2. 设置 agent 位置到指定位置
        3. 通过四元数旋转设置 agent 朝向（绕 Y 轴旋转 yaw 角度）
        4. 调用 get_sensor_observations() 获取传感器数据
        5. 从观测字典中提取 color_sensor 和 depth_sensor 数据

    注意：
        Habitat-Sim 中 agent 的默认朝向是 +Z 方向（yaw=0），
        因此我们只需要绕 Y 轴旋转 yaw 角度即可。
        """
        agent = self.sim.agents[0]
        state = agent.get_state()
        state.position = position
        # 将 yaw 角转换为四元数旋转
        # yaw=0 时四元数为单位四元数（朝向 +Z）
        state.rotation = quat_from_angle_axis(yaw, np.array([0, 1, 0]))
        agent.set_state(state)

        # ---------- 获取传感器观测 ----------
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
    # 生命周期管理
    # =========================================================================

    def close(self):
        """关闭模拟器，释放 GPU 资源。"""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __del__(self):
        """析构函数：确保模拟器被正确关闭。"""
        self.close()
