"""
v8 仿真环境适配器 —— env_adapter.py
==================================

职责：
    1. 模块化复用 ea_avs_mvp_v7.environment.habitat_env.HabitatEnv 生命周期管理；
    2. 提供 Habitat Pathfinder 导航网格与可达性查询接口；
    3. 严格禁止在此处实现任何自主路径规划或强化学习逻辑。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 纯模块化复用 v7 仿真环境底座
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv

logger = logging.getLogger(__name__)


class V8EnvironmentAdapter:
    """v8 仿真环境适配器，封装 Habitat-Sim 与 Pathfinder。"""

    def __init__(
        self,
        habitat_cfg: Optional[Dict[str, Any]] = None,
        sensor_cfg: Optional[Dict[str, Any]] = None,
    ):
        self._env = HabitatEnv(habitat_cfg, sensor_cfg)
        self._sim = None

    @property
    def env(self) -> HabitatEnv:
        return self._env

    @property
    def sim(self):
        return self._sim

    @property
    def is_running(self) -> bool:
        return self._env.is_running and self._sim is not None

    def start(self):
        """启动 Habitat-Sim 仿真实例。"""
        self._sim = self._env.start()
        return self._sim

    def close(self):
        """安全关闭仿真环境。"""
        self._env.close()
        self._sim = None

    def is_navigable(self, point: Union[List[float], np.ndarray], max_snap_dist: float = 0.5) -> bool:
        """检查三维坐标点是否在 NavMesh 可行域内。"""
        if self._sim is None or not hasattr(self._sim, "pathfinder") or self._sim.pathfinder is None:
            # 无物理引擎或离线测试模式下默认通过
            return True

        pf = self._sim.pathfinder
        pt = np.asarray(point, dtype=np.float32)
        return bool(pf.is_navigable(pt, max_snap_dist))

    def snap_point(self, point: Union[List[float], np.ndarray]) -> Optional[List[float]]:
        """将任意三维点对齐/吸附到最近的合法 NavMesh 地面上。"""
        if self._sim is None or not hasattr(self._sim, "pathfinder") or self._sim.pathfinder is None:
            return list(point)

        pf = self._sim.pathfinder
        pt = np.asarray(point, dtype=np.float32)
        snapped = pf.snap_point(pt)
        if np.isnan(snapped).any():
            return None
        return [float(x) for x in snapped]

    def check_path_exists(
        self,
        start_point: Union[List[float], np.ndarray],
        end_point: Union[List[float], np.ndarray],
    ) -> Tuple[bool, float]:
        """检查从起点到终点在 NavMesh 上的可达性与测地线距离。"""
        if self._sim is None or not hasattr(self._sim, "pathfinder") or self._sim.pathfinder is None:
            dist = float(np.linalg.norm(np.array(start_point) - np.array(end_point)))
            return True, dist

        pf = self._sim.pathfinder
        if not pf.is_loaded:
            dist = float(np.linalg.norm(np.array(start_point) - np.array(end_point)))
            return True, dist

        try:
            import habitat_sim
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(start_point, dtype=np.float32)
            path.requested_end = np.asarray(end_point, dtype=np.float32)

            found = pf.find_path(path)
            if found:
                return True, float(path.geodesic_distance)
            else:
                return False, float("inf")
        except Exception as e:
            logger.warning("Error computing shortest path in pathfinder: %s", e)
            dist = float(np.linalg.norm(np.array(start_point) - np.array(end_point)))
            return True, dist
