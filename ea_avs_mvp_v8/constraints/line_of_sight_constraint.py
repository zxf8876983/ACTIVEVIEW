"""
视线光线追踪无遮挡约束 —— line_of_sight_constraint.py
=====================================================

职责：
    1. 基于光线追踪 (Ray Cast) 校验机器人相机与人体目标部位 (Pelvis / Head) 之间的视线通视性；
    2. 判断机器人与人体之间是否存在实体墙壁、立柱或高大家具阻挡；
    3. 过滤视线被硬阻挡、完全无法观测到人体的无效视点。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter

logger = logging.getLogger(__name__)


class LineOfSightConstraint:
    """机器人与人体视线无遮挡光线追踪检查器。"""

    def __init__(
        self,
        env_adapter: Optional[V8EnvironmentAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env_adapter = env_adapter
        self.config = config or {}
        self.check_enabled = bool(self.config.get("check_line_of_sight", True))
        self.hit_buffer_m = float(self.config.get("hit_buffer_m", 0.25))

    def evaluate(
        self,
        viewpoint: CandidateViewpoint,
        human_position: Union[List[float], np.ndarray],
        human_target_offset: Optional[List[float]] = None,
    ) -> Tuple[bool, str]:
        """使用物理光线追踪检查视线是否通畅。

        参数：
            viewpoint: 待评估候选视点
            human_position: 人体基座坐标 [hx, hy, hz]
            human_target_offset: 目标检测点偏移 (默认 [0.0, 0.80, 0.0]，对应骨盆/躯干高程)

        返回：
            (is_clear: bool, reason: str)
        """
        if not self.check_enabled:
            return True, "valid"

        hx, hy, hz = [float(x) for x in human_position]
        offset = human_target_offset or [0.0, 0.80, 0.0]
        target_pos = np.array([hx + offset[0], hy + offset[1], hz + offset[2]], dtype=np.float32)

        rx, ry, rz = [float(x) for x in viewpoint.position]
        cam_height = float(viewpoint.camera_height)
        cam_pos = np.array([rx, ry + cam_height, rz], dtype=np.float32)

        ray_vec = target_pos - cam_pos
        dist = float(np.linalg.norm(ray_vec))

        if dist < 1e-4:
            return True, "valid"

        ray_dir = ray_vec / dist

        # 若在 Habitat 仿真环境中，执行物理光线追踪
        if self.env_adapter is not None and self.env_adapter.sim is not None:
            sim = self.env_adapter.sim
            try:
                import habitat_sim
                ray = habitat_sim.geo.Ray()
                ray.origin = cam_pos
                ray.direction = ray_dir

                results = sim.cast_ray(ray, max_distance=dist)
                if results.has_hits():
                    # 遍历命中结果，检查是否有早于人体位置的场景阻挡
                    for hit in results.hits:
                        hit_dist = float(hit.ray_distance)
                        # 如果在人体之前提前击中墙壁/家具 (距离小于 dist - buffer)
                        if hit_dist < (dist - self.hit_buffer_m):
                            logger.debug(
                                "Viewpoint %s line of sight blocked at dist %.2f m (human dist: %.2f m)",
                                viewpoint.viewpoint_id,
                                hit_dist,
                                dist,
                            )
                            return False, "line_of_sight_blocked"
            except Exception as e:
                logger.debug("Raycast check encountered error: %s", e)

        return True, "valid"
