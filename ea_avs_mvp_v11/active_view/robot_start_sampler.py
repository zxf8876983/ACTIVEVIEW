"""
机器人初始观察位姿随机采样器 —— robot_start_sampler.py (v11.3)
=============================================================

职责：
    1. 针对指定的人体空间位置，在多场景 Habitat 环境中随机采样合法的机器人初始位姿；
    2. 施加约束：
       - 处于场景可导航区域内部；
       - 距人体初始欧氏距离处于 [min_dist, max_dist]（默认 2.0m ~ 8.0m）；
       - 机器人主视线自动朝向人体 (Face-Human Yaw)；
       - 记录真实的初始观察距离与相对方位角；
    3. 输出标准 current_viewpoint 元数据字典。
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from ea_avs_mvp_v11.active_view.scene_manager import SceneMetadata, SceneManager

logger = logging.getLogger("robot_start_sampler")


class RobotStartSampler:
    """机器人初始观察位姿采样器。"""

    def __init__(
        self,
        scene_manager: Optional[SceneManager] = None,
        seed: int = 42,
    ):
        self.scene_mgr = scene_manager or SceneManager()
        self.rng = random.Random(seed)

    def sample_robot_start(
        self,
        human_placement: Dict[str, Any],
        min_dist: float = 2.0,
        max_dist: float = 8.0,
        camera_height: float = 1.25,
    ) -> Dict[str, Any]:
        """
        根据人体位置在物理场景中采样机器人初始观察位姿。

        返回:
            current_viewpoint 字典
        """
        scene_id = human_placement.get("scene_id", "apartment_1")
        scene_meta = self.scene_mgr.get_scene(scene_id)
        b_min = scene_meta.bounds_min
        b_max = scene_meta.bounds_max
        floor_y = scene_meta.floor_height

        hx, hy, hz = human_placement["human_position"]

        # 随机采样距离与相对方位角，确保边界剪裁后仍满足距离下界
        for _ in range(50):
            dist = self.rng.uniform(min_dist, max_dist)
            theta_deg = self.rng.uniform(0.0, 360.0)
            theta_rad = math.radians(theta_deg)

            rx = hx + dist * math.cos(theta_rad)
            rz = hz + dist * math.sin(theta_rad)

            rx = max(b_min[0] + 0.5, min(b_max[0] - 0.5, rx))
            rz = max(b_min[2] + 0.5, min(b_max[2] - 0.5, rz))

            dx = hx - rx
            dz = hz - rz
            actual_dist = math.sqrt(dx**2 + dz**2)
            if actual_dist >= min_dist - 0.3:
                break
        else:
            actual_dist = math.sqrt((hx - rx)**2 + (hz - rz)**2)

        ry = floor_y
        yaw_to_human = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
        angle_to_human = (math.degrees(math.atan2(rz - hz, rx - hx)) + 360.0) % 360.0

        current_viewpoint = {
            "position": [round(float(rx), 4), round(float(ry), 4), round(float(rz), 4)],
            "rotation": [round(float(yaw_to_human), 2), 0.0],
            "yaw": round(float(yaw_to_human), 2),
            "pitch": 0.0,
            "distance_to_human": round(float(actual_dist), 4),
            "angle_to_human": round(float(angle_to_human), 2),
            "camera_height": round(float(camera_height), 4),
        }
        return current_viewpoint
