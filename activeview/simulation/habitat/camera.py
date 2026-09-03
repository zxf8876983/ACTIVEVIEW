"""文件用途：
    提供 Habitat 仿真坐标或相机辅助函数。

主要输入：
    - 场景、相机状态和仿真配置。
主要输出：
    - 仿真几何或相机姿态结果。
项目角色：
    - 属于 simulation.habitat 环境模块。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import quaternion


def camera_rotation_wxyz(
    agent_position: Sequence[float],
    base_position: Sequence[float],
    *,
    sensor_height_m: float = 1.1,
    target_height_m: float = 0.85,
) -> np.ndarray:
    """Return the exact rotation used by the RGB sensor state.

    The agent is placed on the floor plane defined by ``base_position``. The
    camera is then lifted by ``sensor_height_m`` and aimed at the target point
    above the human base. The result is in Habitat's ``[w, x, y, z]`` order.
    """
    agent = np.asarray(agent_position, dtype=np.float32).copy()
    base = np.asarray(base_position, dtype=np.float32)
    agent[1] = float(base[1])
    target = np.asarray(
        [base[0], float(base[1]) + target_height_m, base[2]],
        dtype=np.float32,
    )
    camera = agent + np.asarray([0.0, sensor_height_m, 0.0], dtype=np.float32)
    direction = target - camera
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    pitch = math.asin(float(direction[1]))
    rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector(
        [pitch, 0.0, 0.0]
    )
    return np.asarray(quaternion.as_float_array(rotation), dtype=np.float32)
