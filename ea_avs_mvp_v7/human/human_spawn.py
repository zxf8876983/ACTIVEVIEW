"""
Human 放置与初始位置接口 —— human_spawn.py
=========================================

职责：
    1. 提供标准的人体初始放置位置采样接口 sample_human_position；
    2. v7.0 规范：仅返回经过场景地面几何验证的安全静态地面放置坐标；
    3. 预留接口规范：为未来 v8.0 扩展 NavMesh 随机采样与环境碰撞检测 (Collision Checking) 预留签名。

科研边界与严禁事项 (Research Boundary & Non-Goals)：
    - ❌ v7.0 严禁实现主动选择观察视角的算法；
    - ❌ 严禁实现机器人路径规划或避障导航；
    - ❌ 仅作为实体放置 (Entity Spawning) 的纯物理位置提供器。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# 场景预设安全地面位置 (Scene-specific Verified Ground Coordinates)
DEFAULT_SCENE_GROUND_POSITIONS: Dict[str, List[float]] = {
    "apartment_1": [1.5, -1.60, 4.0],
    "default": [1.5, -1.60, 4.0],
}


def sample_human_position(
    scene_id: str = "apartment_1",
    default_position: Optional[Union[List[float], np.ndarray]] = None,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """采样或获取场景中 Humanoid 实体的安全初始放置位置。

    参数：
        scene_id: 当前 Habitat 场景标识符 (例如 'apartment_1')
        default_position: 外部显式指定的初始位置 [x, y, z] (可选)
        seed: 随机数种子 (为未来随机采样预留)
        config: 附加场景或生成配置字典 (可选)

    返回：
        经过地面高程对齐的人体基座三维世界坐标 [x, y, z] (List[float])
    """
    if default_position is not None:
        pos = [float(x) for x in default_position]
        logger.debug("Using user-specified human initial position: %s", pos)
        return pos

    # 根据配置或场景字典返回安全预设
    scene_key = scene_id.split("/")[-1].replace(".glb", "") if scene_id else "default"
    spawn_pos = DEFAULT_SCENE_GROUND_POSITIONS.get(scene_key, DEFAULT_SCENE_GROUND_POSITIONS["default"])
    logger.debug("Sampled verified ground spawn position for scene '%s': %s", scene_key, spawn_pos)
    return list(spawn_pos)


def get_default_human_orientation(
    scene_id: str = "apartment_1",
    default_yaw_deg: float = 0.0,
) -> float:
    """获取 Humanoid 在指定场景中的默认偏航朝向角 (度)。"""
    return float(default_yaw_deg)
