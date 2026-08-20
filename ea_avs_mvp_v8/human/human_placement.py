"""
Humanoid 场景放置与初始位姿接口 —— human_placement.py
===================================================

职责：
    1. 提供标准的人体初始放置位置采样接口 sample_position 与合法性校验 validate_position；
    2. v8.0 Phase 1 规范：返回经过场景物理地面高程验证的合法放置坐标；
    3. 预留接口规范：为未来扩展 NavMesh 多区域采样与障碍物碰撞检测预留签名。

科研边界与严禁事项：
    - ❌ Phase 1 严禁实现主动选择观察视角的算法；
    - ❌ 严禁实现强化学习或优化选点模型；
    - ❌ 仅作为纯物理放置 (Entity Placement) 的位置提供器。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v8.core.types import HumanPlacementPose

logger = logging.getLogger(__name__)

# 场景预设验证地面位姿 (Verified Ground Placement Poses per Scene)
DEFAULT_VERIFIED_PLACEMENTS: Dict[str, Dict[str, Any]] = {
    "apartment_1": {
        "position": [1.5, -1.60, 4.0],
        "yaw_deg": 0.0,
        "scale": 1.0,
        "floor_height": -1.60,
    },
    "default": {
        "position": [1.5, -1.60, 4.0],
        "yaw_deg": 0.0,
        "scale": 1.0,
        "floor_height": -1.60,
    },
}


class HumanPlacement:
    """人体场景放置与空间几何校验管理器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def sample_position(
        self,
        scene_id: str = "apartment_1",
        default_position: Optional[Union[List[float], np.ndarray]] = None,
        default_yaw_deg: float = 0.0,
        seed: Optional[int] = None,
    ) -> HumanPlacementPose:
        """采样或获取场景中 Humanoid 实体的合法初始放置位姿。

        参数：
            scene_id: 当前场景标识 (例如 'apartment_1')
            default_position: 显式指定的位置 [x, y, z] (可选)
            default_yaw_deg: 显式指定的偏航朝向角 (度)
            seed: 随机种子 (为未来随机采样预留)

        返回：
            HumanPlacementPose 结构体对象
        """
        scene_key = scene_id.split("/")[-1].replace(".glb", "") if scene_id else "default"
        preset = DEFAULT_VERIFIED_PLACEMENTS.get(scene_key, DEFAULT_VERIFIED_PLACEMENTS["default"])

        if default_position is not None:
            pos = [float(x) for x in default_position]
            yaw = float(default_yaw_deg)
        else:
            pos = list(preset["position"])
            yaw = float(preset.get("yaw_deg", 0.0))

        is_valid = self.validate_position(pos, scene_id=scene_id)

        pose = HumanPlacementPose(
            position=pos,
            yaw_deg=yaw,
            scale=float(preset.get("scale", 1.0)),
            is_valid=is_valid,
            scene_id=scene_id,
            metadata={
                "scene_key": scene_key,
                "floor_height": preset.get("floor_height", -1.60),
                "sampling_mode": "deterministic_preset" if default_position is None else "user_specified",
            },
        )
        logger.debug("Sampled human placement pose: %s (valid=%s)", pos, is_valid)
        return pose

    def validate_position(
        self,
        position: Union[List[float], np.ndarray],
        scene_id: str = "apartment_1",
        tolerance: float = 0.5,
    ) -> bool:
        """校验给定的人体位置在场景中是否位于合理的高程与边界内。"""
        pos = np.asarray(position, dtype=np.float32)
        if len(pos) != 3:
            return False

        # 校验高程是否在合理地面区间附近 (以 apartment_1 的 -1.60m 为例)
        scene_key = scene_id.split("/")[-1].replace(".glb", "") if scene_id else "default"
        preset = DEFAULT_VERIFIED_PLACEMENTS.get(scene_key, DEFAULT_VERIFIED_PLACEMENTS["default"])
        expected_floor = float(preset.get("floor_height", -1.60))

        if abs(pos[1] - expected_floor) > tolerance:
            logger.warning(
                "Human position Y (%.3f m) deviates from expected floor height (%.3f m by > %.2f m)",
                pos[1],
                expected_floor,
                tolerance,
            )
            return False

        return True
