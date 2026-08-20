"""
Humanoid 代理类 —— humanoid_agent.py
===================================

功能：
    1. 在 Habitat-Sim 中实例化 KinematicHumanoid；
    2. 控制 Humanoid 的基座位置、朝向与关节姿态；
    3. 支持与 MotionPlayer 联动执行动作回放；
    4. 提取 Ground-Truth 3D 关节位置供数据集记录与评估。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import magnum as mn
    from omegaconf import DictConfig
    from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid
except ImportError:
    mn = None
    DictConfig = None
    KinematicHumanoid = None

from .humanoid_loader import HumanoidAssetBundle, resolve_humanoid_assets

logger = logging.getLogger(__name__)

# 标准 15 关节名称映射至 KinematicHumanoid link name
KEYPOINT_LINK_MAP = {
    "pelvis": "pelvis",
    "spine3": "spine3",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
}


class HumanoidAgent:
    """Habitat 室内场景中的拟人化 Agent 封装。"""

    def __init__(
        self,
        sim,
        config: Optional[Dict] = None,
        assets: Optional[HumanoidAssetBundle] = None,
    ):
        self.sim = sim
        self.config = config or {}
        self.humanoid_cfg = self.config.get("humanoid", {})
        self.assets = assets or resolve_humanoid_assets(self.config)

        self._agent: Optional[KinematicHumanoid] = None
        self._base_pos = np.zeros(3, dtype=np.float32)
        self._base_yaw = 0.0
        self._semantic_id = int(self.humanoid_cfg.get("semantic_id", 100))

    @property
    def agent(self) -> KinematicHumanoid:
        if self._agent is None:
            raise RuntimeError("HumanoidAgent is not loaded. Call load() first.")
        return self._agent

    @property
    def base_position(self) -> np.ndarray:
        return self._base_pos.copy()

    @property
    def base_yaw(self) -> float:
        return self._base_yaw

    def load(self) -> None:
        """在仿真环境中实例化 Humanoid。"""
        if KinematicHumanoid is None:
            raise RuntimeError("KinematicHumanoid not available. Please run in habitat conda env.")

        agent_config = DictConfig({
            "articulated_agent_urdf": self.assets.urdf_path,
            "motion_data_path": self.assets.motion_data_path,
            "auto_update_sensor_transform": True,
        })

        self._agent = KinematicHumanoid(agent_config, self.sim)
        self._agent.reconfigure()
        self._agent.update()

        # 默认放置在原点并处于 rest 姿态
        self._agent.base_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._agent.base_rot = 0.0
        self._agent.set_rest_position()
        logger.info("HumanoidAgent '%s' loaded into scene.", self.assets.avatar_name)

    def set_base_pose(
        self,
        position: Union[List[float], np.ndarray],
        yaw_rad: float = 0.0,
    ) -> None:
        """设置 Humanoid 在场景中的根基座位置与航向角。"""
        pos = np.asarray(position, dtype=np.float32)
        self._base_pos = pos
        self._base_yaw = float(yaw_rad)

        if self._agent is not None and mn is not None:
            self._agent.base_pos = mn.Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
            self._agent.base_rot = float(yaw_rad)
            self._agent.update()

    def apply_motion_frame(
        self,
        joints_pose: np.ndarray,
        root_transform_mat: Optional[np.ndarray] = None,
    ) -> None:
        """将 MotionPlayer 提取的当前帧姿态应用到仿真模型。"""
        if self._agent is None:
            raise RuntimeError("HumanoidAgent is not loaded.")

        joint_list = list(joints_pose.astype(float))

        offset_t = mn.Matrix4()
        if root_transform_mat is not None:
            offset_t = mn.Matrix4(root_transform_mat)

        self._agent.set_joint_transform(
            joint_list=joint_list,
            offset_transform=offset_t,
            base_transform=self._agent.base_transformation,
        )
        self._agent.update()

    def get_gt_joint_positions(self) -> Dict[str, np.ndarray]:
        """提取当前时刻各核心关节的 3D 世界坐标 Ground-Truth。"""
        if self._agent is None:
            return {}

        ao = self._agent.sim_obj
        positions: Dict[str, np.ndarray] = {}

        for kpt_name, link_name in KEYPOINT_LINK_MAP.items():
            if kpt_name == "pelvis":
                continue
            try:
                link_id = ao.get_link_id_from_name(link_name)
                if link_id is not None and int(link_id) >= 0:
                    node = ao.get_link_scene_node(link_id)
                    pos = np.array(node.transformation.translation, dtype=np.float32)
                    positions[kpt_name] = pos
            except Exception:
                pass

        # pelvis 由左右髋中点几何推导（若有）或读取根节点变换
        if "left_hip" in positions and "right_hip" in positions:
            positions["pelvis"] = (0.5 * (positions["left_hip"] + positions["right_hip"])).astype(np.float32)
        else:
            try:
                positions["pelvis"] = np.array(ao.transformation.translation, dtype=np.float32)
            except Exception:
                positions["pelvis"] = self._base_pos.copy()

        return positions
