"""
Habitat Humanoid 实体代理 —— humanoid_agent.py
=============================================

职责：
    1. 在 Habitat 物理世界中实例化 KinematicHumanoid (neutral_0)；
    2. 设置人体基座在场景中的初始位置与朝向；
    3. 接收 MotionPlayer 输出的关节姿态并驱动人形模型变形；
    4. 提取 16 个核心关节的 3D 世界坐标并封装为 HumanState。

边界约束：
    - 仅负责人体模型在仿真环境中的控制与状态提取，不干预机器人或视角决策。
"""

import logging
from pathlib import Path
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

from ea_avs_mvp_v7.core.paths import get_repo_root, get_data_root
from .human_state import HumanState

logger = logging.getLogger(__name__)

# 标准 15 关节名称映射至 KinematicHumanoid URDF 关节 link name
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


def resolve_humanoid_urdf_path(config: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
    """解析 Humanoid URDF 与运动数据路径。"""
    avatar_name = "neutral_0"
    cfg_root = None
    if config:
        avatar_name = config.get("avatar_name", avatar_name)
        cfg_root = config.get("assets_root")

    candidate_roots = []
    if cfg_root:
        p = Path(cfg_root)
        if not p.is_absolute():
            candidate_roots.append(get_repo_root().parent / p)
            candidate_roots.append(get_data_root() / p)
        else:
            candidate_roots.append(p)

    candidate_roots.extend([
        get_repo_root().parent / "robot" / "habitat-lab" / "data" / "versioned_data" / "habitat_humanoids",
        get_data_root() / "assets" / "habitat_humanoids",
    ])

    found_root = None
    for r in candidate_roots:
        if r.exists() and (r / avatar_name).exists():
            found_root = r
            break

    if not found_root:
        raise FileNotFoundError(f"Humanoid assets not found for avatar '{avatar_name}' in: {candidate_roots}")

    urdf_path = (found_root / avatar_name / f"{avatar_name}.urdf").resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"Humanoid URDF file missing: {urdf_path}")

    motion_path = (found_root / "walking_motion_processed_smplx.pkl").resolve()
    if not motion_path.exists():
        motion_path = (found_root / avatar_name / "motion_data.pkl").resolve()

    return urdf_path, motion_path if motion_path.exists() else urdf_path.parent


class HumanoidAgent:
    """Habitat 室内环境中的拟人化 Humanoid 实体代理。"""

    def __init__(
        self,
        sim,
        humanoid_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.sim = sim
        self.config = humanoid_cfg or {}
        self.urdf_path, self.default_motion_path = resolve_humanoid_urdf_path(self.config)

        self._agent: Optional[KinematicHumanoid] = None
        self._base_pos = np.zeros(3, dtype=np.float32)
        self._base_yaw = 0.0

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
        """在仿真环境中实例化 KinematicHumanoid。"""
        if KinematicHumanoid is None:
            raise RuntimeError("KinematicHumanoid not available. Please run in habitat conda env.")

        agent_config = DictConfig({
            "articulated_agent_urdf": str(self.urdf_path),
            "motion_data_path": str(self.default_motion_path),
            "auto_update_sensor_transform": True,
        })

        self._agent = KinematicHumanoid(agent_config, self.sim)
        self._agent.reconfigure()
        self._agent.update()

        # 默认放置在原点并置为 rest 姿态
        self._agent.base_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._agent.base_rot = 0.0
        self._agent.set_rest_position()
        logger.info("HumanoidAgent loaded: %s", self.urdf_path.name)

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
        """将当前帧姿态更新至人形模型关节与刚体。"""
        if self._agent is None:
            raise RuntimeError("HumanoidAgent is not loaded.")

        joint_list = list(joints_pose.astype(float))
        offset_t = mn.Matrix4(root_transform_mat) if root_transform_mat is not None else mn.Matrix4()

        self._agent.set_joint_transform(
            joint_list=joint_list,
            offset_transform=offset_t,
            base_transform=self._agent.base_transformation,
        )
        self._agent.update()

    def get_gt_joint_positions(self) -> Dict[str, List[float]]:
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

        # pelvis 由左右髋中点几何推导
        if "left_hip" in positions and "right_hip" in positions:
            positions["pelvis"] = 0.5 * (positions["left_hip"] + positions["right_hip"])
        else:
            try:
                positions["pelvis"] = np.array(ao.transformation.translation, dtype=np.float32)
            except Exception:
                positions["pelvis"] = self._base_pos.copy()

        return {k: [float(x) for x in v] for k, v in positions.items()}

    def get_human_state(
        self,
        frame_id: int = 0,
        timestamp: float = 0.0,
        action_class: str = "unknown",
        action_label: str = "unknown",
    ) -> HumanState:
        """获取当前时刻完整的 HumanState。"""
        gt_joints = self.get_gt_joint_positions()
        return HumanState(
            position=[float(x) for x in self._base_pos],
            orientation_yaw_rad=float(self._base_yaw),
            frame_id=frame_id,
            timestamp=timestamp,
            action_class=action_class,
            action_label=action_label,
            joint_positions_3d_world=gt_joints,
        )
