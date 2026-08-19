"""
Humanoid 管理器模块 —— humanoid_manager.py
============================================

功能：
    加载 Humanoid、放入场景、设置位置/朝向/姿态/动作、reset、读取状态、清理。
"""

import logging
from typing import Optional

import numpy as np
try:
    import magnum as mn
    from omegaconf import DictConfig
    from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid
    from habitat.articulated_agent_controllers import HumanoidRearrangeController
except ImportError:
    mn = None
    DictConfig = None
    KinematicHumanoid = None
    HumanoidRearrangeController = None

from .humanoid_assets import (
    HumanoidAssetBundle,
    resolve_humanoid_assets,
    validate_humanoid_assets,
)
from .humanoid_motion import HumanoidMotionController
from .humanoid_state import HumanoidState

logger = logging.getLogger(__name__)


class HumanoidManager:
    """Habitat Humanoid 管理器。"""

    def __init__(
        self,
        runner,
        config: dict,
        assets: Optional[HumanoidAssetBundle] = None,
    ):
        self.config = config
        self.humanoid_cfg = config["humanoid"]
        self.sim = runner.sim

        self.assets = assets or resolve_humanoid_assets(config)
        validate_humanoid_assets(self.assets)

        self._agent: Optional[KinematicHumanoid] = None
        self._controller: Optional[HumanoidRearrangeController] = None
        self._motion: Optional[HumanoidMotionController] = None
        self._state = HumanoidState(
            base_position=np.zeros(3, dtype=np.float32),
            base_yaw=0.0,
            pose_name=self.humanoid_cfg.get("default_pose", "standing"),
            motion_frame=None,
            semantic_id=int(self.humanoid_cfg.get("semantic_id", 100)),
        )

    def load(self):
        if KinematicHumanoid is None:
            raise RuntimeError("Habitat-Lab KinematicHumanoid 不可用，请在 habitat conda 环境中运行。")

        agent_config = DictConfig({
            "articulated_agent_urdf": self.assets.urdf_path,
            "motion_data_path": self.assets.motion_data_path,
            "auto_update_sensor_transform": True,
        })

        try:
            self._agent = KinematicHumanoid(agent_config, self.sim)
            self._agent.reconfigure()
            self._agent.update()
        except Exception as e:
            raise RuntimeError(
                f"加载 Humanoid '{self.assets.avatar_name}' 失败: {e}"
            ) from e

        try:
            self._controller = HumanoidRearrangeController(self.assets.motion_data_path)
        except Exception as e:
            raise RuntimeError(f"加载 HumanoidRearrangeController 失败: {e}") from e

        self._motion = HumanoidMotionController(
            rearrange_controller=self._controller,
            supported_official_states=self.humanoid_cfg.get(
                "supported_official_states", ["standing", "walking"]),
            strict_motion_loading=self.humanoid_cfg.get("strict_motion_loading", True),
        )

        self._agent.base_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._agent.base_rot = 0.0
        self.set_rest_position()
        self._sync_controller_base()

        logger.info("Humanoid object created: %s", self.assets.avatar_name)

    def set_rest_position(self):
        if self._agent is None:
            return
        self._agent.set_rest_position()
        self._agent.update()

    def reset(self):
        if self._agent is None:
            return
        self.set_rest_position()
        if self._motion is not None:
            self._motion.reset()
        self._state.pose_name = "standing"
        self._state.motion_frame = None
        self._sync_controller_base()

    def set_base_pose(self, position: np.ndarray, yaw: float):
        self._agent.base_pos = mn.Vector3(
            float(position[0]), float(position[1]), float(position[2]))
        self._agent.base_rot = float(yaw)
        self._state.base_position = np.asarray(position, dtype=np.float32)
        self._state.base_yaw = float(yaw)
        self._state.requested_yaw = float(yaw)
        self._state.actual_base_yaw = float(self._agent.base_rot)
        self._update()
        self._sync_controller_base()

    def set_pose(self, pose_name: str):
        self._motion.set_state(pose_name)
        self._state.pose_name = pose_name
        self._state.motion_frame = self._motion.current_frame
        if pose_name == "standing":
            self.set_rest_position()
        else:
            self._sync_controller_base()
            self._apply_controller_to_agent()

    def step_motion(self, dt: float):
        frame = self._motion.step(dt)
        self._state.motion_frame = frame
        self._apply_controller_to_agent()
        self._sync_base_state()

    def _sync_controller_base(self):
        if self._controller is None or self._agent is None:
            return
        self._controller.reset(self._agent.base_transformation)

    def _sync_base_state(self):
        if self._agent is not None:
            bpos = np.asarray(self._agent.base_pos, dtype=np.float32).copy()
            self._state.base_position = bpos
            self._state.base_yaw = float(self._agent.base_rot)
            self._state.actual_base_yaw = float(self._agent.base_rot)

    def _apply_controller_to_agent(self):
        if self._controller is None or self._agent is None:
            return
        new_pose = self._controller.get_pose()
        new_joints = new_pose[:-32]
        new_transform_base = new_pose[-16:]
        new_transform_offset = new_pose[-32:-16]
        if np.asarray(new_transform_offset).sum() != 0:
            vecs_offset = [
                mn.Vector4(new_transform_offset[i * 4:(i + 1) * 4])
                for i in range(4)
            ]
            vecs_base = [
                mn.Vector4(new_transform_base[i * 4:(i + 1) * 4])
                for i in range(4)
            ]
            self._agent.set_joint_transform(
                new_joints,
                mn.Matrix4(*vecs_offset),
                mn.Matrix4(*vecs_base),
            )
            self._agent.update()

    def _update(self):
        if self._agent is not None:
            self._agent.update()

    @property
    def sim_obj(self):
        return self._agent.sim_obj if self._agent is not None else None

    @property
    def motion_controller(self) -> Optional[HumanoidMotionController]:
        return self._motion

    def get_humanoid_forward_vector(self) -> np.ndarray:
        axis_name = self.humanoid_cfg.get("forward_axis", "+Z")
        axis_map = {
            "+Z": np.array([0.0, 0.0, 1.0]),
            "-Z": np.array([0.0, 0.0, -1.0]),
            "+X": np.array([1.0, 0.0, 0.0]),
            "-X": np.array([-1.0, 0.0, 0.0]),
        }
        if axis_name not in axis_map:
            raise ValueError(f"humanoid.forward_axis 不合法: {axis_name}")
        local = axis_map[axis_name]
        base_yaw = self._state.base_yaw
        cos, sin = np.cos(base_yaw), np.sin(base_yaw)
        wx = cos * local[0] + sin * local[2]
        wz = -sin * local[0] + cos * local[2]
        return np.array([wx, 0.0, wz])

    def get_humanoid_object_ids(self) -> set:
        ids = set()
        if self._agent is None:
            return ids
        obj = self._agent.sim_obj
        try:
            root_id = int(obj.object_id)
            ids.add(root_id)
        except Exception:
            pass
        try:
            lo = obj.link_object_ids
            if lo:
                for oid in lo.keys():
                    ids.add(int(oid))
        except Exception:
            pass
        return ids

    def get_humanoid_link_metadata(self) -> dict:
        meta = {}
        if self._agent is None:
            return meta
        obj = self._agent.sim_obj
        try:
            lo = obj.link_object_ids
            lt2o = obj.link_ids_to_object_ids
        except Exception:
            return meta
        for oid, lid in lo.items():
            try:
                name = obj.get_link_name(lid)
            except Exception:
                name = None
            if name is None:
                continue
            meta[name] = {
                "link_name": name,
                "link_id": int(lid),
                "object_id": int(oid),
            }
        if lt2o is not None:
            for lid, oid in lt2o.items():
                try:
                    name = obj.get_link_name(lid)
                except Exception:
                    name = None
                if name is None or name in meta:
                    continue
                meta[name] = {
                    "link_name": name,
                    "link_id": int(lid),
                    "object_id": int(oid),
                }
        return meta

    def assign_semantic_id_to_links(self, semantic_id: int) -> int:
        if self._agent is None:
            return 0
        obj = self._agent.sim_obj
        count = 0
        try:
            lo = obj.link_object_ids
            link_ids = set(lo.values())
        except Exception:
            return 0
        for lid in link_ids:
            try:
                node = obj.get_link_scene_node(lid)
                node.semantic_id = semantic_id
                count += 1
            except Exception:
                pass
            try:
                for vn in obj.get_link_visual_nodes(lid):
                    vn.semantic_id = semantic_id
            except Exception:
                pass
        try:
            obj.root_scene_node.semantic_id = semantic_id
        except Exception:
            pass
        return count

    def get_state(self) -> HumanoidState:
        self._sync_base_state()
        return HumanoidState(
            base_position=self._state.base_position,
            base_yaw=self._state.base_yaw,
            requested_yaw=self._state.requested_yaw,
            actual_base_yaw=self._state.actual_base_yaw,
            pose_name=self._state.pose_name,
            motion_frame=self._state.motion_frame,
            semantic_id=self._state.semantic_id,
        )

    def close(self):
        if self._agent is not None:
            self._agent = None
            self._controller = None
            self._motion = None
