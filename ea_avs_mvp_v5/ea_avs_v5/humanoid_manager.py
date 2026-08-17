"""
Humanoid 管理器模块 —— humanoid_manager.py
============================================

职责：
    加载 Humanoid、放入场景、设置位置/朝向/姿态/动作、reset、读取状态、清理。
    这是 v5.0 的核心新增模块，封装 Habitat-Lab 的 KinematicHumanoid。

依赖的 Habitat-Lab API（参考 humanoids_tutorial.ipynb / test_humanoid.py）：
    KinematicHumanoid(agent_config, sim)
        .reconfigure()      # 在场景中实例化（加载 URDF，设置为 KINEMATIC）
        .update()
        .base_pos / base_rot  # 设置脚底中心位置与朝向
        .set_rest_position()
        .sim_obj             # ManagedArticulatedObject，用于读链接变换与关节

说明：
    本类持有 runner.sim，直接在已有 HabitatRunner 主流程中嵌入 Humanoid，
    不将项目迁移为 Habitat Rearrange Environment。
"""

import logging
from typing import Optional

import magnum as mn
import numpy as np
from omegaconf import DictConfig

from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid
from habitat.articulated_agent_controllers import HumanoidRearrangeController

from .humanoid_assets import (
    HumanoidAssetBundle,
    resolve_humanoid_assets,
    validate_humanoid_assets,
)
from .humanoid_motion import HumanoidMotionController
from .humanoid_state import HumanoidState

logger = logging.getLogger(__name__)


class HumanoidManager:
    """Habitat Humanoid 管理器。

    参数：
        runner: HabitatRunner 实例（提供 sim）。
        config: 配置字典（含 humanoid 配置段）。
        assets: 可选 HumanoidAssetBundle；None 时自动 resolve+validate。
    """

    def __init__(
        self,
        runner,
        config: dict,
        assets: Optional[HumanoidAssetBundle] = None,
    ):
        self.config = config
        self.humanoid_cfg = config["humanoid"]
        self.sim = runner.sim

        # 资源定位与校验（缺失时明确报错，禁止退化为抽象 skeleton 渲染）
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

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def load(self):
        """加载 Humanoid 并放入场景。

        抛出异常：
            RuntimeError: Habitat-Lab 未暴露所需 Humanoid API。
        """
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
                f"加载 Humanoid '{self.assets.avatar_name}' 失败: {e}\n"
                f"请检查 Habitat-Lab 与 Habitat-Sim 版本配对以及 "
                f"URDF/GLB 资产。"
            ) from e

        # 动作控制器（官方 walking 链路）
        try:
            self._controller = HumanoidRearrangeController(
                self.assets.motion_data_path)
        except Exception as e:
            raise RuntimeError(
                f"加载 HumanoidRearrangeController 失败: {e}") from e

        self._motion = HumanoidMotionController(
            rearrange_controller=self._controller,
            supported_official_states=self.humanoid_cfg.get(
                "supported_official_states", ["standing", "walking"]),
            strict_motion_loading=self.humanoid_cfg.get(
                "strict_motion_loading", True),
        )

        # 初始置于原点（位置在 set_base_pose 中设置）
        self._agent.base_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._agent.base_rot = 0.0
        self.set_rest_position()
        # 同步 walking controller 的世界基准到当前 base
        self._sync_controller_base()

        logger.info("Humanoid object created: %s (robot sim id %d)",
                    self.assets.avatar_name, self._agent.get_robot_sim_id())

    def set_rest_position(self):
        """将 Humanoid 置于静止姿态（standing）。"""
        if self._agent is None:
            return
        self._agent.set_rest_position()
        self._agent.update()

    def reset(self):
        """重置为默认姿态（standing）并保持当前位置。"""
        if self._agent is None:
            return
        self.set_rest_position()
        if self._motion is not None:
            self._motion.reset()
        self._state.pose_name = "standing"
        self._state.motion_frame = None
        self._sync_controller_base()

    # ------------------------------------------------------------------
    # 位姿
    # ------------------------------------------------------------------

    def set_base_pose(self, position: np.ndarray, yaw: float):
        """设置 Humanoid 脚底中心位置与基准朝向。

        参数：
            position: 脚底中心（导航点），shape=(3,)。
            yaw: 基准朝向（弧度）。实际 mesh 朝向由 forward_axis 决定，
                 该 yaw 作为 human_yaw 约定（见 get_humanoid_forward_vector）。

        设置后会同步 walking controller 的世界基准。
        """
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
        """通过动作控制器设置姿态。

        standing / walking 直接支持；research 姿态缺失时严格报错。

        注意：
            standing 用 set_rest_position（保持当前 base 位置，不覆盖）；
            walking 才通过 controller 逐帧更新关节与基底变换，且进入前
            需同步 controller 世界基准（set_base_pose/reset 已完成）。
        """
        self._motion.set_state(pose_name)
        self._state.pose_name = pose_name
        self._state.motion_frame = self._motion.current_frame
        if pose_name == "standing":
            # 保持 set_base_pose 设置的位置，仅应用静止关节姿态
            self.set_rest_position()
        else:
            self._sync_controller_base()  # 进入 walking 前强制同步
            self._apply_controller_to_agent()

    def step_motion(self, dt: float):
        """推进动作动画（walking 时更新 mesh 姿态）。"""
        frame = self._motion.step(dt)
        self._state.motion_frame = frame
        self._apply_controller_to_agent()
        # 动画驱动后更新记录的 base 位置（get_state 从 agent 读取）
        self._sync_base_state()

    def _sync_controller_base(self):
        """将 walking controller 的世界基准与 Humanoid 当前 base 对齐。

        Habitat 官方：controller.reset(base_transformation) 建立控制器当前
        世界基准，避免后续 walking 相对位移时出现瞬移到原点的问题。
        调用时机：load / set_base_pose / reset / 进入 walking 前。
        """
        if self._controller is None or self._agent is None:
            return
        self._controller.reset(self._agent.base_transformation)

    def _sync_base_state(self):
        """从 agent 读取当前脚底中心位置/朝向以同步 GT 状态。"""
        if self._agent is not None:
            bpos = np.asarray(self._agent.base_pos, dtype=np.float32).copy()
            self._state.base_position = bpos
            self._state.base_yaw = float(self._agent.base_rot)
            self._state.actual_base_yaw = float(self._agent.base_rot)

    def _apply_controller_to_agent(self):
        """将 controller 计算出的关节/基底变换应用到 Humanoid mesh。"""
        if self._controller is None or self._agent is None:
            return
        new_pose = self._controller.get_pose()
        new_joints = new_pose[:-32]
        new_transform_base = new_pose[-16:]
        new_transform_offset = new_pose[-32:-16]
        # 全部为 0 表示未设置人体关节，跳过
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

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def sim_obj(self):
        return self._agent.sim_obj if self._agent is not None else None

    @property
    def motion_controller(self) -> Optional[HumanoidMotionController]:
        return self._motion

    def get_humanoid_forward_vector(self) -> np.ndarray:
        """计算 Humanoid 在 X-Z 平面上的前向向量（模型 forward 轴）。

        v5.0 前向校准：
            不再假设 base_yaw=0 即 mesh 朝 +Z。human_yaw 表示"人面向的
            X-Z 平面方向"，而 mesh 的真实前向基准轴由配置 humanoid.forward_axis
            （+Z/-Z/+X/-X，由 debug_humanoid_forward_axis.py 一次性确认）决定。

            统一约定：forward = RotY(base_yaw) · world_axis(forward_axis)
            其中 base_yaw 由 set_base_pose 设置，与 human_yaw 一致。

        所有模型的 forward convention 统一在此函数处理，禁止在别处散布
        magic +pi / -pi/2。
        """
        axis_name = self.humanoid_cfg.get("forward_axis", "+Z")
        axis_map = {
            "+Z": np.array([0.0, 0.0, 1.0]),
            "-Z": np.array([0.0, 0.0, -1.0]),
            "+X": np.array([1.0, 0.0, 0.0]),
            "-X": np.array([-1.0, 0.0, 0.0]),
        }
        if axis_name not in axis_map:
            raise ValueError(
                f"humanoid.forward_axis 不合法: {axis_name}，允许 {list(axis_map)}")
        local = axis_map[axis_name]
        base_yaw = self._state.base_yaw
        cos, sin = np.cos(base_yaw), np.sin(base_yaw)
        # RotY(base_yaw): x' = cos*x + sin*z ; z' = -sin*x + cos*z
        wx = cos * local[0] + sin * local[2]
        wz = -sin * local[0] + cos * local[2]
        return np.array([wx, 0.0, wz])

    def get_humanoid_object_ids(self) -> set:
        """返回 Humanoid 相关的 object id 集合（用于遮挡来源判定）。

        v5.0：从 current Habitat API 读取：
            - 人体 articulated object 自身的 object_id
            - link_object_ids 中映射到的各 link object id
        命中这些 id 的射线视为命中"人体自身"（self-occlusion / 目标表面）。
        """
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
            lo = obj.link_object_ids  # dict link_id -> object_id
            if lo:
                for _, oid in lo.items():
                    ids.add(int(oid))
        except Exception:
            pass
        return ids

    def get_state(self) -> HumanoidState:
        """返回当前 GT 状态副本（base 位置从 agent 实时读取）。"""
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
        """清理 Humanoid（移除 articulated object）。"""
        if self._agent is not None:
            try:
                self.sim.get_rigid_object_manager()  # no-op；humanoid 由 sim 管理
            except Exception:
                pass
            self._agent = None
            self._controller = None
            self._motion = None