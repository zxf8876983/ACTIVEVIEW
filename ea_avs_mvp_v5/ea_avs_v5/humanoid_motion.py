"""
Humanoid 动作/姿态控制模块 —— humanoid_motion.py
=================================================

功能：
    统一控制 Humanoid 动作/姿态。

v5.0 至少支持：
    - standing：静态站立（使用 motion pkl 的 stop_pose）
    - walking：官方 walking 动画（验证 animation pipeline，借助
      HumanoidRearrangeController.calculate_walk_pose 推进 mocap 帧）

对于 research 姿态（sitting / bending / lying_fallen），当配置
strict_motion_loading: true 且对应资源缺失时，必须明确 FileNotFoundError，
绝不能用 standing 静默替代。

依赖的 Habitat-Lab API（参考 humanoids_tutorial / test_humanoid）：
    HumanoidRearrangeController(walk_pose_path)
        .reset(base_transformation)
        .calculate_walk_pose(relative_target)   # 推进 walk_mocap_frame
        .get_pose()                             # joints + offset + base
        .calculate_stop_pose()                  # 回到停止姿态
    KinematicHumanoid.set_joint_transform(joints, offset_transform, base_transform)

为后续 AMASS / SMPL-X 动作素材预留 MotionProvider 接口。
"""

import magnum as mn

# 通用人体姿态名集合
STANDING = "standing"
WALKING = "walking"
RESEARCH_POSES = ("sitting", "bending", "lying_fallen")


class MotionProvider:
    """动作提供者接口：为后续 AMASS / SMPL-X 动作素材预留。

    子类实现：
        load_pose(name): 加载并缓存指定姿态/动作
        get_pose(frame_idx): 返回指定帧的姿态（关节值 + 位移）
    """

    def load_pose(self, pose_name: str):
        raise NotImplementedError

    def get_pose(self, frame_idx: int):
        raise NotImplementedError


class HumanoidMotionController:
    """Humanoid 动作控制器门面。

    持有底层 HumanoidRearrangeController（由 manager 注入），对外提供
    apply_walk_step / apply_stop_pose 等面向动作的接口，并将"姿态名"映射到底层动画。

    注意：此控制器只负责计算关节/基底变换，不直接触碰 sim。
    """

    def __init__(
        self,
        rearrange_controller,
        supported_official_states=None,
        strict_motion_loading: bool = True,
        motion_fps: int = 30,
    ):
        """
        参数：
            rearrange_controller: HumanoidRearrangeController 实例（已加载 walk）。
            supported_official_states: 官方支持的状态列表，如 ["standing","walking"]。
            strict_motion_loading: 请求 research 姿态缺失资源时报错。
            motion_fps: 动作帧率。
        """
        self._controller = rearrange_controller
        self._supported = set(supported_official_states or [])
        self._strict = strict_motion_loading
        self._motion_fps = motion_fps
        self._current_state = STANDING
        self._frame = 0

    # ------------------------------------------------------------------
    # 官方姿态
    # ------------------------------------------------------------------

    def apply_stop_pose(self):
        """回到静态停止姿态（standing）。"""
        self._controller.calculate_stop_pose()
        self._current_state = STANDING
        self._frame = self._controller.walk_mocap_frame

    def apply_walk_step(self, relative_target: "mn.Vector3") -> int:
        """执行一步 walking 动画，推进 mocap 帧并返回当前帧。

        参数：
            relative_target: 相对人体根节点的目标位置（mn.Vector3）。

        返回：
            当前 mocap 帧索引。
        """
        self._controller.calculate_walk_pose(relative_target)
        self._current_state = WALKING
        self._frame = self._controller.walk_mocap_frame
        return self._frame

    # ------------------------------------------------------------------
    # 高层状态接口
    # ------------------------------------------------------------------

    def set_state(self, state_name: str):
        """设置 Humanoid 状态/姿态。

        支持 standing / walking。
        research 姿态（sitting/bending/lying_fallen）若配置了文件则加载，
        否则在 strict 模式下明确报误。
        """
        if state_name == STANDING:
            self.apply_stop_pose()
            return
        if state_name == WALKING:
            if WALKING not in self._supported:
                raise RuntimeError(
                    f"walking 不在 supported_official_states: {self._supported}")
            self._current_state = WALKING
            self._frame = self._controller.walk_mocap_frame
            return
        # 非官方状态（research 姿态）
        raise FileNotFoundError(
            f"Humanoid motion for research pose '{state_name}' not available. "
            f"请准备对应的 AMASS/SMPL-X 动作素材并配置 "
            f"humanoid.research_pose_files.{state_name}。"
            f"（strict_motion_loading=True，禁止用 standing 静默替代）"
        )

    def step(self, dt: float) -> int:
        """推进动作动画（walking 时前进一帧；standing 静止）。

        ⚠ walking 目标为**相对根节点的局部前进方向**（relative local walking
        target），仅用于 smoke test 验证 animation pipeline，不用于主 NBV
        实验中让人体自行移动。

        返回当前动作帧索引。
        """
        if self._current_state == WALKING:
            n = max(1, int(round(self._motion_fps * dt)))
            for _ in range(n):
                self.apply_walk_step(mn.Vector3(0.1, 0.0, 0.1))
        self._frame = self._controller.walk_mocap_frame
        return self._frame

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def current_frame(self) -> int:
        return self._frame

    def reset(self):
        """重置为 standing 静态姿态。"""
        self._controller.calculate_stop_pose()
        self._current_state = STANDING
        self._frame = 0