"""
Humanoid 动作/姿态控制模块 —— humanoid_motion.py
=================================================

功能：
    统一控制 Humanoid 动作/姿态。
"""

try:
    import magnum as mn
except ImportError:
    mn = None

STANDING = "standing"
WALKING = "walking"
RESEARCH_POSES = ("sitting", "bending", "lying_fallen")


class MotionProvider:
    """动作提供者接口：为后续 AMASS / SMPL-X 动作素材预留。"""

    def load_pose(self, pose_name: str):
        raise NotImplementedError

    def get_pose(self, frame_idx: int):
        raise NotImplementedError


class HumanoidMotionController:
    """Humanoid 动作控制器门面。"""

    def __init__(
        self,
        rearrange_controller,
        supported_official_states=None,
        strict_motion_loading: bool = True,
        motion_fps: int = 30,
    ):
        self._controller = rearrange_controller
        self._supported = set(supported_official_states or [])
        self._strict = strict_motion_loading
        self._motion_fps = motion_fps
        self._current_state = STANDING
        self._frame = 0

    def apply_stop_pose(self):
        self._controller.calculate_stop_pose()
        self._current_state = STANDING
        self._frame = getattr(self._controller, "walk_mocap_frame", 0)

    def apply_walk_step(self, relative_target) -> int:
        self._controller.calculate_walk_pose(relative_target)
        self._current_state = WALKING
        self._frame = getattr(self._controller, "walk_mocap_frame", 0)
        return self._frame

    def set_state(self, state_name: str):
        if state_name == STANDING:
            self.apply_stop_pose()
            return
        if state_name == WALKING:
            if WALKING not in self._supported:
                raise RuntimeError(f"walking 不在 supported_official_states: {self._supported}")
            self._current_state = WALKING
            self._frame = getattr(self._controller, "walk_mocap_frame", 0)
            return
        raise FileNotFoundError(
            f"Humanoid motion for research pose '{state_name}' not available. "
            f"请准备对应的 AMASS/SMPL-X 动作素材并配置 "
            f"humanoid.research_pose_files.{state_name}。"
            f"（strict_motion_loading=True，禁止用 standing 静默替代）"
        )

    def step(self, dt: float) -> int:
        if self._current_state == WALKING and mn is not None:
            n = max(1, int(round(self._motion_fps * dt)))
            for _ in range(n):
                self.apply_walk_step(mn.Vector3(0.1, 0.0, 0.1))
        self._frame = getattr(self._controller, "walk_mocap_frame", 0)
        return self._frame

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def current_frame(self) -> int:
        return self._frame

    def reset(self):
        self._controller.calculate_stop_pose()
        self._current_state = STANDING
        self._frame = 0
