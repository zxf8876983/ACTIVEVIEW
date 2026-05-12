"""
到达后真实评估模块 —— true_evaluator.py
=========================================

功能：
    在策略选定并渲染图像后，对选中位姿进行真实评估。

v2.0 约束：
    - 必须在策略选定并调用 render_at() 后计算
    - 字段名必须与 pred 分开（_true 后缀）
    - 不能参与策略选择

v2.0 简化说明：
    由于抽象 skeleton 未真实渲染进图像，true_evaluator 目前
    仍然使用几何 FOV 计算 true 指标。但这满足了两个关键要求：
    1. 必须在 render_at() 之后调用
    2. 字段名与 pred 严格区分

后续版本（v2.4+）将使用 depth 图像进行遮挡判断。
"""

from typing import Dict
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .skeleton import KEYPOINT_GROUPS


class TrueEvaluator:
    """到达后真实评估器 —— 在渲染图像后评估视角质量。"""

    def __init__(self, config: dict):
        """初始化真实评估器。

        参数：
            config: 需要 camera、visibility、score_weights、
                   true_score、distance_score 配置段。
        """
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.score_weights = config["score_weights"]
        self.true_cfg = config["true_score"]
        self.dist_cfg = config["distance_score"]

    def score_view_true(
        self,
        obs: dict,
        view_pos: np.ndarray,
        view_yaw: float,
        human_base_pos: np.ndarray,
        human_skeleton: Dict[str, np.ndarray],
    ) -> dict:
        """对已渲染的视角进行真实评估。

        ⚠ 此函数必须在策略选择并调用 render_at() 之后调用。
        ⚠ 此函数的结果不应用于策略选择。

        参数：
            obs: render_at() 返回的观测字典（含 rgb 和 depth）。
            view_pos: 视角位置，shape=(3,)。
            view_yaw: 视角朝向（弧度）。
            human_base_pos: 人体基座位置。
            human_skeleton: 人体骨架字典。

        返回：
            {"S_kp_true", "S_center_true", "S_dist_true", "Q_true",
             "torso_visibility_true", "lower_body_visibility_true",
             "head_visibility_true"}
        """
        # 使用几何 FOV 判断关键点可见性
        visible_kps = []
        invisible_kps = []
        horizontal_angles = []

        for name, pos in human_skeleton.items():
            result = angle_in_camera_fov(
                camera_base_pos=view_pos,
                camera_yaw=view_yaw,
                point=pos,
                hfov_deg=self.camera_cfg["hfov_deg"],
                vfov_deg=self.camera_cfg["vfov_deg"],
                camera_height=self.camera_cfg["camera_height"],
                min_depth=self.vis_cfg["min_depth"],
                max_depth=self.vis_cfg["max_depth"],
            )
            if result["in_fov"]:
                visible_kps.append(name)
            else:
                invisible_kps.append(name)
            horizontal_angles.append(result["horizontal_angle"])

        # 分组可见率
        def group_vis(group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            visible_count = sum(1 for kp in group_kps if kp in visible_kps)
            return visible_count / len(group_kps)

        torso_vis = group_vis("torso")
        lower_body_vis = group_vis("lower_body")
        head_vis = group_vis("head")

        # S_kp_true
        S_kp_true = (
            self.score_weights["torso"] * torso_vis
            + self.score_weights["lower_body"] * lower_body_vis
            + self.score_weights["head"] * head_vis
        )

        # S_center_true
        hfov_half_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs_angle = float(np.mean(np.abs(horizontal_angles)))
            center_error = mean_abs_angle / hfov_half_rad
            S_center_true = 1.0 - min(max(center_error, 0.0), 1.0)
        else:
            S_center_true = 0.0

        # S_dist_true
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_true = gaussian_score(
            dist_to_human,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

        # Q_true（v2.0 使用独立的权重）
        Q_true = (
            self.true_cfg["w_kp_true"] * S_kp_true
            + self.true_cfg["w_center_true"] * S_center_true
            + self.true_cfg["w_dist_true"] * S_dist_true
        )

        return {
            "S_kp_true": S_kp_true,
            "S_center_true": S_center_true,
            "S_dist_true": S_dist_true,
            "Q_true": Q_true,
            "torso_visibility_true": torso_vis,
            "lower_body_visibility_true": lower_body_vis,
            "head_visibility_true": head_vis,
        }
