"""
到达后真实评估模块 —— true_evaluator.py
=========================================

功能：
    在策略选定并渲染图像后，计算动作感知导向的真实评估指标。

v3.0 新增：
    - S_action_part_true：动作关键部位真实可见性
    - S_orient_true：朝向适配真实得分

注意：
    - 必须在 render_at() 之后调用
    - 不参与策略选择
    - true 不包含移动代价 C_move
"""

from typing import Dict
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights


class TrueEvaluator:
    """到达后真实评估器。"""

    def __init__(self, config: dict):
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.true_cfg = config["true_score"]
        self.dist_cfg = config["distance_score"]

    def score_view_true(
        self,
        obs: dict,
        view_pos: np.ndarray,
        view_yaw: float,
        human_base_pos: np.ndarray,
        human_yaw: float,
        pose_type: str,
        human_skeleton: Dict[str, np.ndarray],
    ) -> dict:
        """对已渲染的视角进行真实评估。

        ⚠ 必须在策略选择并 render_at() 之后调用。
        ⚠ 不参与策略选择。

        参数：
            obs: render_at() 返回的观测字典。
            view_pos: 视角位置，shape=(3,)。
            view_yaw: 视角朝向（弧度）。
            human_base_pos: 人体基座位置。
            human_yaw: 人体朝向（弧度）。
            pose_type: 姿态类型。
            human_skeleton: 世界坐标人体骨架。

        返回：
            {"S_action_part_true", "S_kp_true", "S_orient_true",
             "S_center_true", "S_dist_true", "Q_true",
             "relative_view_angle_true", ...分组可见率}
        """
        # 判断关键点 FOV 可见性
        visible_kps = []
        invisible_kps = []
        horizontal_angles = []

        for name, pos in human_skeleton.items():
            result = angle_in_camera_fov(
                camera_base_pos=view_pos, camera_yaw=view_yaw, point=pos,
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
        def group_vis(g: str) -> float:
            gkps = KEYPOINT_GROUPS[g]
            if len(gkps) == 0:
                return 0.0
            return sum(1 for kp in gkps if kp in visible_kps) / len(gkps)

        torso_vis = group_vis("torso")
        lower_body_vis = group_vis("lower_body")
        head_vis = group_vis("head")
        arms_vis = group_vis("arms")

        # 通用 S_kp_true
        sw = self.config.get("score_weights", {})
        S_kp_true = (
            sw.get("torso", 0.40) * torso_vis
            + sw.get("lower_body", 0.40) * lower_body_vis
            + sw.get("head", 0.10) * head_vis
            + sw.get("arms", 0.10) * arms_vis
        )

        # 动作关键部位 S_action_part_true
        apw = get_action_part_weights(pose_type, self.config)
        S_action_part_true = (
            apw.get("torso", 0.40) * torso_vis
            + apw.get("lower_body", 0.40) * lower_body_vis
            + apw.get("head", 0.10) * head_vis
            + apw.get("arms", 0.10) * arms_vis
        )

        # 朝向评分 S_orient_true
        rel_angle = compute_relative_view_angle(human_base_pos, human_yaw, view_pos)
        S_orient_true = compute_orientation_score(rel_angle, self.config)

        # 居中度 S_center_true
        hfov_half = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs = float(np.mean(np.abs(horizontal_angles)))
            S_center_true = 1.0 - min(max(mean_abs / hfov_half, 0.0), 1.0)
        else:
            S_center_true = 0.0

        # 距离评分 S_dist_true
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_true = gaussian_score(
            dist_to_human, self.dist_cfg["optimal_distance"], self.dist_cfg["sigma"],
        )

        # Q_true（v3.0 不含移动代价）
        tc = self.true_cfg
        Q_true = (
            tc["w_action_part_true"] * S_action_part_true
            + tc["w_orient_true"] * S_orient_true
            + tc["w_center_true"] * S_center_true
            + tc["w_dist_true"] * S_dist_true
        )

        return {
            "S_action_part_true": S_action_part_true,
            "S_kp_true": S_kp_true,
            "S_orient_true": S_orient_true,
            "S_center_true": S_center_true,
            "S_dist_true": S_dist_true,
            "Q_true": Q_true,
            "relative_view_angle_true": rel_angle,
            "torso_visibility_true": torso_vis,
            "lower_body_visibility_true": lower_body_vis,
            "head_visibility_true": head_vis,
            "arms_visibility_true": arms_vis,
        }
