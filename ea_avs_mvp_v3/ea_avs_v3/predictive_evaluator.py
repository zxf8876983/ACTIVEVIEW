"""
移动前预测评估模块 —— predictive_evaluator.py
===============================================

功能：
    在移动前对候选观察位姿进行动作感知导向的预测评分。

v3.0 核心变化（相比 v2.0）：
    1. S_action_part_pred —— 新增动作关键部位可见性
    2. S_orient_pred —— 新增朝向适配评分
    3. Q_pred 权重调整为含动作和朝向

预测评分函数：
    Q_pred = w_action_part_pred * S_action_part_pred
           + w_orient_pred * S_orient_pred
           + w_center_pred * S_center_pred
           + w_dist_pred * S_dist_pred
           - w_move * C_move

约束：
    - 不允许调用 render_at()
    - 不允许使用候选点图像
    - 不允许使用 true_score
"""

from typing import Dict
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights


class PredictiveEvaluator:
    """移动前预测评估器 —— 动作感知导向的视角质量预测。"""

    def __init__(self, config: dict):
        """初始化预测评估器。

        参数：
            config: 配置字典。
        """
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.pred_cfg = config["predictive_score"]
        self.dist_cfg = config["distance_score"]

    def score_view_pred(
        self,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        human_base_pos: np.ndarray,
        human_yaw: float,
        pose_type: str,
        human_skeleton: Dict[str, np.ndarray],
        geodesic_distance: float,
    ) -> dict:
        """对候选视角进行动作感知导向的预测评分。

        ⚠ 不调用 render_at()，不使用候选点图像。

        参数：
            view_pos: 候选视角位置，shape=(3,)。
            view_yaw: 候选视角朝向（弧度）。
            robot_start_pos: 机器人起始位置。
            human_base_pos: 人体基座位置。
            human_yaw: 人体朝向角（弧度）。
            pose_type: 姿态类型，如 "standing"。
            human_skeleton: 世界坐标人体骨架字典。
            geodesic_distance: 测地距离。

        返回：
            {"S_action_part_pred", "S_kp_pred", "S_orient_pred",
             "S_center_pred", "S_dist_pred", "C_move", "Q_pred",
             "relative_view_angle", ...分组可见率, "visible_keypoints_pred"...}
        """
        # =====================================================================
        # 1. 判断每个关键点的 FOV 可见性
        # =====================================================================
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

        # =====================================================================
        # 2. 计算分组可见率（含新增的 arms 组）
        # =====================================================================
        def group_vis(group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            visible_count = sum(1 for kp in group_kps if kp in visible_kps)
            return visible_count / len(group_kps)

        torso_vis = group_vis("torso")
        lower_body_vis = group_vis("lower_body")
        head_vis = group_vis("head")
        arms_vis = group_vis("arms")

        # =====================================================================
        # 3. 计算 S_kp_pred（通用关键点可见率，与 v2.0 相同）
        # =====================================================================
        score_weights = self.config.get("score_weights", {})
        w_torso = score_weights.get("torso", 0.40)
        w_lower = score_weights.get("lower_body", 0.40)
        w_head = score_weights.get("head", 0.10)
        w_arms = score_weights.get("arms", 0.10)

        S_kp_pred = (
            w_torso * torso_vis + w_lower * lower_body_vis
            + w_head * head_vis + w_arms * arms_vis
        )

        # =====================================================================
        # 4. 动作关键部位可见性 S_action_part_pred（v3.0 新增）
        # =====================================================================
        ap_weights = get_action_part_weights(pose_type, self.config)
        S_action_part_pred = (
            ap_weights.get("torso", w_torso) * torso_vis
            + ap_weights.get("lower_body", w_lower) * lower_body_vis
            + ap_weights.get("head", w_head) * head_vis
            + ap_weights.get("arms", w_arms) * arms_vis
        )

        # =====================================================================
        # 5. 朝向评分 S_orient_pred（v3.0 新增）
        # =====================================================================
        relative_view_angle = compute_relative_view_angle(
            human_base_pos, human_yaw, view_pos
        )
        S_orient_pred = compute_orientation_score(relative_view_angle, self.config)

        # =====================================================================
        # 6. 居中度 S_center_pred
        # =====================================================================
        hfov_half_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs_angle = float(np.mean(np.abs(horizontal_angles)))
            center_error = mean_abs_angle / hfov_half_rad
            S_center_pred = 1.0 - min(max(center_error, 0.0), 1.0)
        else:
            S_center_pred = 0.0

        # =====================================================================
        # 7. 距离评分 S_dist_pred
        # =====================================================================
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_pred = gaussian_score(
            dist_to_human,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

        # =====================================================================
        # 8. 运动代价 C_move
        # =====================================================================
        max_geo = self.config["candidate_sampling"]["max_geodesic_distance"]
        C_move = 0.0 if geodesic_distance == 0.0 else min(max(geodesic_distance / max_geo, 0.0), 1.0)

        # =====================================================================
        # 9. 最终 Q_pred（v3.0 公式）
        # =====================================================================
        pc = self.pred_cfg
        Q_pred = (
            pc["w_action_part_pred"] * S_action_part_pred
            + pc["w_orient_pred"] * S_orient_pred
            + pc["w_center_pred"] * S_center_pred
            + pc["w_dist_pred"] * S_dist_pred
            - pc["w_move"] * C_move
        )

        return {
            "S_action_part_pred": S_action_part_pred,
            "S_kp_pred": S_kp_pred,
            "S_orient_pred": S_orient_pred,
            "S_center_pred": S_center_pred,
            "S_dist_pred": S_dist_pred,
            "C_move": C_move,
            "Q_pred": Q_pred,
            "relative_view_angle": relative_view_angle,
            "torso_visibility_pred": torso_vis,
            "lower_body_visibility_pred": lower_body_vis,
            "head_visibility_pred": head_vis,
            "arms_visibility_pred": arms_vis,
            "visible_keypoints_pred": visible_kps,
            "invisible_keypoints_pred": invisible_kps,
        }
