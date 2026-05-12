"""
移动前预测评估模块 —— predictive_evaluator.py
===============================================

功能：
    在移动前对候选观察位姿进行预测评分。

核心约束（v2.0 关键规范）：
    - 不允许调用 render_at()
    - 不允许使用候选点 RGB 或 depth 图像
    - 只使用几何信息（FOV、角度、距离）进行预测
    - 输出字段必须以 _pred 后缀命名

预测评分函数：
    Q_pred = w_kp_pred * S_kp_pred
           + w_center_pred * S_center_pred
           + w_dist_pred * S_dist_pred
           - w_move * C_move
"""

from typing import Dict
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .skeleton import KEYPOINT_GROUPS


class PredictiveEvaluator:
    """移动前预测评估器 —— 仅使用几何信息预测视角质量。"""

    def __init__(self, config: dict):
        """初始化预测评估器。

        参数：
            config: 需要 camera、visibility、score_weights、
                   predictive_score、distance_score、candidate_sampling 配置段。
        """
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.score_weights = config["score_weights"]
        self.pred_cfg = config["predictive_score"]
        self.dist_cfg = config["distance_score"]

    def score_view_pred(
        self,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        human_base_pos: np.ndarray,
        human_skeleton: Dict[str, np.ndarray],
        geodesic_distance: float,
    ) -> dict:
        """对候选视角进行预测评分。

        ⚠ 此函数不调用 render_at()，不使用候选点图像。

        参数：
            view_pos: 候选视角位置，shape=(3,)。
            view_yaw: 候选视角朝向（弧度）。
            robot_start_pos: 机器人起始位置（用于 C_move）。
            human_base_pos: 人体基座位置。
            human_skeleton: 人体骨架字典。
            geodesic_distance: 测地距离。

        返回：
            {"S_kp_pred", "S_center_pred", "S_dist_pred", "C_move", "Q_pred",
             "torso_visibility_pred", "lower_body_visibility_pred",
             "head_visibility_pred", "visible_keypoints_pred", "invisible_keypoints_pred"}
        """
        # Step 1: 判断每个关键点的 FOV 可见性
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

        # Step 2: 分组可见率
        def group_vis(group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            visible_count = sum(1 for kp in group_kps if kp in visible_kps)
            return visible_count / len(group_kps)

        torso_vis = group_vis("torso")
        lower_body_vis = group_vis("lower_body")
        head_vis = group_vis("head")

        # Step 3: S_kp_pred
        S_kp_pred = (
            self.score_weights["torso"] * torso_vis
            + self.score_weights["lower_body"] * lower_body_vis
            + self.score_weights["head"] * head_vis
        )

        # Step 4: S_center_pred
        hfov_half_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs_angle = float(np.mean(np.abs(horizontal_angles)))
            center_error = mean_abs_angle / hfov_half_rad
            S_center_pred = 1.0 - min(max(center_error, 0.0), 1.0)
        else:
            S_center_pred = 0.0

        # Step 5: S_dist_pred
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_pred = gaussian_score(
            dist_to_human,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

        # Step 6: C_move
        max_geo = self.config["candidate_sampling"]["max_geodesic_distance"]
        if geodesic_distance == 0.0:
            C_move = 0.0
        else:
            C_move = min(max(geodesic_distance / max_geo, 0.0), 1.0)

        # Step 7: Q_pred
        Q_pred = (
            self.pred_cfg["w_kp_pred"] * S_kp_pred
            + self.pred_cfg["w_center_pred"] * S_center_pred
            + self.pred_cfg["w_dist_pred"] * S_dist_pred
            - self.pred_cfg["w_move"] * C_move
        )

        return {
            "S_kp_pred": S_kp_pred,
            "S_center_pred": S_center_pred,
            "S_dist_pred": S_dist_pred,
            "C_move": C_move,
            "Q_pred": Q_pred,
            "torso_visibility_pred": torso_vis,
            "lower_body_visibility_pred": lower_body_vis,
            "head_visibility_pred": head_vis,
            "visible_keypoints_pred": visible_kps,
            "invisible_keypoints_pred": invisible_kps,
        }
