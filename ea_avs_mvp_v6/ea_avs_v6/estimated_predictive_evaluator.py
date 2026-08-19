"""
估计状态移动前预测评估模块 —— estimated_predictive_evaluator.py
============================================================

功能：
    在移动前基于纯视觉 EstimatedHumanState（Proxy 骨架、估计位置与朝向）
    对候选观察位姿进行遮挡感知预测评分。

重要科学约束：
    - 严禁传入或读取任何 Humanoid GT 变量及 object IDs。
    - 遮挡预测采用 GT-free 几何光路检测。
    - 严格检查 occ_valid：valid=False 的关键点绝不计入 visible_occ。
    - 保持与主实验完全相同的预测评分权重公式。
"""

from typing import Dict, List, Optional
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .keypoint_schema import EA_AVS_15_KEYPOINTS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights
from .occlusion import (
    compute_estimated_keypoint_occlusion,
    compute_estimated_occlusion_stats,
)
from .estimated_human_state import EstimatedHumanState


class EstimatedPredictiveEvaluator:
    """基于估计状态的候选视角预测评估器（GT-Free）。"""

    GROUP_NAMES = ["torso", "lower_body", "head", "arms"]

    def __init__(self, config: dict):
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.pred_cfg = config["predictive_score"]
        self.dist_cfg = config["distance_score"]
        self.occ_cfg = config.get("occlusion", {})

    def score_view_pred(
        self,
        runner,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        estimated_state: EstimatedHumanState,
        geodesic_distance: float,
        pose_type: str = "standing",
    ) -> dict:
        """对候选视角进行基于 EstimatedHumanState 的预测评分。

        参数：
            runner: HabitatRunner 实例（用于静态地图几何 ray casting，不访问 humanoid_manager）。
            view_pos: 候选视角位置。
            view_yaw: 候选视角朝向（弧度）。
            robot_start_pos: 机器人起始位置。
            estimated_state: 纯视觉估计得到的人体状态 EstimatedHumanState。
            geodesic_distance: 测地移动距离。
            pose_type: 动作姿态类型（v6.0 主实验为 "standing" 常量）。

        返回：
            预测评分结果字典（Q_pred 等）。
        """
        # 1. 状态有效性检查
        if (
            not estimated_state.valid
            or estimated_state.human_position_world is None
            or estimated_state.human_yaw is None
            or not estimated_state.proxy_full_skeleton
        ):
            return {
                "Q_pred": -1.0,
                "S_action_occ_pred": 0.0,
                "S_kp_occ_pred": 0.0,
                "S_orient_pred": 0.0,
                "S_center_pred": 0.0,
                "S_dist_pred": 0.0,
                "C_move": 1.0,
                "relative_view_angle_deg": 0.0,
                "visible_keypoints_occ_pred": [],
                "occluded_keypoints_pred": [],
                "fov_visible_keypoints_pred": [],
                "is_occlusion_valid_pred": False,
                "invalid_occlusion_keypoint_count_pred": len(EA_AVS_15_KEYPOINTS),
                "invalid_occlusion_keypoint_rate_pred": 1.0,
                "is_estimated_state": True,
            }

        human_pos = estimated_state.human_position_world
        human_yaw = estimated_state.human_yaw
        proxy_skeleton = estimated_state.proxy_full_skeleton
        kp_names = list(proxy_skeleton.keys())

        # 2. 基于已知场景地图做 GT-free 几何光路射线遮挡检测
        occlusion_map = compute_estimated_keypoint_occlusion(
            runner=runner,
            view_pos=view_pos,
            proxy_skeleton=proxy_skeleton,
            camera_height=self.camera_cfg["camera_height"],
            config=self.config,
        )

        fov_visible = []
        occluded_all = []
        visible_occ = []
        horizontal_angles = []

        for name, kpt_pos in proxy_skeleton.items():
            fov_res = angle_in_camera_fov(
                camera_base_pos=view_pos,
                camera_yaw=view_yaw,
                point=kpt_pos,
                hfov_deg=self.camera_cfg["hfov_deg"],
                width=self.camera_cfg["width"],
                height=self.camera_cfg["height"],
                camera_height=self.camera_cfg["camera_height"],
                min_depth=self.vis_cfg["min_depth"],
                max_depth=self.vis_cfg["max_depth"],
            )

            occ_info = occlusion_map.get(name, {})
            occ_valid = bool(occ_info.get("valid", False))
            is_occ = bool(occ_info.get("occluded", False))
            in_fov = fov_res["in_fov"]

            if in_fov:
                fov_visible.append(name)
                horizontal_angles.append(fov_res["horizontal_angle"])

            if is_occ and occ_valid:
                occluded_all.append(name)

            # 核心科学修正：只有在 FOV 内、遮挡判定有效且未被遮挡时才允许进入 visible_occ
            if in_fov and occ_valid and not is_occ:
                visible_occ.append(name)

        # 3. 动作部位权重与可见性
        weights = get_action_part_weights(pose_type, self.config)
        s_part_scores = {}
        for group in self.GROUP_NAMES:
            members = KEYPOINT_GROUPS[group]
            vis_count = sum(1 for k in members if k in visible_occ)
            s_part_scores[f"S_{group}_occ_pred"] = vis_count / float(len(members))

        S_action_occ_pred = sum(
            weights[g] * s_part_scores[f"S_{g}_occ_pred"] for g in self.GROUP_NAMES
        )
        S_kp_occ_pred = len(visible_occ) / float(len(kp_names)) if kp_names else 0.0

        # 4. 朝向偏角与朝向得分（基于估计朝向）
        rel_angle = compute_relative_view_angle(human_pos, human_yaw, view_pos)
        S_orient_pred = compute_orientation_score(rel_angle, self.config)

        # 5. 图像中心与最佳距离得分（基于估计位置）
        hfov_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if horizontal_angles:
            mean_angle = float(np.mean(np.abs(horizontal_angles)))
            S_center_pred = float(np.clip(1.0 - mean_angle / hfov_rad, 0.0, 1.0))
        else:
            S_center_pred = 0.0

        euclidean_dist = float(np.linalg.norm(view_pos - human_pos))
        S_dist_pred = gaussian_score(
            euclidean_dist,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

        # 6. 移动代价
        max_geo = self.config["candidate_sampling"]["max_geodesic_distance"]
        if geodesic_distance == float("inf"):
            C_move = 1.0
        else:
            C_move = float(np.clip(geodesic_distance / max_geo, 0.0, 1.0))

        w_action = self.pred_cfg["w_action_occ_pred"]
        w_orient = self.pred_cfg["w_orient_pred"]
        w_center = self.pred_cfg["w_center_pred"]
        w_dist = self.pred_cfg["w_dist_pred"]
        w_move = self.pred_cfg["w_move"]

        Q_pred = (
            w_action * S_action_occ_pred
            + w_orient * S_orient_pred
            + w_center * S_center_pred
            + w_dist * S_dist_pred
            - w_move * C_move
        )

        occ_stats = compute_estimated_occlusion_stats(occlusion_map, kp_names)
        invalid_cnt = len(kp_names) - occ_stats["occlusion_valid_keypoint_count"]
        invalid_rate = invalid_cnt / float(len(kp_names)) if kp_names else 1.0
        max_invalid_rate = float(
            self.occ_cfg.get("max_invalid_occlusion_rate", self.occ_cfg.get("max_raycast_error_rate", 0.10))
        )
        is_occ_valid = invalid_rate <= max_invalid_rate

        res = {
            "Q_pred": float(Q_pred),
            "S_action_occ_pred": float(S_action_occ_pred),
            "S_kp_occ_pred": float(S_kp_occ_pred),
            "S_orient_pred": float(S_orient_pred),
            "S_center_pred": float(S_center_pred),
            "S_dist_pred": float(S_dist_pred),
            "C_move": float(C_move),
            "relative_view_angle_deg": float(np.rad2deg(rel_angle)),
            "visible_keypoints_occ_pred": visible_occ,
            "occluded_keypoints_pred": occluded_all,
            "fov_visible_keypoints_pred": fov_visible,
            "is_occlusion_valid_pred": is_occ_valid,
            "invalid_occlusion_keypoint_count_pred": invalid_cnt,
            "invalid_occlusion_keypoint_rate_pred": float(invalid_rate),
            "is_estimated_state": True,
        }
        res.update(s_part_scores)
        res.update(occ_stats)
        return res
