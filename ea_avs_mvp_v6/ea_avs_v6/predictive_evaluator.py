"""
GT 状态移动前预测评估模块 —— predictive_evaluator.py
======================================================

功能：
    在移动前对候选观察位姿进行基于 GT 人体状态的遮挡感知动作导向预测评分。
    供 GTState-Ours 特权基线使用。
"""

from typing import Dict, List, Optional, Set
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights
from .occlusion import compute_keypoint_occlusion, compute_occlusion_stats


class PredictiveEvaluator:
    """GT 状态预测评估器。"""

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
        human_base_pos: np.ndarray,
        human_yaw: float,
        pose_type: str,
        human_skeleton: Dict[str, np.ndarray],
        geodesic_distance: float,
        humanoid_object_ids: Optional[Set[int]] = None,
        keypoint_meta: Optional[dict] = None,
    ) -> dict:
        """对候选视角进行基于 GT 状态的预测评分。"""
        kp_names = list(human_skeleton.keys())
        fov_visible = []
        occluded_all = []
        visible_occ = []
        horizontal_angles = []

        occlusion_map = compute_keypoint_occlusion(
            runner=runner,
            view_pos=view_pos,
            human_skeleton=human_skeleton,
            camera_height=self.camera_cfg["camera_height"],
            config=self.config,
            humanoid_object_ids=humanoid_object_ids,
            keypoint_meta=keypoint_meta,
        )

        for name, kpt_pos in human_skeleton.items():
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
            is_occ = occ_info.get("occluded", False)
            in_fov = fov_res["in_fov"]

            if in_fov:
                fov_visible.append(name)
                horizontal_angles.append(fov_res["horizontal_angle"])

            if is_occ:
                occluded_all.append(name)

            if in_fov and not is_occ:
                visible_occ.append(name)

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

        rel_angle = compute_relative_view_angle(human_base_pos, human_yaw, view_pos)
        S_orient_pred = compute_orientation_score(rel_angle, self.config)

        hfov_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if horizontal_angles:
            mean_angle = float(np.mean(np.abs(horizontal_angles)))
            S_center_pred = float(np.clip(1.0 - mean_angle / hfov_rad, 0.0, 1.0))
        else:
            S_center_pred = 0.0

        euclidean_dist = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_pred = gaussian_score(
            euclidean_dist,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

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

        occ_stats = compute_occlusion_stats(occlusion_map, kp_names)
        invalid_cnt = len(kp_names) - occ_stats["occlusion_valid_keypoint_count"]
        invalid_rate = invalid_cnt / float(len(kp_names)) if kp_names else 1.0
        max_invalid_rate = float(self.occ_cfg.get("max_raycast_error_rate", 0.10))
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
        }
        res.update(s_part_scores)
        res.update(occ_stats)
        return res
