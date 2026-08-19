"""
到达后真实评估模块 —— true_evaluator.py
=========================================

功能：
    在策略选定并渲染图像后，计算遮挡感知动作导向的真实评估指标（Q_true, S_action_occ_true 等）。
"""

from typing import Dict, List, Optional, Set
import numpy as np

from .geometry import (
    angle_in_camera_fov,
    compute_camera_intrinsics,
    gaussian_score,
)
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights
from .occlusion import compute_keypoint_occlusion, compute_occlusion_stats


class TrueEvaluator:
    """到达后真实评估器。"""

    GROUP_NAMES = ["torso", "lower_body", "head", "arms"]

    def __init__(self, config: dict):
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.true_cfg = config["true_score"]
        self.dist_cfg = config["distance_score"]
        self.occ_cfg = config.get("occlusion", {})

    def _depth_occlusion(
        self,
        obs: Optional[dict],
        view_pos: np.ndarray,
        view_yaw: float,
        keypoint_pos: np.ndarray,
        camera_pos: np.ndarray,
    ) -> dict:
        """利用渲染深度图对关键点做传感器级遮挡判断。"""
        if obs is None or obs.get("depth") is None:
            return {"occluded": False, "depth_valid": False,
                    "sampled_depth": None, "center_depth": None,
                    "target_z_depth": None}

        depth = np.asarray(obs["depth"])
        if depth.ndim == 3:
            depth = depth[..., 0]
        height, width = depth.shape

        intrinsics = compute_camera_intrinsics(
            width, height, self.camera_cfg["hfov_deg"]
        )
        fx = intrinsics["fx"]

        v = np.array(keypoint_pos, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
        th = view_yaw
        right = np.array([-np.cos(th), 0.0, np.sin(th)])
        forward = np.array([np.sin(th), 0.0, np.cos(th)])
        z_cam = float(v[0] * forward[0] + v[2] * forward[2])
        if z_cam <= 0.0:
            return {"occluded": False, "depth_valid": False,
                    "sampled_depth": None, "center_depth": None,
                    "target_z_depth": z_cam}
        x_cam = float(v[0] * right[0] + v[2] * right[2])
        y_cam = float(v[1])

        u = fx * x_cam / z_cam + width / 2.0
        v_px = height / 2.0 - fx * y_cam / z_cam
        ui, vi = int(round(u)), int(round(v_px))

        valid_depths = []
        center_depth = None
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu, vv = ui + du, vi + dv
                if 0 <= uu < width and 0 <= vv < height:
                    dd = float(depth[vv, uu])
                    if dd > 0.0:
                        valid_depths.append(dd)
                        if du == 0 and dv == 0:
                            center_depth = dd

        if not valid_depths:
            return {"occluded": False, "depth_valid": False,
                    "sampled_depth": None, "center_depth": center_depth,
                    "target_z_depth": z_cam}

        sampled_depth = float(np.median(valid_depths))
        tolerance = self.occ_cfg.get("target_tolerance", 0.08)
        occluded = bool(sampled_depth < z_cam - tolerance)

        return {
            "occluded": occluded,
            "depth_valid": True,
            "sampled_depth": sampled_depth,
            "center_depth": center_depth,
            "target_z_depth": z_cam,
        }

    def score_view_true(
        self,
        runner,
        obs: Optional[dict],
        view_pos: np.ndarray,
        view_yaw: float,
        human_base_pos: np.ndarray,
        human_yaw: float,
        pose_type: str,
        human_skeleton: Dict[str, np.ndarray],
        humanoid_object_ids: Optional[Set[int]] = None,
        keypoint_meta: Optional[dict] = None,
    ) -> dict:
        """对已选定的视角进行遮挡感知真实评估。"""
        kp_names = list(human_skeleton.keys())
        fov_visible = []
        occluded_all = []
        visible_occ = []
        horizontal_angles = []

        for name, pos in human_skeleton.items():
            result = angle_in_camera_fov(
                camera_base_pos=view_pos, camera_yaw=view_yaw, point=pos,
                hfov_deg=self.camera_cfg["hfov_deg"],
                width=self.camera_cfg["width"],
                height=self.camera_cfg["height"],
                camera_height=self.camera_cfg["camera_height"],
                min_depth=self.vis_cfg["min_depth"],
                max_depth=self.vis_cfg["max_depth"],
            )
            horizontal_angles.append(result["horizontal_angle"])
            if result["in_fov"]:
                fov_visible.append(name)

        geom_occlusion = compute_keypoint_occlusion(
            runner=runner,
            view_pos=view_pos,
            human_skeleton=human_skeleton,
            camera_height=self.camera_cfg["camera_height"],
            config=self.config,
            humanoid_object_ids=humanoid_object_ids,
            keypoint_meta=keypoint_meta,
        )
        camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
            [0.0, self.camera_cfg["camera_height"], 0.0])

        occlusion_result = {}
        occlusion_source_true = {}
        depth_valid_count = 0
        depth_invalid_count = 0
        geom_target_surface_count = 0
        geom_environment_count = 0
        geom_self_count = 0
        geom_unknown_count = 0
        geom_none_count = 0
        depth_occluded_count = 0
        geo_disc_agreement_count = 0
        geo_disc_disagreement_count = 0

        for name in kp_names:
            geom_info = geom_occlusion.get(name, {})
            geom_valid = bool(geom_info.get("valid", False))
            geom_occ = bool(geom_info.get("occluded", False))
            geom_cause = geom_info.get("occlusion_cause", "unknown")

            dres = self._depth_occlusion(
                obs=obs, view_pos=view_pos, view_yaw=view_yaw,
                keypoint_pos=human_skeleton[name], camera_pos=camera_pos,
            )
            target_z = dres["target_z_depth"]
            depth_valid = bool(dres["depth_valid"])
            depth_occ = bool(dres["occluded"]) if depth_valid else None

            if geom_cause == "unknown" or not geom_valid:
                geom_unknown_count += 1
            elif geom_cause == "target_surface":
                geom_target_surface_count += 1
            elif geom_cause == "environment":
                geom_environment_count += 1
            elif geom_cause == "humanoid_self":
                geom_self_count += 1
            elif geom_cause not in ("none",):
                geom_unknown_count += 1
            else:
                geom_none_count += 1

            if depth_valid:
                depth_valid_count += 1
                if depth_occ:
                    depth_occluded_count += 1
            else:
                depth_invalid_count += 1

            entry = {
                "target_z_depth": target_z,
                "evaluation_source": "none",
                "depth_valid": depth_valid,
                "depth_occluded": depth_occ,
                "geometry_valid": geom_valid,
                "geometry_occluded": geom_occ,
                "geometry_occlusion_cause": geom_cause if geom_valid else "unknown",
                "occluded": False,
                "occlusion_cause": "unknown",
                "depth_geometry_occlusion_agreement": None,
            }

            if geom_valid and geom_cause == "target_surface":
                entry["occluded"] = False
                entry["occlusion_cause"] = "target_surface"
                if depth_valid:
                    entry["depth_geometry_occlusion_agreement"] = bool(depth_occ == geom_occ)
            elif depth_valid:
                final_occ = depth_occ
                if final_occ:
                    if geom_valid and geom_occ and geom_cause in ("humanoid_self", "environment"):
                        entry["occlusion_cause"] = geom_cause
                    elif geom_valid and geom_cause == "target_surface":
                        entry["occlusion_cause"] = "target_surface"
                    else:
                        entry["occlusion_cause"] = "unknown"
                else:
                    entry["occlusion_cause"] = "none"
                entry["occluded"] = final_occ
                if geom_valid:
                    entry["depth_geometry_occlusion_agreement"] = bool(depth_occ == geom_occ)
            elif geom_valid:
                entry["occluded"] = geom_occ
                entry["occlusion_cause"] = geom_cause
                entry["evaluation_source"] = "geometry"
            else:
                entry["occluded"] = False
                entry["occlusion_cause"] = "unknown"

            entry["valid"] = bool(depth_valid or geom_valid)
            if depth_valid or geom_valid:
                entry["evaluation_source"] = "depth" if depth_valid else "geometry"
                entry["status"] = "ok"
            else:
                entry["status"] = "unknown"

            ag = entry["depth_geometry_occlusion_agreement"]
            if ag is not None:
                if ag:
                    geo_disc_agreement_count += 1
                else:
                    geo_disc_disagreement_count += 1

            occlusion_result[name] = entry
            occlusion_source_true[name] = "depth" if entry["evaluation_source"] == "depth" else "geom"
            if entry["occluded"]:
                occluded_all.append(name)

        for name in kp_names:
            if (name in fov_visible and name not in occluded_all
                    and occlusion_result.get(name, {}).get("valid", True)):
                visible_occ.append(name)

        def group_vis(visible_set: set, group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            return sum(1 for kp in group_kps if kp in visible_set) / float(len(group_kps))

        fov_visible_set = set(fov_visible)
        visible_occ_set = set(visible_occ)

        fov_group = {g: group_vis(fov_visible_set, g) for g in self.GROUP_NAMES}
        occ_group = {g: group_vis(visible_occ_set, g) for g in self.GROUP_NAMES}

        score_weights = self.config.get("score_weights", {})
        w_general = {
            "torso": score_weights.get("torso", 0.40),
            "lower_body": score_weights.get("lower_body", 0.40),
            "head": score_weights.get("head", 0.10),
            "arms": score_weights.get("arms", 0.10),
        }

        S_kp_true = sum(w_general[g] * fov_group[g] for g in self.GROUP_NAMES)
        S_kp_occ_true = sum(w_general[g] * occ_group[g] for g in self.GROUP_NAMES)

        ap_weights = get_action_part_weights(pose_type, self.config)

        S_action_part_true = sum(
            ap_weights.get(g, w_general[g]) * fov_group[g]
            for g in self.GROUP_NAMES
        )
        S_action_occ_true = sum(
            ap_weights.get(g, w_general[g]) * occ_group[g]
            for g in self.GROUP_NAMES
        )

        rel_angle = compute_relative_view_angle(human_base_pos, human_yaw, view_pos)
        S_orient_true = compute_orientation_score(rel_angle, self.config)

        hfov_half = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs = float(np.mean(np.abs(horizontal_angles)))
            S_center_true = 1.0 - min(max(mean_abs / hfov_half, 0.0), 1.0)
        else:
            S_center_true = 0.0

        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_true = gaussian_score(
            dist_to_human, self.dist_cfg["optimal_distance"], self.dist_cfg["sigma"],
        )

        occ_stats = compute_occlusion_stats(occlusion_result, kp_names)
        occlusion_rate_true = occ_stats["occlusion_rate"]
        occlusion_valid_keypoint_count_true = occ_stats["occlusion_valid_keypoint_count"]
        raycast_error_count_true = occ_stats["raycast_error_count"]
        occluded_keypoint_count_true = len(occluded_all)
        environment_occluded_keypoint_count_true = occ_stats["environment_occluded_keypoint_count"]
        self_occluded_keypoint_count_true = occ_stats["self_occluded_keypoint_count"]
        unknown_occlusion_keypoint_count_true = occ_stats["unknown_occlusion_keypoint_count"]
        target_surface_keypoint_count_true = occ_stats["target_surface_keypoint_count"]

        depth_geometry_occlusion_agreement_rate = (
            geo_disc_agreement_count / float(geo_disc_agreement_count + geo_disc_disagreement_count)
            if (geo_disc_agreement_count + geo_disc_disagreement_count) > 0 else None
        )

        fov_kp_names = [n for n in kp_names if n in fov_visible]
        depth_valid_in_fov_count = sum(
            1 for n in fov_kp_names
            if occlusion_result.get(n, {}).get("evaluation_source") == "depth"
            and occlusion_result.get(n, {}).get("valid", False)
        )
        fov_visible_keypoint_count_true = len(fov_kp_names)
        depth_coverage_true = (
            depth_valid_in_fov_count / float(max(fov_visible_keypoint_count_true, 1))
            if fov_visible_keypoint_count_true > 0 else 0.0
        )

        true_evaluation_source = (
            "depth"
            if (obs is not None and obs.get("depth") is not None and depth_valid_count > 0)
            else "geometry_fallback"
        )

        tc = self.true_cfg
        Q_true = (
            tc["w_action_occ_true"] * S_action_occ_true
            + tc["w_orient_true"] * S_orient_true
            + tc["w_center_true"] * S_center_true
            + tc["w_dist_true"] * S_dist_true
        )

        return {
            "S_action_occ_true": float(S_action_occ_true),
            "S_kp_occ_true": float(S_kp_occ_true),
            "occlusion_rate_true": float(occlusion_rate_true),
            "occluded_keypoint_count_true": int(occluded_keypoint_count_true),
            "occlusion_valid_keypoint_count_true": int(occlusion_valid_keypoint_count_true),
            "raycast_error_count_true": int(raycast_error_count_true),
            "raycast_error_rate_true": occ_stats["raycast_error_rate"],
            "environment_occluded_keypoint_count_true": int(environment_occluded_keypoint_count_true),
            "self_occluded_keypoint_count_true": int(self_occluded_keypoint_count_true),
            "target_surface_keypoint_count_true": int(target_surface_keypoint_count_true),
            "unknown_occlusion_keypoint_count_true": int(unknown_occlusion_keypoint_count_true),
            "geometry_target_surface_count_true": int(geom_target_surface_count),
            "geometry_environment_occluded_count_true": int(geom_environment_count),
            "geometry_self_occluded_count_true": int(geom_self_count),
            "geometry_unknown_count_true": int(geom_unknown_count),
            "geometry_none_count_true": int(geom_none_count),
            "depth_occluded_keypoint_count_true": int(depth_occluded_count),
            "depth_geometry_occlusion_agreement_count": int(geo_disc_agreement_count),
            "depth_geometry_occlusion_disagreement_count": int(geo_disc_disagreement_count),
            "depth_geometry_occlusion_agreement_rate": depth_geometry_occlusion_agreement_rate,
            "fov_visible_keypoint_count_true": int(fov_visible_keypoint_count_true),
            "depth_valid_in_fov_count_true": int(depth_valid_in_fov_count),
            "depth_coverage_true": float(depth_coverage_true),
            "depth_valid_keypoint_count_true": int(depth_valid_count),
            "depth_invalid_keypoint_count_true": int(depth_invalid_count),
            "true_evaluation_source": true_evaluation_source,
            "visible_keypoints_occ_true": visible_occ,
            "occluded_keypoints_true": occluded_all,
            "occlusion_source_true": occlusion_source_true,
            "depth_occluded_keypoints_true": [
                n for n in occluded_all
                if occlusion_source_true.get(n) == "depth"
            ],
            "torso_visibility_occ_true": float(occ_group["torso"]),
            "lower_body_visibility_occ_true": float(occ_group["lower_body"]),
            "head_visibility_occ_true": float(occ_group["head"]),
            "arms_visibility_occ_true": float(occ_group["arms"]),
            "S_action_part_true": float(S_action_part_true),
            "S_kp_true": float(S_kp_true),
            "torso_visibility_true": float(fov_group["torso"]),
            "lower_body_visibility_true": float(fov_group["lower_body"]),
            "head_visibility_true": float(fov_group["head"]),
            "arms_visibility_true": float(fov_group["arms"]),
            "visible_keypoints_true": fov_visible,
            "S_orient_true": float(S_orient_true),
            "S_center_true": float(S_center_true),
            "S_dist_true": float(S_dist_true),
            "relative_view_angle_true": float(rel_angle),
            "Q_true": float(Q_true),
            "occlusion_result": occlusion_result,
        }
