"""
到达后真实评估模块 —— true_evaluator.py
=========================================

功能：
    在策略选定并渲染图像后，计算遮挡感知动作导向的真实评估指标。

v4.0 新增：
    - S_action_occ_true：遮挡感知动作关键部位真实可见性
    - S_kp_occ_true：遮挡感知通用关键点真实可见性
    - 真实遮挡判断分两级：
        1) 几何层：基于 Habitat 场景 mesh 的 ray casting（与 pred 相同，作为 fallback）
        2) 观测层：利用渲染深度图在关键点投影像素处做传感器级遮挡验证（true 独有）
      两者结合使 true 比 pred 多使用"实际渲染观测"，相互独立。

注意：
    - 必须在策略选择之后调用（渲染后）
    - 不参与策略选择
    - true 不包含移动代价 C_move
    - obs 为 None（如 Oracle 离线评估）时，观测层不可用，自动退回几何层判断
"""

from typing import Dict, List, Optional

import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights
from .occlusion import compute_keypoint_occlusion, compute_occlusion_rate


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
    ):
        """利用渲染深度图对关键点做传感器级遮挡判断。

        v4.0：true 层比 pred 层多使用"实际渲染观测"验证遮挡，
        使 true 与 pred（仅静态地图 ray cast）相互独立。

        投影约定与 geometry 模型一致：
            - 相机前向（yaw+π 渲染后）= (sin yaw, 0, cos yaw)
            - 相机右向 = (-cos yaw, 0, sin yaw)
            - u = fx * x_cam / z_cam + u0
            - v = v0 - fx * y_cam / z_cam
        该投影公式已通过"渲染深度 vs 射线命中距离"实证验证。

        参数：
            obs: render_at() 返回的观测字典（含 depth）。
            view_pos: 视角位置（机器人基座位置）。
            view_yaw: 视角朝向（弧度，模型约定 +Z）。
            keypoint_pos: 关键点世界坐标。
            camera_pos: 真实相机位置（= view_pos + camera_height）。

        返回：
            (occluded: bool, depth_valid: bool, sampled_depth: Optional[float])
            - depth_valid=False 表示深度图不可用/采样无效（调用方应退回几何判断）
        """
        if obs is None or obs.get("depth") is None:
            return False, False, None

        depth = np.asarray(obs["depth"])
        if depth.ndim == 3:
            depth = depth[..., 0]
        height, width = depth.shape

        # 相机内参（方形像素，fx == fy）
        hfov_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        fx = (width / 2.0) / np.tan(hfov_rad)

        # 世界点 -> 相机坐标系
        v = np.array(keypoint_pos, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
        th = view_yaw
        right = np.array([-np.cos(th), 0.0, np.sin(th)])
        forward = np.array([np.sin(th), 0.0, np.cos(th)])
        z_cam = float(v[0] * forward[0] + v[2] * forward[2])
        if z_cam <= 0.0:
            # 关键点在相机后方，深度图无法采样（几何层已排除）
            return False, False, None
        x_cam = float(v[0] * right[0] + v[2] * right[2])
        y_cam = float(v[1])

        u = fx * x_cam / z_cam + width / 2.0
        v_px = height / 2.0 - fx * y_cam / z_cam
        ui, vi = int(round(u)), int(round(v_px))

        # 采样 3x3 邻域的最小深度，容忍轻微投影误差
        best_depth = None
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu, vv = ui + du, vi + dv
                if 0 <= uu < width and 0 <= vv < height:
                    dd = float(depth[vv, uu])
                    if dd > 0.0 and (best_depth is None or dd < best_depth):
                        best_depth = dd

        if best_depth is None:
            return False, False, None

        ray_distance = float(np.linalg.norm(
            np.array(keypoint_pos) - np.array(camera_pos)))
        tolerance = self.occ_cfg.get("target_tolerance", 0.08)
        occluded = bool(best_depth < ray_distance - tolerance)
        return occluded, True, best_depth

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
    ) -> dict:
        """对已选定的视角进行遮挡感知真实评估。

        ⚠ 必须在策略选择并 render_at() 之后调用。
        ⚠ 不参与策略选择。

        参数：
            runner: HabitatRunner 实例（提供 ray casting）。
            obs: render_at() 返回的观测字典；Oracle 离线评估时可为 None。
            view_pos: 视角位置，shape=(3,)。
            view_yaw: 视角朝向（弧度）。
            human_base_pos: 人体基座位置。
            human_yaw: 人体朝向（弧度）。
            pose_type: 姿态类型。
            human_skeleton: 世界坐标人体骨架。

        返回：
            {"S_action_occ_true", "S_kp_occ_true", "Q_true", ...含遮挡指标}
        """
        # =====================================================================
        # 1. FOV 可见性与水平角
        # =====================================================================
        kp_names = list(human_skeleton.keys())
        fov_visible = []
        occluded_all = []
        visible_occ = []
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
            horizontal_angles.append(result["horizontal_angle"])
            if result["in_fov"]:
                fov_visible.append(name)

        # =====================================================================
        # 2. 环境遮挡分析（v4.0）
        #    先用静态地图 ray cast 得到几何遮挡；再用渲染深度图做传感器级
        #    独立验证（true 层独有，pred 层不可见未来图像）。
        #    visible_after_occlusion = in_fov AND NOT occluded
        # =====================================================================
        geom_occlusion = compute_keypoint_occlusion(
            runner=runner,
            view_pos=view_pos,
            human_skeleton=human_skeleton,
            camera_height=self.camera_cfg["camera_height"],
            config=self.config,
        )
        camera_pos = np.array(view_pos, dtype=np.float64) + np.array(
            [0.0, self.camera_cfg["camera_height"], 0.0])

        occlusion_result = {}
        occlusion_source_true = {}
        for name in kp_names:
            geom_info = geom_occlusion.get(name, {})
            geom_occ = geom_info.get("occluded", False)

            depth_occ, depth_valid, sampled_depth = self._depth_occlusion(
                obs=obs, view_pos=view_pos, view_yaw=view_yaw,
                keypoint_pos=human_skeleton[name], camera_pos=camera_pos,
            )

            if depth_valid:
                occ = depth_occ
                source = "depth"
                hit_distance = sampled_depth
            else:
                occ = geom_occ
                source = "geom"
                hit_distance = geom_info.get("hit_distance", float("inf"))

            occlusion_result[name] = {
                "occluded": occ,
                "hit_distance": hit_distance,
                "target_distance": geom_info.get("target_distance", 0.0),
                "source": source,
            }
            occlusion_source_true[name] = source
            if occ:
                occluded_all.append(name)

        for name in kp_names:
            if name in fov_visible and name not in occluded_all:
                visible_occ.append(name)

        # =====================================================================
        # 3. 分组可见率（遮挡后 / 纯 FOV）
        # =====================================================================
        def group_vis(visible_set: set, group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            return sum(1 for kp in group_kps if kp in visible_set) / len(group_kps)

        fov_visible_set = set(fov_visible)
        visible_occ_set = set(visible_occ)

        fov_group = {g: group_vis(fov_visible_set, g) for g in self.GROUP_NAMES}
        occ_group = {g: group_vis(visible_occ_set, g) for g in self.GROUP_NAMES}

        # =====================================================================
        # 4. 通用关键点可见率
        # =====================================================================
        score_weights = self.config.get("score_weights", {})
        w_general = {
            "torso": score_weights.get("torso", 0.40),
            "lower_body": score_weights.get("lower_body", 0.40),
            "head": score_weights.get("head", 0.10),
            "arms": score_weights.get("arms", 0.10),
        }

        S_kp_true = sum(w_general[g] * fov_group[g] for g in self.GROUP_NAMES)
        S_kp_occ_true = sum(w_general[g] * occ_group[g] for g in self.GROUP_NAMES)

        # =====================================================================
        # 5. 动作关键部位可见率（遮挡后 / 纯 FOV）
        # =====================================================================
        ap_weights = get_action_part_weights(pose_type, self.config)

        S_action_part_true = sum(
            ap_weights.get(g, w_general[g]) * fov_group[g]
            for g in self.GROUP_NAMES
        )
        S_action_occ_true = sum(
            ap_weights.get(g, w_general[g]) * occ_group[g]
            for g in self.GROUP_NAMES
        )

        # =====================================================================
        # 6. 朝向 / 居中 / 距离评分
        # =====================================================================
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

        # =====================================================================
        # 7. 遮挡率统计
        # =====================================================================
        occlusion_rate_true = compute_occlusion_rate(occlusion_result, kp_names)
        occluded_keypoint_count_true = len(occluded_all)

        # =====================================================================
        # 8. Q_true（v4.0，不含移动代价）
        # =====================================================================
        tc = self.true_cfg
        Q_true = (
            tc["w_action_occ_true"] * S_action_occ_true
            + tc["w_orient_true"] * S_orient_true
            + tc["w_center_true"] * S_center_true
            + tc["w_dist_true"] * S_dist_true
        )

        return {
            # 遮挡感知（v4.0 核心）
            "S_action_occ_true": float(S_action_occ_true),
            "S_kp_occ_true": float(S_kp_occ_true),
            "occlusion_rate_true": float(occlusion_rate_true),
            "occluded_keypoint_count_true": int(occluded_keypoint_count_true),
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
            # 纯 FOV（v3.0 口径）
            "S_action_part_true": float(S_action_part_true),
            "S_kp_true": float(S_kp_true),
            "torso_visibility_true": float(fov_group["torso"]),
            "lower_body_visibility_true": float(fov_group["lower_body"]),
            "head_visibility_true": float(fov_group["head"]),
            "arms_visibility_true": float(fov_group["arms"]),
            "visible_keypoints_true": fov_visible,
            # 通用评分项
            "S_orient_true": float(S_orient_true),
            "S_center_true": float(S_center_true),
            "S_dist_true": float(S_dist_true),
            "relative_view_angle_true": float(rel_angle),
            "Q_true": float(Q_true),
            # 调试信息
            "occlusion_result": occlusion_result,
        }