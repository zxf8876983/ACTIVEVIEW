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
      观测层判定：occluded = median(3×3 有效深度) < z_cam - tolerance
      （用前向深度 z_cam 而非欧氏距离，避免图像边缘误判；用中位数而非最小值，
       避免轮廓边缘被单个前景像素误判）

注意：
    - 必须在策略选择之后调用（渲染后）
    - 不参与策略选择
    - true 不包含移动代价 C_move
    - true_score["true_evaluation_source"] 标记评价口径：
      "depth"（使用渲染深度）或 "geometry_fallback"（obs=None 或 depth 不可用）
    - Oracle 只允许比较 "depth" 来源且 depth_coverage_true 达标的候选点，
      保证同口径

v5.0 第三轮（概念分离）：
    - evaluation_source ∈ {depth, geometry, none}：判定依据
    - occlusion_cause ∈ {none, target_surface, environment, humanoid_self,
      unknown}：遮挡原因（由几何 raycast 提供 hint）
    - depth 只能判断 occluded（前方是否有更近表面），不能区分环境/自遮挡；
      geometry 提供 cause。两者冲突（如 depth 说 occluded 而 geometry 说 none）
      时记 occlusion_cause=unknown 并记录 disagreement。
"""

from typing import Dict, List, Optional

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
        """利用渲染深度图对关键点做传感器级遮挡判断。

        v4.0：true 层比 pred 层多使用"实际渲染观测"验证遮挡，
        使 true 与 pred（仅静态地图 ray cast）相互独立。

        遮挡判定使用"前向深度"：
            occluded = sampled_depth < z_cam - tolerance
        其中 z_cam 是关键点在相机坐标系中的前向深度。depth 图像存储的正是
        前向深度，而图像边缘处欧氏距离恒大于前向深度，因此绝不能把 sampled_depth
        与 三维欧氏距离 直接比较（会导致边缘像素误判为遮挡）。

        3×3 邻域采样使用中位数而非最小值：避免关键点位于物体轮廓边缘时，
        被单个邻近前景像素把 median 拉低而误判遮挡。

        投影约定与 geometry 模型一致（相机前向 = +Z 模型方向）：
            - 相机前向（yaw+π 渲染后）= (sin yaw, 0, cos yaw)
            - 相机右向 = (-cos yaw, 0, sin yaw)
            - u = fx * x_cam / z_cam + u0
            - v = v0 - fx * y_cam / z_cam

        参数：
            obs: render_at() 返回的观测字典（含 depth）。
            view_pos: 视角位置（机器人基座位置）。
            view_yaw: 视角朝向（弧度，模型约定 +Z）。
            keypoint_pos: 关键点世界坐标。
            camera_pos: 真实相机位置（= view_pos + camera_height）。

        返回：
            {
                "occluded": bool,
                "depth_valid": bool,
                "sampled_depth": Optional[float],   # 3×3 有效深度中位数
                "center_depth": Optional[float],    # 中心像素深度
                "target_z_depth": Optional[float],  # 关键点前向深度 z_cam
            }
            - depth_valid=False 表示深度图不可用/采样无效（调用方应退回几何判断）
        """
        if obs is None or obs.get("depth") is None:
            return {"occluded": False, "depth_valid": False,
                    "sampled_depth": None, "center_depth": None,
                    "target_z_depth": None}

        depth = np.asarray(obs["depth"])
        if depth.ndim == 3:
            depth = depth[..., 0]
        height, width = depth.shape

        # 统一相机内参（与预测模型 / Habitat render 同一套）
        intrinsics = compute_camera_intrinsics(
            width, height, self.camera_cfg["hfov_deg"]
        )
        fx = intrinsics["fx"]

        # 世界点 -> 相机坐标系
        v = np.array(keypoint_pos, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
        th = view_yaw
        right = np.array([-np.cos(th), 0.0, np.sin(th)])
        forward = np.array([np.sin(th), 0.0, np.cos(th)])
        z_cam = float(v[0] * forward[0] + v[2] * forward[2])
        if z_cam <= 0.0:
            # 关键点在相机后方，深度图无法采样（几何层会排除）
            return {"occluded": False, "depth_valid": False,
                    "sampled_depth": None, "center_depth": None,
                    "target_z_depth": z_cam}
        x_cam = float(v[0] * right[0] + v[2] * right[2])
        y_cam = float(v[1])

        u = fx * x_cam / z_cam + width / 2.0
        v_px = height / 2.0 - fx * y_cam / z_cam
        ui, vi = int(round(u)), int(round(v_px))

        # 收集 3×3 邻域中的所有有效深度，优先使用中位数（非最小值）
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
        # 核心比较：采样深度 vs 关键点前向深度（z_cam），而非欧氏距离
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
        humanoid_object_ids: set = None,
        keypoint_meta: dict = None,
    ) -> dict:
        """对已选定的视角进行遮挡感知真实评估。

        ⚠ 必须在策略选择并 render_at() 之后调用。
        ⚠ 不参与策略选择。

        概念分离（v5.0 第三轮）：
            - evaluation_source：该关键点的遮挡判定依据
              ∈ {"depth", "geometry", "none"}
            - occlusion_cause：遮挡的"原因"（几何 raycast 提供的 hint）
              ∈ {"none", "target_surface", "environment", "humanoid_self", "unknown"}
            - depth 只判断"joint 前方是否有更近表面"（occluded）；它不能区分
              该表面是桌子还是人体躯干。
            - geometry raycast 提供 occlusion_cause。
            - 若 depth 判定 occluded 而 geometry 判定 none → cause=unknown，
              并记录 depth_geometry_disagreement。

        参数：
            runner: HabitatRunner 实例（提供 ray casting）。
            obs: render_at() 返回的观测字典；Oracle 离线评估时可为 None。
            view_pos: 视角位置，shape=(3,)。
            view_yaw: 视角朝向（弧度）。
            human_base_pos: 人体基座位置。
            human_yaw: 人体朝向（弧度）。
            pose_type: 姿态类型。
            human_skeleton: 世界坐标人体骨架。
            humanoid_object_ids: Humanoid object id 集合（遮挡来源分析用）。
            keypoint_meta: 关键点→target link 元数据（区分 target_surface/self）。

        返回：
            {"S_action_occ_true", "S_kp_occ_true", "Q_true", ...含遮挡/深度覆盖指标}

        说明（v5.0 边界）：
            Q_true 是基于几何/depth 的可见性与遮挡代理指标（joint-depth
            visibility / occlusion proxy），不等同于真实视觉关键点检测成功率。
            joint 中心通常位于人体内部，rendered depth 是人体表面。真正的
            视觉关键点检测留待后续 Estimated-State 阶段。
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
                width=self.camera_cfg["width"],
                height=self.camera_cfg["height"],
                camera_height=self.camera_cfg["camera_height"],
                min_depth=self.vis_cfg["min_depth"],
                max_depth=self.vis_cfg["max_depth"],
            )
            horizontal_angles.append(result["horizontal_angle"])
            if result["in_fov"]:
                fov_visible.append(name)

        # =====================================================================
        # 2. 遮挡分析（v5.0 第三轮：分离 evaluation_source 与 occlusion_cause）
        #    - geometry raycast 提供 occluded(bool) 与 occlusion_cause hint
        #    - rendered depth 独立判断 joint 前方是否有更近表面（occluded）
        #    - depth 决定 occluded；geometry 提供 cause；冲突时记 disagreement
        # =====================================================================
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
        # ---- 分析计数 ----
        geom_target_surface_count = 0
        geom_environment_count = 0
        geom_self_count = 0
        geom_unknown_count = 0
        depth_occluded_count = 0
        geo_disc_agreement_count = 0
        geo_disc_disagreement_count = 0

        for name in kp_names:
            geom_info = geom_occlusion.get(name, {})
            geom_valid = geom_info.get("valid", True)
            geom_occ = geom_info.get("occluded", False)
            geom_cause = geom_info.get("occlusion_source", "unknown")

            dres = self._depth_occlusion(
                obs=obs, view_pos=view_pos, view_yaw=view_yaw,
                keypoint_pos=human_skeleton[name], camera_pos=camera_pos,
            )
            target_z = dres["target_z_depth"]

            # 几何 cause 计数（仅在 geometry 有效时）
            if geom_valid:
                if geom_cause == "target_surface":
                    geom_target_surface_count += 1
                elif geom_cause == "environment":
                    geom_environment_count += 1
                elif geom_cause == "humanoid_self":
                    geom_self_count += 1
                elif geom_cause == "unknown":
                    geom_unknown_count += 1

            entry = {
                "target_z_depth": target_z,
                "evaluation_source": "geometry",
                "occlusion_cause": geom_cause if geom_valid else "unknown",
                "geometry_occluded": bool(geom_occ),
                "geometry_occlusion_cause": geom_cause if geom_valid else "unknown",
            }

            if dres["depth_valid"]:
                # 观测层：depth 判定是否 occluded；geometry 提供 cause
                depth_occ = dres["occluded"]
                if depth_occ:
                    depth_occluded_count += 1
                entry.update({
                    "occluded": bool(depth_occ),
                    "depth_occluded": bool(depth_occ),
                    "valid": True,
                    "status": "ok",
                    "hit_distance": dres["sampled_depth"],
                    "center_depth": dres["center_depth"],
                    "evaluation_source": "depth",
                    "sampled_depth": dres["sampled_depth"],
                })
                depth_valid_count += 1
                # depth vs geometry 一致性
                agreement = (bool(depth_occ) == bool(geom_occ))
                entry["depth_geometry_occlusion_agreement"] = agreement
                if agreement:
                    geo_disc_agreement_count += 1
                else:
                    geo_disc_disagreement_count += 1
                    # depth 说 occluded 但 geometry 说 none → cause=unknown
                    if depth_occ and not geom_occ:
                        entry["occlusion_cause"] = "unknown"
            elif geom_valid:
                # 观测层不可用，退回几何层（ray cast 成功的结果）
                occ = geom_occ
                entry.update({
                    "occluded": bool(occ),
                    "depth_occluded": None,
                    "depth_geometry_occlusion_agreement": None,
                    "valid": True,
                    "status": geom_info.get("status", "ok"),
                    "hit_distance": geom_info.get("hit_distance", float("inf")),
                    "evaluation_source": "geometry",
                    "occlusion_cause": geom_cause,
                })
                depth_invalid_count += 1
            else:
                # 深度不可用且几何 ray cast 也失败 → unknown（≠ 未遮挡）
                entry.update({
                    "occluded": False,
                    "depth_occluded": None,
                    "valid": False,
                    "status": "raycast_error",
                    "hit_distance": None,
                    "evaluation_source": "geometry",
                    "occlusion_cause": "unknown",
                    "error": geom_info.get("error", ""),
                })
                depth_invalid_count += 1

            occlusion_result[name] = entry
            occlusion_source_true[name] = "depth" if entry["evaluation_source"] == "depth" else "geom"
            if entry["occluded"]:
                occluded_all.append(name)

        for name in kp_names:
            if (name in fov_visible and name not in occluded_all
                    and occlusion_result.get(name, {}).get("valid", True)):
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
        # 7. 遮挡率统计（只统计有效关键点，ray cast 失败不当作未遮挡）
        # =====================================================================
        occ_stats = compute_occlusion_stats(occlusion_result, kp_names)
        occlusion_rate_true = occ_stats["occlusion_rate"]
        occlusion_valid_keypoint_count_true = occ_stats["occlusion_valid_keypoint_count"]
        raycast_error_count_true = occ_stats["raycast_error_count"]
        occluded_keypoint_count_true = len(occluded_all)
        environment_occluded_keypoint_count_true = occ_stats[
            "environment_occluded_keypoint_count"]
        self_occluded_keypoint_count_true = occ_stats["self_occluded_keypoint_count"]

        # ---- v5.0 第三轮：几何层 cause 统计（true 层分析）----
        total_geom_determined = (
            geom_target_surface_count + geom_environment_count
            + geom_self_count + geom_unknown_count)
        depth_geometry_occlusion_agreement_rate = (
            geo_disc_agreement_count / total_geom_determined
            if total_geom_determined > 0 else None)

        # ---- depth coverage（Oracle 同口径门槛）----
        # depth 有效且在 FOV 内的关键点数
        fov_kp_names = [n for n in kp_names if n in fov_visible]
        depth_valid_in_fov_count = sum(
            1 for n in fov_kp_names
            if occlusion_result.get(n, {}).get("evaluation_source") == "depth"
            and occlusion_result.get(n, {}).get("valid", False)
        )
        fov_visible_keypoint_count_true = len(fov_kp_names)
        depth_coverage_true = (
            depth_valid_in_fov_count / max(fov_visible_keypoint_count_true, 1)
            if fov_visible_keypoint_count_true > 0 else 0.0)

        # true_score 来源：是否真正使用了渲染 depth 作为评价口径
        # （Oracle 只允许比较 depth 来源 + depth_coverage 达标的部分候选）
        true_evaluation_source = (
            "depth"
            if (obs is not None and obs.get("depth") is not None and depth_valid_count > 0)
            else "geometry_fallback"
        )

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
            "occlusion_valid_keypoint_count_true": int(
                occlusion_valid_keypoint_count_true),
            "raycast_error_count_true": int(raycast_error_count_true),
            "raycast_error_rate_true": occ_stats["raycast_error_rate"],
            "environment_occluded_keypoint_count_true": int(
                environment_occluded_keypoint_count_true),
            "self_occluded_keypoint_count_true": int(
                self_occluded_keypoint_count_true),
            # ---- v5.0 第三轮：几何 cause 分析 ----
            "geometry_target_surface_count_true": int(geom_target_surface_count),
            "geometry_environment_occluded_count_true": int(geom_environment_count),
            "geometry_self_occluded_count_true": int(geom_self_count),
            "geometry_unknown_count_true": int(geom_unknown_count),
            "depth_occluded_keypoint_count_true": int(depth_occluded_count),
            "depth_geometry_occlusion_agreement_count": int(
                geo_disc_agreement_count),
            "depth_geometry_occlusion_disagreement_count": int(
                geo_disc_disagreement_count),
            "depth_geometry_occlusion_agreement_rate": (
                depth_geometry_occlusion_agreement_rate),
            # ---- depth coverage（Oracle 同口径门槛）----
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