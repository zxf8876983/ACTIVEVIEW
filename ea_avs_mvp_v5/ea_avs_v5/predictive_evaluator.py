"""
移动前预测评估模块 —— predictive_evaluator.py
===============================================

功能：
    在移动前对候选观察位姿进行遮挡感知动作导向的预测评分。

v4.0 核心变化（相比 v3.0）：
    1. 新增环境遮挡判断：visible = in_fov AND NOT occluded
    2. S_action_occ_pred —— 遮挡感知动作关键部位可见性（核心新增）
    3. S_kp_occ_pred —— 遮挡感知通用关键点可见性
    4. Q_pred 使用 w_action_occ_pred 汇聚遮挡后动作部位可见性

预测评分函数：
    Q_pred =
        w_action_occ_pred * S_action_occ_pred
      + w_orient_pred * S_orient_pred
      + w_center_pred * S_center_pred
      + w_dist_pred * S_dist_pred
      - w_move * C_move

约束：
    - 不允许调用 render_at()
    - 不允许使用候选点图像
    - 不允许使用 true_score
    - 允许使用已知地图静态几何 ray casting（视为已知场景信息）
"""

from typing import Dict

import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score
from .action_pose_library import KEYPOINT_GROUPS
from .orientation import compute_relative_view_angle, compute_orientation_score
from .action_part_weights import get_action_part_weights
from .occlusion import compute_keypoint_occlusion, compute_occlusion_stats


class PredictiveEvaluator:
    """移动前预测评估器 —— 遮挡感知动作导向的视角质量预测。"""

    # 关键点分组名称（用于可见率计算）
    GROUP_NAMES = ["torso", "lower_body", "head", "arms"]

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
        runner,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        human_base_pos: np.ndarray,
        human_yaw: float,
        pose_type: str,
        human_skeleton: Dict[str, np.ndarray],
        geodesic_distance: float,
        humanoid_object_ids: set = None,
    ) -> dict:
        """对候选视角进行遮挡感知动作导向的预测评分。

        ⚠ 不调用 render_at()，不使用候选点图像。

        参数：
            runner: HabitatRunner 实例（提供 ray casting）。
            view_pos: 候选视角位置，shape=(3,)。
            view_yaw: 候选视角朝向（弧度）。
            robot_start_pos: 机器人起始位置。
            human_base_pos: 人体基座位置。
            human_yaw: 人体朝向角（弧度）。
            pose_type: 姿态类型，如 "standing"。
            human_skeleton: 世界坐标人体骨架字典。
            geodesic_distance: 测地距离。
            humanoid_object_ids: Humanoid 相关 object id 集合；传入时可将遮挡
                来源区分为 environment / humanoid_self（不参与评分，仅供分析）。

        返回：
            见类头部注释：含 S_action_occ_pred / S_kp_occ_pred / Q_pred 等。
        """
        # =====================================================================
        # 1. FOV 可见性与水平角
        # =====================================================================
        kp_names = list(human_skeleton.keys())
        fov_visible = []
        occluded_all = []
        visible_occ = []
        horizontal_angles = []

        # 逐关键点判断 FOV（VFOV 由统一相机内参推导）
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
        # 2. 环境/自遮挡分析（v4.0 核心）—— 射线起点 = 真实相机位置
        # =====================================================================
        occlusion_result = compute_keypoint_occlusion(
            runner=runner,
            view_pos=view_pos,
            human_skeleton=human_skeleton,
            camera_height=self.camera_cfg["camera_height"],
            config=self.config,
            humanoid_object_ids=humanoid_object_ids,
        )

        for name in kp_names:
            occ_info = occlusion_result.get(name, {})
            if occ_info.get("occluded", False):
                occluded_all.append(name)

        # visible_after_occlusion = in_fov AND NOT occluded AND 遮挡状态有效
        # （ray cast 失败即无法确认可见性，不视为可见，避免乐观偏置）
        for name in kp_names:
            info = occlusion_result.get(name, {})
            if (name in fov_visible
                    and name not in occluded_all
                    and info.get("valid", True)):
                visible_occ.append(name)

        # =====================================================================
        # 3. 分组可见率（遮挡后 / 纯 FOV 两套）
        # =====================================================================
        def group_vis(visible_set: set, group_name: str) -> float:
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            return sum(1 for kp in group_kps if kp in visible_set) / len(group_kps)

        fov_visible_set = set(fov_visible)
        visible_occ_set = set(visible_occ)

        # 纯 FOV 可见率（v3.0 口径，供消融/对比）
        fov_group = {g: group_vis(fov_visible_set, g) for g in self.GROUP_NAMES}
        # 遮挡后可见率（v4.0 口径）
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

        S_kp_pred = sum(
            w_general[g] * fov_group[g] for g in self.GROUP_NAMES
        )
        S_kp_occ_pred = sum(
            w_general[g] * occ_group[g] for g in self.GROUP_NAMES
        )

        # =====================================================================
        # 5. 动作关键部位可见性（v3.0 口径 + v4.0 遮挡口径）
        # =====================================================================
        ap_weights = get_action_part_weights(pose_type, self.config)

        S_action_part_pred = sum(
            ap_weights.get(g, w_general[g]) * fov_group[g]
            for g in self.GROUP_NAMES
        )
        S_action_occ_pred = sum(
            ap_weights.get(g, w_general[g]) * occ_group[g]
            for g in self.GROUP_NAMES
        )

        # =====================================================================
        # 6. 朝向评分 S_orient_pred
        # =====================================================================
        relative_view_angle = compute_relative_view_angle(
            human_base_pos, human_yaw, view_pos
        )
        S_orient_pred = compute_orientation_score(relative_view_angle, self.config)

        # =====================================================================
        # 7. 居中度 S_center_pred
        # =====================================================================
        hfov_half_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs_angle = float(np.mean(np.abs(horizontal_angles)))
            S_center_pred = 1.0 - min(max(mean_abs_angle / hfov_half_rad, 0.0), 1.0)
        else:
            S_center_pred = 0.0

        # =====================================================================
        # 8. 距离评分 S_dist_pred
        # =====================================================================
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist_pred = gaussian_score(
            dist_to_human,
            self.dist_cfg["optimal_distance"],
            self.dist_cfg["sigma"],
        )

        # =====================================================================
        # 9. 运动代价 C_move
        # =====================================================================
        max_geo = self.config["candidate_sampling"]["max_geodesic_distance"]
        C_move = 0.0 if geodesic_distance == 0.0 else min(
            max(geodesic_distance / max_geo, 0.0), 1.0
        )

        # =====================================================================
        # 10. 遮挡率统计（只统计有效关键点；不参与 Q_pred 双重计权）
        #     ray cast 失败不计入"未遮挡"，单独输出 error rate
        # =====================================================================
        occ_stats = compute_occlusion_stats(occlusion_result, kp_names)
        occlusion_rate_pred = occ_stats["occlusion_rate"]
        occlusion_valid_keypoint_count_pred = occ_stats["occlusion_valid_keypoint_count"]
        raycast_error_count_pred = occ_stats["raycast_error_count"]
        raycast_error_rate_pred = occ_stats["raycast_error_rate"]
        occluded_keypoint_count_pred = len(occluded_all)
        # v5.0：遮挡来源拆分（环境 / 自遮挡 / 未知）
        environment_occluded_keypoint_count_pred = occ_stats[
            "environment_occluded_keypoint_count"]
        self_occluded_keypoint_count_pred = occ_stats["self_occluded_keypoint_count"]
        unknown_occlusion_keypoint_count_pred = occ_stats[
            "unknown_occlusion_keypoint_count"]

        # 单视角遮挡判断有效性：ray cast 失败率超过阈值则标记无效
        max_error_rate = self.config.get("occlusion", {}).get(
            "max_raycast_error_rate", 0.10)
        is_occlusion_valid_pred = bool(
            raycast_error_rate_pred <= max_error_rate
        )

        # =====================================================================
        # 11. 最终 Q_pred（v4.0 公式）
        # =====================================================================
        pc = self.pred_cfg
        Q_pred = (
            pc["w_action_occ_pred"] * S_action_occ_pred
            + pc["w_orient_pred"] * S_orient_pred
            + pc["w_center_pred"] * S_center_pred
            + pc["w_dist_pred"] * S_dist_pred
            - pc["w_move"] * C_move
        )

        return {
            # 遮挡感知（v4.0 核心）
            "S_action_occ_pred": float(S_action_occ_pred),
            "S_kp_occ_pred": float(S_kp_occ_pred),
            "occlusion_rate_pred": float(occlusion_rate_pred),
            "occlusion_valid_keypoint_count_pred": int(
                occlusion_valid_keypoint_count_pred),
            "occluded_keypoint_count_pred": int(occluded_keypoint_count_pred),
            "raycast_error_count_pred": int(raycast_error_count_pred),
            "raycast_error_rate_pred": float(raycast_error_rate_pred),
            "environment_occluded_keypoint_count_pred": int(
                environment_occluded_keypoint_count_pred),
            "self_occluded_keypoint_count_pred": int(
                self_occluded_keypoint_count_pred),
            "unknown_occlusion_keypoint_count_pred": int(
                unknown_occlusion_keypoint_count_pred),
            "is_occlusion_valid_pred": is_occlusion_valid_pred,
            "visible_keypoints_occ_pred": visible_occ,
            "occluded_keypoints_pred": occluded_all,
            "torso_visibility_occ_pred": float(occ_group["torso"]),
            "lower_body_visibility_occ_pred": float(occ_group["lower_body"]),
            "head_visibility_occ_pred": float(occ_group["head"]),
            "arms_visibility_occ_pred": float(occ_group["arms"]),
            # 纯 FOV（v3.0 口径，用于消融比较）
            "S_action_part_pred": float(S_action_part_pred),
            "S_kp_pred": float(S_kp_pred),
            "torso_visibility_pred": float(fov_group["torso"]),
            "lower_body_visibility_pred": float(fov_group["lower_body"]),
            "head_visibility_pred": float(fov_group["head"]),
            "arms_visibility_pred": float(fov_group["arms"]),
            "visible_keypoints_pred": fov_visible,
            "invisible_keypoints_pred": [
                n for n in kp_names if n not in fov_visible_set
            ],
            # 通用评分项
            "S_orient_pred": float(S_orient_pred),
            "S_center_pred": float(S_center_pred),
            "S_dist_pred": float(S_dist_pred),
            "C_move": float(C_move),
            "relative_view_angle": float(relative_view_angle),
            "Q_pred": float(Q_pred),
            # 调试信息（逐关键点遮挡结果）
            "occlusion_result": occlusion_result,
        }