"""
实验指标记录模块 —— metrics.py
=================================

功能：
    输出 metrics.csv 和 episodes.jsonl。

v5.0 新增字段（在 v4.0 基础上）：
    - Humanoid 状态：humanoid_enabled, humanoid_avatar_name, humanoid_pose_name,
      humanoid_motion_frame, humanoid_base_x/y/z, humanoid_yaw,
      humanoid_gt_skeleton_source, humanoid_render_success,
      humanoid_self_occlusion_supported_pred
    - 继续保留 v4.0 全部指标（S_action_occ_pred/true, Q_pred/true,
      occlusion_rate, oracle_gap 等）
"""

import csv
import json
import math
import os

import numpy as np


def _convert_numpy(obj):
    """递归将 numpy 类型转换为 Python 原生类型（非有限浮点转为 None）。"""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


class MetricsWriter:
    """指标写入器 —— 管理 metrics.csv 和 episodes.jsonl。"""

    # v4.0 CSV 字段 —— 保留 v3.0 字段，新增遮挡与 Oracle 指标
    CSV_FIELDS = [
        "episode_id", "scene_id", "policy", "status",
        "num_candidates", "selected_is_current",
        "pose_type", "human_yaw", "relative_view_angle",
        "human_x", "human_y", "human_z",
        "robot_start_x", "robot_start_y", "robot_start_z",
        "selected_x", "selected_y", "selected_z",
        "selected_yaw",
        "geodesic_distance",
        # ---- 预测指标（v3.0 全字段）----
        "S_action_part_pred", "S_kp_pred", "S_orient_pred",
        "S_center_pred", "S_dist_pred", "C_move", "Q_pred",
        "torso_visibility_pred", "lower_body_visibility_pred",
        "head_visibility_pred", "arms_visibility_pred",
        # ---- v4.0 新增遮挡感知预测指标 ----
        "S_action_occ_pred", "S_kp_occ_pred",
        "occlusion_rate_pred", "occluded_keypoint_count_pred",
        "occlusion_valid_keypoint_count_pred",
        "raycast_error_count_pred", "raycast_error_rate_pred",
        "invalid_occlusion_keypoint_count_pred",
        "invalid_occlusion_keypoint_rate_pred",
        "target_surface_keypoint_count_pred",
        "environment_occluded_keypoint_count_pred",
        "self_occluded_keypoint_count_pred",
        "unknown_occlusion_keypoint_count_pred",
        "is_occlusion_valid_pred",
        "torso_visibility_occ_pred", "lower_body_visibility_occ_pred",
        "head_visibility_occ_pred", "arms_visibility_occ_pred",
        # ---- 真实指标（v3.0 全字段）----
        "S_action_part_true", "S_kp_true", "S_orient_true",
        "S_center_true", "S_dist_true", "Q_true",
        "torso_visibility_true", "lower_body_visibility_true",
        "head_visibility_true", "arms_visibility_true",
        # ---- v4.0 新增遮挡感知真实指标 ----
        "S_action_occ_true", "S_kp_occ_true",
        "occlusion_rate_true", "occluded_keypoint_count_true",
        "occlusion_valid_keypoint_count_true",
        "raycast_error_count_true", "raycast_error_rate_true",
        "environment_occluded_keypoint_count_true",
        "self_occluded_keypoint_count_true",
        "target_surface_keypoint_count_true",
        "unknown_occlusion_keypoint_count_true",
        # ---- v5.0 第三轮真实分析指标 ----
        "geometry_target_surface_count_true",
        "geometry_environment_occluded_count_true",
        "geometry_self_occluded_count_true",
        "geometry_unknown_count_true",
        "geometry_none_count_true",
        "depth_occluded_keypoint_count_true",
        "depth_geometry_occlusion_agreement_count",
        "depth_geometry_occlusion_disagreement_count",
        "depth_geometry_occlusion_agreement_rate",
        "fov_visible_keypoint_count_true",
        "depth_valid_in_fov_count_true",
        "depth_coverage_true",
        "depth_valid_keypoint_count_true", "depth_invalid_keypoint_count_true",
        "true_evaluation_source",
        "torso_visibility_occ_true", "lower_body_visibility_occ_true",
        "head_visibility_occ_true", "arms_visibility_occ_true",
        # ---- 差异 / 增益指标 ----
        "pred_true_gap",
        "action_part_gain_pred", "action_part_gain_true",
        "visibility_gain_pred", "visibility_gain_true",
        "occlusion_gain_true",
        # ---- Oracle 上界指标 ----
        "oracle_Q_true", "oracle_available",
        "oracle_gap", "oracle_gap_valid", "oracle_gap_reason",
        "oracle_valid_true_candidate_count", "oracle_selected_candidate_id",
        "oracle_depth_eligible_candidate_count",
        "oracle_excluded_low_depth_coverage_count",
        # ---- 在线选择有效性（Ours 过滤 invalid-occlusion）----
        "selected_occlusion_valid_pred",
        "ours_invalid_occ_candidate_excluded_count",
        "ours_fallback_to_current_due_to_no_valid_occ_candidate",
        "ours_stay_by_score",
        "ours_stay_by_fallback",
        # ---- v5.0 Humanoid 状态 ----
        "semantic_sensor_available", "semantic_assignment_ok",
        "semantic_assignment_count",
        "humanoid_enabled", "humanoid_avatar_name", "humanoid_pose_name",
        "humanoid_motion_frame",
        "humanoid_base_x", "humanoid_base_y", "humanoid_base_z",
        "humanoid_yaw",
        "humanoid_gt_skeleton_source",
        "humanoid_gt_keypoint_count",
        "humanoid_gt_direct_link_count",
        "humanoid_gt_link_derived_count",
        "humanoid_gt_fallback_count",
        # ---- current 视角渲染验证 ----
        "current_humanoid_render_success",
        "current_humanoid_validation_source",
        "current_humanoid_visible",
        "current_humanoid_pixel_count",
        "current_humanoid_pixel_ratio",
        "current_humanoid_match_ratio",
        "current_humanoid_bbox_x1", "current_humanoid_bbox_y1",
        "current_humanoid_bbox_x2", "current_humanoid_bbox_y2",
        "current_humanoid_depth_valid_ratio",
        "current_humanoid_proxy_match_ratio",
        # ---- selected 视角渲染验证（每策略各自的选中位姿）----
        "selected_humanoid_render_success",
        "selected_humanoid_validation_source",
        "selected_humanoid_visible",
        "selected_humanoid_pixel_count",
        "selected_humanoid_pixel_ratio",
        "selected_humanoid_match_ratio",
        "selected_humanoid_bbox_x1", "selected_humanoid_bbox_y1",
        "selected_humanoid_bbox_x2", "selected_humanoid_bbox_y2",
        "selected_humanoid_depth_valid_ratio",
        "selected_humanoid_proxy_match_ratio",
        # ---- 渲染/可见性增益 ----
        "humanoid_pixel_gain",
        "humanoid_proxy_match_gain",
        "requested_human_yaw",
        "actual_humanoid_yaw",
        "humanoid_self_occlusion_status_pred",
    ]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._csv_file = open(
            os.path.join(output_dir, "metrics.csv"), "w", newline=""
        )
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()

        self._jsonl_file = open(
            os.path.join(output_dir, "episodes.jsonl"), "w"
        )

    def write_metric_row(self, row: dict):
        """写入 metrics.csv 的一行。"""
        filtered = {k: row.get(k, "") for k in self.CSV_FIELDS}
        self._csv_writer.writerow(filtered)
        self._csv_file.flush()

    def write_episode_summary(self, summary: dict):
        """写入 episodes.jsonl 的一行。"""
        clean = _convert_numpy(summary)
        self._jsonl_file.write(json.dumps(clean, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def close(self):
        if hasattr(self, "_csv_file") and not self._csv_file.closed:
            self._csv_file.close()
        if hasattr(self, "_jsonl_file") and not self._jsonl_file.closed:
            self._jsonl_file.close()