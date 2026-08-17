"""
实验指标记录模块 —— metrics.py
=================================

功能：
    输出 metrics.csv 和 episodes.jsonl。

v4.0 新增字段（相比 v3.0）：
    - 遮挡指标：occlusion_rate_pred/true, occluded_keypoint_count_pred/true
    - ray cast 健康指标：raycast_error_count/rate_pred/true, is_occlusion_valid_pred
    - depth 观测健康指标：depth_valid/invalid_keypoint_count_true
    - true 评价口径：true_evaluation_source（depth / geometry_fallback）
    - 遮挡感知得分：S_action_occ_pred/true, S_kp_occ_pred/true
    - 遮挡后分组可见率：torso/lower_body/head/arms_visibility_occ_pred/true
    - 遮挡增益：occlusion_gain_true
    - Oracle 指标：oracle_Q_true, oracle_gap,
      oracle_valid_true_candidate_count, oracle_selected_candidate_id
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
        "oracle_Q_true", "oracle_gap",
        "oracle_valid_true_candidate_count", "oracle_selected_candidate_id",
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