"""
实验指标记录模块 —— metrics.py
=================================

功能：
    输出 metrics.csv 和 episodes.jsonl。

v2.0 新增字段：
    - 所有 true 指标（S_kp_true, Q_true 等）
    - selected_is_current（策略是否选择不移动）
    - pred_true_gap（Q_true - Q_pred）
    - visibility_gain_pred（S_kp_pred_selected - S_kp_pred_current）
    - visibility_gain_true（S_kp_true_selected - S_kp_true_current）
"""

import csv
import json
import os
import numpy as np


def _convert_numpy(obj):
    """递归将 numpy 类型转换为 Python 原生类型（JSON 序列化用）。"""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class MetricsWriter:
    """指标写入器 —— 管理 metrics.csv 和 episodes.jsonl。"""

    # v2.0 CSV 字段 —— 同时包含 pred 和 true 指标
    CSV_FIELDS = [
        "episode_id", "scene_id", "policy", "status",
        "num_candidates", "selected_is_current",
        "human_x", "human_y", "human_z",
        "robot_start_x", "robot_start_y", "robot_start_z",
        "selected_x", "selected_y", "selected_z",
        "selected_yaw",
        "geodesic_distance",
        # 预测指标
        "S_kp_pred", "S_center_pred", "S_dist_pred", "C_move", "Q_pred",
        "torso_visibility_pred", "lower_body_visibility_pred", "head_visibility_pred",
        # 真实指标
        "S_kp_true", "S_center_true", "S_dist_true", "Q_true",
        "torso_visibility_true", "lower_body_visibility_true", "head_visibility_true",
        # 差异指标
        "pred_true_gap", "visibility_gain_pred", "visibility_gain_true",
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
        """关闭所有输出文件。"""
        if hasattr(self, "_csv_file") and not self._csv_file.closed:
            self._csv_file.close()
        if hasattr(self, "_jsonl_file") and not self._jsonl_file.closed:
            self._jsonl_file.close()
