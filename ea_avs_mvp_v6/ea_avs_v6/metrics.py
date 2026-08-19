"""
评估指标统计与记录模块 —— metrics.py
======================================

功能：
    定义与记录 v6.0 的三大核心指标：
        1. 状态估计精度（位置误差、朝向误差、MPJPE、3D 可观测关节数）
        2. 策略选择质量与比较（EstimatedState-Ours vs GTState-Ours vs Baselines vs Oracle）
        3. 动作导向真实得分（Q_true, S_action_occ_true, 遮挡率, Oracle Gap, Estimation Gap）
"""

import csv
import json
import os
from typing import Dict, List, Optional
import numpy as np

from .geometry import normalize_angle
from .estimated_human_state import EstimatedHumanState


def compute_state_estimation_metrics(
    estimated_state: EstimatedHumanState,
    gt_human_pos: np.ndarray,
    gt_human_yaw: float,
    gt_skeleton: Dict[str, np.ndarray],
) -> dict:
    """计算当前 RGB-D 人体状态估计与 GT 之间的精度误差。

    参数：
        estimated_state: 估计的人体状态。
        gt_human_pos: GT 人体位置 shape=(3,)。
        gt_human_yaw: GT 人体朝向角（弧度）。
        gt_skeleton: GT 人体 15 关键点世界坐标字典。

    返回：
        状态估计误差指标字典。
    """
    if not estimated_state.valid or estimated_state.human_position_world is None:
        return {
            "state_valid": False,
            "state_failure_reason": estimated_state.failure_reason or "invalid_state",
            "pos_error_m": None,
            "pos_error_xz_m": None,
            "yaw_error_deg": None,
            "scale_error": None,
            "observable_joint_error_mean_m": None,
            "proxy_skeleton_mpjpe_m": None,
            "num_visible_2d_keypoints": len(estimated_state.visible_2d_keypoints),
            "num_observable_3d_keypoints": len(estimated_state.observable_3d_keypoints),
            "num_template_completed_keypoints": len(estimated_state.template_completed_keypoints),
        }

    est_pos = estimated_state.human_position_world
    pos_err = float(np.linalg.norm(est_pos - gt_human_pos))
    pos_err_xz = float(np.linalg.norm(est_pos[[0, 2]] - gt_human_pos[[0, 2]]))

    if estimated_state.human_yaw is not None:
        yaw_diff = normalize_angle(estimated_state.human_yaw - gt_human_yaw)
        yaw_err_deg = float(np.rad2deg(abs(yaw_diff)))
    else:
        yaw_err_deg = None

    scale_err = (
        float(abs(estimated_state.body_scale - 1.0))
        if estimated_state.body_scale is not None
        else None
    )

    # 1. 仅真实观测到的 3D 关节误差
    obs_joint_errors = []
    for k in estimated_state.observable_3d_keypoints:
        if k in estimated_state.observed_skeleton and k in gt_skeleton:
            err = float(np.linalg.norm(estimated_state.observed_skeleton[k] - gt_skeleton[k]))
            obs_joint_errors.append(err)
    obs_mpjpe = float(np.mean(obs_joint_errors)) if obs_joint_errors else None

    # 2. Proxy 完整 15 关节 MPJPE
    all_joint_errors = []
    for k, pos in estimated_state.proxy_full_skeleton.items():
        if k in gt_skeleton:
            err = float(np.linalg.norm(pos - gt_skeleton[k]))
            all_joint_errors.append(err)
    proxy_mpjpe = float(np.mean(all_joint_errors)) if all_joint_errors else None

    return {
        "state_valid": True,
        "state_failure_reason": None,
        "pos_error_m": pos_err,
        "pos_error_xz_m": pos_err_xz,
        "yaw_error_deg": yaw_err_deg,
        "scale_error": scale_err,
        "observable_joint_error_mean_m": obs_mpjpe,
        "proxy_skeleton_mpjpe_m": proxy_mpjpe,
        "num_visible_2d_keypoints": len(estimated_state.visible_2d_keypoints),
        "num_observable_3d_keypoints": len(estimated_state.observable_3d_keypoints),
        "num_template_completed_keypoints": len(estimated_state.template_completed_keypoints),
    }


class MetricsWriter:
    """指标记录与落盘工具。"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "metrics.csv")
        self.jsonl_path = os.path.join(output_dir, "episodes.jsonl")
        self.summary_path = os.path.join(output_dir, "summary.json")
        self.records: List[dict] = []
        self._csv_file = None
        self._csv_writer = None

    def log_episode(self, episode_data: dict):
        self.records.append(episode_data)

        # 写入 JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_data, ensure_ascii=False) + "\n")

        # 写入 CSV (在 close 时统一格式化写入，或使用固定字段)
        pass

    def close(self):
        if self.records:
            # 统一计算所有 records 的平坦字段集合
            flat_records = [self._flatten_dict(r) for r in self.records]
            all_fields = []
            seen = set()
            for r in flat_records:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        all_fields.append(k)

            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
                writer.writeheader()
                for r in flat_records:
                    writer.writerow(r)

            summary = self._compute_summary(self.records)
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if k in ("observed_skeleton", "proxy_full_skeleton", "occlusion_result", "keypoint_meta"):
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            elif isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _compute_summary(self, records: List[dict]) -> dict:
        summary = {"num_episodes": len(records)}

        def agg_mean(key_path):
            vals = []
            for r in records:
                cur = r
                for k in key_path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        cur = None
                        break
                if cur is not None and isinstance(cur, (int, float)) and not np.isnan(cur):
                    vals.append(cur)
            return float(np.mean(vals)) if vals else None

        # 估计指标聚合
        summary["mean_pos_error_m"] = agg_mean(["estimation_metrics", "pos_error_m"])
        summary["mean_pos_error_xz_m"] = agg_mean(["estimation_metrics", "pos_error_xz_m"])
        summary["mean_yaw_error_deg"] = agg_mean(["estimation_metrics", "yaw_error_deg"])
        summary["mean_proxy_mpjpe_m"] = agg_mean(["estimation_metrics", "proxy_skeleton_mpjpe_m"])
        summary["mean_observable_3d_keypoints"] = agg_mean(["estimation_metrics", "num_observable_3d_keypoints"])

        # 策略真实得分聚合
        for pol in ["EstimatedState-Ours", "GTState-Ours", "Fixed", "Random", "Nearest", "Oracle"]:
            summary[f"mean_Q_true_{pol}"] = agg_mean(["policy_results", pol, "true_score", "Q_true"])
            summary[f"mean_S_action_occ_true_{pol}"] = agg_mean(["policy_results", pol, "true_score", "S_action_occ_true"])
            summary[f"mean_occlusion_rate_true_{pol}"] = agg_mean(["policy_results", pol, "true_score", "occlusion_rate_true"])

        summary["mean_oracle_gap_est"] = agg_mean(["comparative_metrics", "oracle_gap_est"])
        summary["mean_oracle_gap_gt"] = agg_mean(["comparative_metrics", "oracle_gap_gt"])
        summary["mean_estimation_gap"] = agg_mean(["comparative_metrics", "estimation_gap"])
        summary["policy_agreement_rate"] = agg_mean(["comparative_metrics", "estimated_gt_agreement"])

        return summary
