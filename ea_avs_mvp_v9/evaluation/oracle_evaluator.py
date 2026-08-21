"""
Oracle 理论最佳视点评测器 —— oracle_evaluator.py
=================================================

职责：
    1. 使用 Ground-Truth 人体 3D 关节点与理想几何可见性计算理论上限最佳视角 (Oracle Upper Bound)；
    2. 作为科研评测基准的上限参考 (Upper Bound)，严禁作为决策模型输入；
    3. 输出 OracleViewpointResult 结构。
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.core.types import OracleViewpointResult
from ea_avs_mvp_v9.features.observation_simulator import ObservationSimulator, compute_observation_quality


class OracleViewEvaluator:
    """Oracle 理论上限视点评测器。"""

    def __init__(self):
        self.obs_sim = ObservationSimulator()

    def evaluate_oracle_best(
        self,
        candidates: List[CandidateViewpoint],
        gt_joints: Dict[str, List[float]],
        human_pos: List[float],
        human_yaw_deg: float = 0.0,
        initial_quality: float = 0.0,
    ) -> OracleViewpointResult:
        """
        在所有可行候选视点中，基于 GT 人体状态与无退化观测计算理论最优上限。
        """
        if not candidates:
            raise ValueError("Candidate viewpoints list is empty")

        best_vp_id = candidates[0].viewpoint_id
        max_quality = -1.0
        max_gain = 0.0
        best_obs = None

        for vp in candidates:
            if not vp.feasible:
                continue

            # 使用理想/真实无退化观测模拟该视角下的极限感知质量
            ideal_obs = self.obs_sim.simulate_observation(
                gt_joints=gt_joints,
                camera_pos=vp.position,
                human_pos=human_pos,
                human_yaw_deg=human_yaw_deg,
                degradation_mode="none",
            )
            q = compute_observation_quality(ideal_obs, dist=vp.radius)
            gain = max(0.0, q - initial_quality)

            if q > max_quality:
                max_quality = q
                max_gain = gain
                best_vp_id = vp.viewpoint_id
                best_obs = ideal_obs

        vis_count = sum(1 for c in best_obs.joint_confidences.values() if c >= 0.70) if best_obs else 16
        body_vis = best_obs.body_part_confidences if best_obs else {}

        return OracleViewpointResult(
            best_viewpoint_id=best_vp_id,
            oracle_visibility_score=round(float(max_quality), 3),
            oracle_quality_score=round(float(max_quality), 3),
            oracle_information_gain=round(float(max_gain), 3),
            oracle_joints_visible_count=vis_count,
            oracle_body_parts_visibility={k: round(v, 3) for k, v in body_vis.items()},
        )
