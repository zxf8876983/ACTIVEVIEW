"""
视点质量综合评价器 —— view_quality.py
======================================

职责：
    1. 计算候选视点的综合观测质量评分 (view_score)；
    2. 基于科学评价公式：
       score = w1 * human_visibility + w2 * pose_coverage - w3 * distance_penalty
    3. 对候选视点进行排序 (ranking) 并输出全局最优视点 (best_view)。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality

logger = logging.getLogger(__name__)


def compute_view_quality_score(
    human_visibility: float,
    pose_coverage: float,
    distance: float,
    optimal_distance: float = 2.0,
    viewing_angle_deg: float = 0.0,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> float:
    """计算单个视点的综合质量得分。

    公式：
        score = w1 * human_visibility + w2 * pose_coverage - w3 * distance_penalty
    """
    dist_diff = abs(distance - optimal_distance)
    dist_penalty = min(1.0, dist_diff / max(1e-4, optimal_distance))

    # 正面观察偏好奖励 (0 度正面观测最具信息量)
    front_rad = math.radians(viewing_angle_deg)
    front_factor = max(0.0, math.cos(front_rad))

    raw_score = w1 * human_visibility * (0.8 + 0.2 * front_factor) + w2 * pose_coverage - w3 * dist_penalty
    norm_score = float(np.clip(raw_score, 0.0, 1.0))
    return round(norm_score, 3)


class ViewQualityEvaluator:
    """视点质量多维评估与排序器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimal_distance = float(self.config.get("optimal_distance", 2.0))
        self.max_distance = float(self.config.get("max_distance", 4.5))
        self.hfov_deg = float(self.config.get("hfov_deg", self.config.get("hfov", 90.0)))
        self.w1 = float(self.config.get("w1_visibility", 0.5))
        self.w2 = float(self.config.get("w2_pose_coverage", 0.3))
        self.w3 = float(self.config.get("w3_distance", 0.2))

    def evaluate_viewpoint(
        self,
        viewpoint: CandidateViewpoint,
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
    ) -> ViewpointQuality:
        """评估单个候选视点的几何观测质量与综合得分。"""
        vp_pos = np.array(viewpoint.position, dtype=np.float32)
        cam_height = float(viewpoint.camera_height)
        cam_pos = np.array([vp_pos[0], vp_pos[1] + cam_height, vp_pos[2]], dtype=np.float32)

        # 1. 距离计算 (到人体躯干/骨盆)
        pelvis = human_joints_3d.get("pelvis", [vp_pos[0], vp_pos[1] + 0.8, vp_pos[2]])
        pelvis_pos = np.array(pelvis, dtype=np.float32)
        dist = float(np.linalg.norm(cam_pos - pelvis_pos))

        # 2. 视角夹角计算 (机器人视线与人体正面朝向夹角)
        # KinematicHumanoid default orientation: 0 deg 面向 +Z
        h_yaw_rad = math.radians(human_yaw_deg)
        h_forward = np.array([math.sin(h_yaw_rad), 0.0, math.cos(h_yaw_rad)], dtype=np.float32)

        line_of_sight = (pelvis_pos - cam_pos)
        line_of_sight[1] = 0.0
        norm_los = np.linalg.norm(line_of_sight)
        if norm_los > 1e-4:
            line_of_sight /= norm_los
        else:
            line_of_sight = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # 机器人视线由 cam 射向 human: 反向向量为 -line_of_sight 与 h_forward 的点积
        dot_val = np.clip(np.dot(-line_of_sight, h_forward), -1.0, 1.0)
        viewing_angle_deg = float(math.degrees(math.acos(dot_val)))

        # 3. 视锥内 16 关键点覆盖率与可见性判定
        half_fov_rad = math.radians(self.hfov_deg / 2.0)
        vp_yaw_rad = math.radians(float(viewpoint.yaw_deg))
        cam_forward = np.array([math.sin(vp_yaw_rad), 0.0, -math.cos(vp_yaw_rad)], dtype=np.float32)

        visible_joints = 0
        total_joints = max(1, len(human_joints_3d))

        for j_name, j_pos in human_joints_3d.items():
            jp = np.array(j_pos, dtype=np.float32)
            vec_to_joint = jp - cam_pos
            vec_horiz = np.array([vec_to_joint[0], 0.0, vec_to_joint[2]], dtype=np.float32)
            dist_j = np.linalg.norm(vec_horiz)

            if dist_j > 1e-4:
                dir_j = vec_horiz / dist_j
                cos_angle = np.clip(np.dot(dir_j, cam_forward), -1.0, 1.0)
                angle_to_cam = math.acos(cos_angle)
                if angle_to_cam <= half_fov_rad and dist_j <= self.max_distance:
                    visible_joints += 1

        pose_coverage = float(visible_joints / total_joints)
        human_visibility = 1.0 if (visible_joints >= 4 and viewpoint.feasible) else float(pose_coverage)

        # 4. 综合得分计算 (w1*vis + w2*cov - w3*dist)
        if not viewpoint.feasible:
            score = 0.0
            is_valid = False
        else:
            score = compute_view_quality_score(
                human_visibility=human_visibility,
                pose_coverage=pose_coverage,
                distance=dist,
                optimal_distance=self.optimal_distance,
                viewing_angle_deg=viewing_angle_deg,
                w1=self.w1,
                w2=self.w2,
                w3=self.w3,
            )
            is_valid = bool(visible_joints > 0 and score > 0.0)

        quality = ViewpointQuality(
            viewpoint_id=viewpoint.viewpoint_id,
            distance=round(dist, 3),
            viewing_angle_deg=round(viewing_angle_deg, 1),
            visible_joints_count=visible_joints,
            visibility_score=score,
            occlusion_ratio=round(1.0 - pose_coverage, 3),
            pose_coverage=round(pose_coverage, 3),
            is_valid=is_valid,
            metadata={
                "w1_visibility": self.w1,
                "w2_pose_coverage": self.w2,
                "w3_distance": self.w3,
                "optimal_distance": self.optimal_distance,
            },
        )
        return quality

    def rank_viewpoints(
        self,
        viewpoints: List[CandidateViewpoint],
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
    ) -> List[Tuple[CandidateViewpoint, ViewpointQuality]]:
        """评估并按综合得分降序排列所有候选视点。"""
        ranked_list: List[Tuple[CandidateViewpoint, ViewpointQuality]] = []
        for vp in viewpoints:
            q = self.evaluate_viewpoint(vp, human_joints_3d, human_yaw_deg=human_yaw_deg)
            ranked_list.append((vp, q))

        # 按 visibility_score 降序排序，若分数相同按正面角度排序 (viewing_angle_deg 越小越优先)
        ranked_list.sort(key=lambda item: (item[1].is_valid, item[1].visibility_score, -item[1].viewing_angle_deg), reverse=True)
        return ranked_list
