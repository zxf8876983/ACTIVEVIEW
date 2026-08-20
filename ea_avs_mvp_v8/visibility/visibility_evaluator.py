"""
视点可见性与观测质量评价器 —— visibility_evaluator.py
======================================================

职责：
    1. 评估候选视点对目标人体姿态的几何可见性；
    2. 计算视点与人体关键部位的基础空间指标（欧氏距离、观测方位角/夹角）；
    3. 基于视锥 (Frustum) 与 16 核心关节点投影计算姿态覆盖率 (pose_coverage) 与可见性评分；
    4. 输出标准化 ViewpointQuality 质量评价对象。

科研边界与严禁事项：
    - ❌ Phase 1 严禁实现深度学习评分模型或视角效用学习；
    - ❌ 仅输出基于几何与视锥投影的基础基线指标。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality

logger = logging.getLogger(__name__)


class VisibilityEvaluator:
    """机器人视点可见性与观测质量评价器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimal_distance = float(self.config.get("optimal_distance", 2.0))
        self.max_distance = float(self.config.get("max_distance", 4.5))
        self.hfov_deg = float(self.config.get("hfov_deg", self.config.get("hfov", 90.0)))

    def evaluate(
        self,
        viewpoint: CandidateViewpoint,
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
        sensor_intrinsics: Optional[Dict[str, Any]] = None,
    ) -> ViewpointQuality:
        """评估单个视点对人体骨架的几何观测质量。"""
        vp_pos = np.array(viewpoint.position, dtype=np.float32)
        cam_height = float(viewpoint.camera_height)
        cam_pos = np.array([vp_pos[0], vp_pos[1] + cam_height, vp_pos[2]], dtype=np.float32)

        # 1. 计算与人体核心部位 (pelvis / 躯干) 的欧氏距离
        pelvis = human_joints_3d.get("pelvis", [vp_pos[0], vp_pos[1] + 0.8, vp_pos[2]])
        pelvis_pos = np.array(pelvis, dtype=np.float32)
        dist = float(np.linalg.norm(cam_pos - pelvis_pos))

        # 2. 计算观测夹角 (机器人视线方向与人体正面朝向的夹角)
        h_yaw_rad = math.radians(human_yaw_deg)
        h_forward = np.array([math.sin(h_yaw_rad), 0.0, math.cos(h_yaw_rad)], dtype=np.float32)

        line_of_sight = (pelvis_pos - cam_pos)
        line_of_sight[1] = 0.0
        norm_los = np.linalg.norm(line_of_sight)
        if norm_los > 1e-4:
            line_of_sight /= norm_los
        else:
            line_of_sight = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        dot_val = np.clip(np.dot(-line_of_sight, h_forward), -1.0, 1.0)
        viewing_angle_deg = float(math.degrees(math.acos(dot_val)))

        # 3. 计算视锥内可见关节点数量 (基于水平 FOV 夹角判定)
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
        occlusion_ratio = float(1.0 - pose_coverage)

        # 4. 计算综合几何可见性得分
        dist_diff = abs(dist - self.optimal_distance)
        dist_score = max(0.0, 1.0 - dist_diff / self.optimal_distance)
        front_factor = max(0.0, math.cos(math.radians(viewing_angle_deg)))
        vis_score = float(0.5 * pose_coverage + 0.3 * front_factor + 0.2 * dist_score)

        quality = ViewpointQuality(
            viewpoint_id=viewpoint.viewpoint_id,
            distance=round(dist, 3),
            viewing_angle_deg=round(viewing_angle_deg, 1),
            visible_joints_count=visible_joints,
            visibility_score=round(vis_score, 3),
            occlusion_ratio=round(occlusion_ratio, 3),
            pose_coverage=round(pose_coverage, 3),
            is_valid=bool(visible_joints > 0 and viewpoint.feasible),
            metadata={
                "total_joints": total_joints,
                "distance_score": round(dist_score, 3),
            },
        )
        return quality

    def evaluate_batch(
        self,
        viewpoints: List[CandidateViewpoint],
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
    ) -> List[ViewpointQuality]:
        """批量评估候选视点质量。"""
        return [
            self.evaluate(vp, human_joints_3d, human_yaw_deg=human_yaw_deg)
            for vp in viewpoints
        ]
