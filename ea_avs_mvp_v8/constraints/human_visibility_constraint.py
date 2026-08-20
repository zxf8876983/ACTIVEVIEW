"""
人体视锥与有效观测质量约束 —— human_visibility_constraint.py
=========================================================

职责：
    1. 校验人体目标是否处于候选视点相机的水平与垂直视场角 (FOV) 内；
    2. 计算人体在图像平面上的投影面积占比 (projected_area_ratio)，确保能有效观察老人而非仅有几个像素；
    3. 提供 pose_visibility_score 接口评估人体各关键区域 (头部、躯干、骨盆、四肢) 的有效可见性；
    4. 过滤背对人体、偏角过大、超出传感器有效距离或投影占比过小的无价值视点。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint

logger = logging.getLogger(__name__)


def compute_projected_area_ratio(
    distance: float,
    human_height_m: float = 1.7,
    human_width_m: float = 0.6,
    hfov_deg: float = 90.0,
    img_width: int = 640,
    img_height: int = 480,
) -> float:
    """计算人体针孔相机投影面积占整个画面的比例 (0.0 ~ 1.0)。"""
    if distance <= 0.1:
        return 1.0

    hfov_rad = math.radians(hfov_deg)
    fx = float(img_width) / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx

    proj_w = (human_width_m * fx) / distance
    proj_h = (human_height_m * fy) / distance

    # 限制在图像边界内
    proj_w = min(float(img_width), proj_w)
    proj_h = min(float(img_height), proj_h)

    area_ratio = (proj_w * proj_h) / float(img_width * img_height)
    return round(float(np.clip(area_ratio, 0.0, 1.0)), 4)


def pose_visibility_score(
    human_joints_3d: Dict[str, List[float]],
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    hfov_deg: float = 90.0,
    max_distance: float = 4.5,
) -> float:
    """评估人体核心关键点区域落入相机视锥的可见性得分 [0.0, 1.0]。"""
    if not human_joints_3d:
        return 1.0

    half_fov_rad = math.radians(hfov_deg / 2.0)
    visible_count = 0

    for j_name, j_pos in human_joints_3d.items():
        jp = np.asarray(j_pos, dtype=np.float32)
        vec = jp - cam_pos
        vec_horiz = np.array([vec[0], 0.0, vec[2]], dtype=np.float32)
        dist_h = float(np.linalg.norm(vec_horiz))

        if dist_h > 1e-4:
            dir_j = vec_horiz / dist_h
            dot = np.clip(np.dot(dir_j, cam_forward), -1.0, 1.0)
            angle = math.acos(dot)
            if angle <= half_fov_rad and dist_h <= max_distance:
                visible_count += 1

    return round(float(visible_count / max(1, len(human_joints_3d))), 3)


class HumanVisibilityConstraint:
    """人体是否进入相机视锥 (FOV) 与基础观测质量检查器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hfov_deg = float(self.config.get("hfov_deg", self.config.get("hfov", 90.0)))
        self.max_distance = float(self.config.get("max_distance", 4.5))
        self.min_distance = float(self.config.get("min_distance", 0.8))
        self.min_projected_area_ratio = float(self.config.get("min_projected_area_ratio", 0.015))
        self.img_width = int(self.config.get("width", 640))
        self.img_height = int(self.config.get("height", 480))

    def evaluate(
        self,
        viewpoint: CandidateViewpoint,
        human_position: Union[List[float], np.ndarray],
        human_joints_3d: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[bool, str]:
        """检查人体中心与关键区域是否在候选视点的相机 FOV 视锥内，且具有充足的投影面积。

        返回：
            (is_visible: bool, reason: str)
        """
        hx, hy, hz = [float(x) for x in human_position]
        rx, ry, rz = [float(x) for x in viewpoint.position]

        cam_pos = np.array([rx, ry + float(viewpoint.camera_height), rz], dtype=np.float32)
        target_pos = np.array([hx, hy + 0.80, hz], dtype=np.float32)

        vec_to_human = target_pos - cam_pos
        dist = float(np.linalg.norm(vec_to_human))

        # 1. 距离区间检查
        if dist > self.max_distance:
            return False, "too_far"
        if dist < self.min_distance:
            return False, "too_close"

        # 2. 相机朝向与人体夹角检查 (水平面 FOV)
        vec_horiz = np.array([vec_to_human[0], 0.0, vec_to_human[2]], dtype=np.float32)
        dist_h = float(np.linalg.norm(vec_horiz))
        if dist_h < 1e-4:
            return True, "valid"

        dir_to_human = vec_horiz / dist_h

        # 相机前向向量 (Habitat 坐标系: 0 deg 对应 -Z)
        yaw_rad = math.radians(float(viewpoint.yaw_deg))
        cam_forward = np.array([math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)

        dot = np.clip(np.dot(dir_to_human, cam_forward), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(dot))

        half_fov = self.hfov_deg / 2.0
        if angle_deg > half_fov:
            return False, "human_out_of_fov"

        # 3. 人体投影面积占比检查 (确保能够有效观察老人)
        proj_ratio = compute_projected_area_ratio(
            distance=dist,
            hfov_deg=self.hfov_deg,
            img_width=self.img_width,
            img_height=self.img_height,
        )
        viewpoint.metadata["projected_area_ratio"] = proj_ratio
        if proj_ratio < self.min_projected_area_ratio:
            return False, "insufficient_projected_area"

        # 4. 关键点区域可见性评分 (若提供关节真值)
        if human_joints_3d is not None:
            p_score = pose_visibility_score(
                human_joints_3d=human_joints_3d,
                cam_pos=cam_pos,
                cam_forward=cam_forward,
                hfov_deg=self.hfov_deg,
                max_distance=self.max_distance,
            )
            viewpoint.metadata["pose_visibility_score"] = p_score
            if p_score < 0.25:
                return False, "low_pose_visibility"

        return True, "valid"
