"""
视点观测多维特征提取器 —— view_feature_extractor.py
=================================================

职责：
    1. 继承并复用 v8 基础几何与视锥投影特性；
    2. 计算 16 骨骼关节点在相机视锥内的精细分区可见性 (Region Coverage):
       - head (头部与颈部)
       - torso (胸腔与脊柱)
       - pelvis (骨盆与臀部)
       - upper_body (上肢与双臂)
       - lower_body (下肢与膝踝)
    3. 输出标准化 ViewFeature 结构，供下游 ActionConditionedScorer 综合决策。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.core.types import ViewFeature

logger = logging.getLogger(__name__)

# 16 核心关节到 7 大解剖身体关键部位的标准语义映射
BODY_PART_JOINT_MAPPINGS: Dict[str, List[str]] = {
    "head": ["head", "neck", "nose", "eyes"],
    "torso": ["spine", "spine1", "spine2", "spine3", "chest", "thorax"],
    "pelvis": ["pelvis", "hip", "hips", "root"],
    "left_hand": ["left_wrist", "left_elbow", "left_hand", "left_shoulder"],
    "right_hand": ["right_wrist", "right_elbow", "right_hand", "right_shoulder"],
    "left_leg": ["left_hip", "left_knee", "left_ankle", "left_foot"],
    "right_leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
}

# 兼容旧版本区域映射
REGION_JOINT_MAPPINGS: Dict[str, List[str]] = {
    **BODY_PART_JOINT_MAPPINGS,
    "upper_body": [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "head", "neck", "chest"
    ],
    "lower_body": [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_foot", "right_foot"
    ],
}


class ViewFeatureExtractor:
    """视角多维观测特征提取器。"""

    def __init__(self, camera_config: Optional[Dict[str, Any]] = None):
        self.camera_config = camera_config or {}
        self.hfov_deg = float(self.camera_config.get("hfov_deg", self.camera_config.get("hfov", 90.0)))
        self.max_distance = float(self.camera_config.get("max_distance", 4.5))
        self.img_width = int(self.camera_config.get("width", 640))
        self.img_height = int(self.camera_config.get("height", 480))

    def extract(
        self,
        viewpoint: CandidateViewpoint,
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
    ) -> ViewFeature:
        """从候选视点与人体 3D 关节姿态中提取精细观测特征。"""
        vp_pos = np.array(viewpoint.position, dtype=np.float32)
        cam_height = float(viewpoint.camera_height)
        cam_pos = np.array([vp_pos[0], vp_pos[1] + cam_height, vp_pos[2]], dtype=np.float32)

        # 1. 距离计算
        pelvis = human_joints_3d.get("pelvis", [vp_pos[0], vp_pos[1] + 0.8, vp_pos[2]])
        pelvis_pos = np.array(pelvis, dtype=np.float32)
        dist = float(np.linalg.norm(cam_pos - pelvis_pos))

        # 2. 视角偏角计算 (正面为 0 度)
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

        # 3. 视锥可见性与关节判定
        half_fov_rad = math.radians(self.hfov_deg / 2.0)
        vp_yaw_rad = math.radians(float(viewpoint.yaw_deg))
        cam_forward = np.array([math.sin(vp_yaw_rad), 0.0, -math.cos(vp_yaw_rad)], dtype=np.float32)

        visible_joints_set = set()
        for j_name, j_pos in human_joints_3d.items():
            jp = np.array(j_pos, dtype=np.float32)
            vec = jp - cam_pos
            vec_horiz = np.array([vec[0], 0.0, vec[2]], dtype=np.float32)
            dist_j = np.linalg.norm(vec_horiz)

            if dist_j > 1e-4:
                dir_j = vec_horiz / dist_j
                cos_a = np.clip(np.dot(dir_j, cam_forward), -1.0, 1.0)
                if math.acos(cos_a) <= half_fov_rad and dist_j <= self.max_distance:
                    visible_joints_set.add(j_name)

        total_joints = max(1, len(human_joints_3d))
        pose_coverage = float(len(visible_joints_set) / total_joints)
        vis_loss = float(np.clip(1.0 - pose_coverage, 0.0, 1.0))

        # 4. 7 大身体关键解剖部位可见性 (Body Part Visibility)
        body_part_visibilities: Dict[str, float] = {}
        for part_name, joint_names in BODY_PART_JOINT_MAPPINGS.items():
            matched_keys = [k for k in human_joints_3d if any(alias in k.lower() for alias in joint_names)]
            if not matched_keys:
                body_part_visibilities[part_name] = round(pose_coverage, 3)
            else:
                vis_in_part = sum(1 for k in matched_keys if k in visible_joints_set)
                body_part_visibilities[part_name] = round(float(vis_in_part / len(matched_keys)), 3)

        # 区域覆盖率 (兼容)
        region_coverages: Dict[str, float] = dict(body_part_visibilities)
        for reg_name in ["upper_body", "lower_body"]:
            joint_names = REGION_JOINT_MAPPINGS[reg_name]
            matched_keys = [k for k in human_joints_3d if any(alias in k.lower() for alias in joint_names)]
            if not matched_keys:
                region_coverages[reg_name] = round(pose_coverage, 3)
            else:
                vis_in_reg = sum(1 for k in matched_keys if k in visible_joints_set)
                region_coverages[reg_name] = round(float(vis_in_reg / len(matched_keys)), 3)

        # 5. 人体投影面积占比估计
        hfov_rad = math.radians(self.hfov_deg)
        fx = float(self.img_width) / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        proj_w = min(float(self.img_width), (0.6 * fx) / max(0.1, dist))
        proj_h = min(float(self.img_height), (1.7 * fy) / max(0.1, dist))
        proj_ratio = round((proj_w * proj_h) / float(self.img_width * self.img_height), 4)

        return ViewFeature(
            viewpoint_id=viewpoint.viewpoint_id,
            distance=round(dist, 3),
            viewing_angle_deg=round(viewing_angle_deg, 1),
            pose_coverage=round(pose_coverage, 3),
            visibility_loss_ratio=round(vis_loss, 3),
            projected_area_ratio=proj_ratio,
            body_part_visibilities=body_part_visibilities,
            region_coverages=region_coverages,
            feasible=viewpoint.feasible,
            metadata=viewpoint.metadata,
        )

    def extract_batch(
        self,
        viewpoints: List[CandidateViewpoint],
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
    ) -> List[ViewFeature]:
        """批量提取候选视点特征。"""
        return [self.extract(vp, human_joints_3d, human_yaw_deg) for vp in viewpoints]
