"""
骨架格式统一、置信度融合与部位可见性评估 —— skeleton_converter.py
============================================================

职责：
    1. 融合 2D 关键点检测置信度 (conf_2d) 与深度空间连续性置信度 (conf_depth)：
       conf_composite = conf_2d * conf_depth
    2. 将 COCO-17 关键点规范化转换为支持动作识别 (ST-GCN / 运动学拓扑) 的标准骨架表示；
    3. 合成关键骨架根节点 (Pelvis 骨盆中心、Neck 颈部中心、Spine 脊柱)；
    4. 执行遮挡与不确定性判定 (Occlusion Flagging)，低于阈值的关节判定为不可见/遮挡；
    5. 输出结构化的 EstimatedSkeleton3D 实体对象。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .depth_projection import DepthProjectionResult
from .pose_estimator import COCO_KEYPOINTS, Pose2DResult

logger = logging.getLogger(__name__)

# 标准 16 关键点拓扑定义 (ST-GCN / NTU-RGBD / Humanoid 标准格式)
UNIFIED_JOINT_NAMES: List[str] = [
    "pelvis",          # 0 (合成: left_hip 与 right_hip 中心)
    "spine",           # 1 (合成: 躯干中心)
    "neck",            # 2 (合成: left_shoulder 与 right_shoulder 中心)
    "head",            # 3 (来自 nose)
    "left_shoulder",   # 4
    "left_elbow",      # 5
    "left_wrist",      # 6
    "right_shoulder",  # 7
    "right_elbow",     # 8
    "right_wrist",     # 9
    "left_hip",        # 10
    "left_knee",       # 11
    "left_ankle",      # 12
    "right_hip",       # 13
    "right_knee",      # 14
    "right_ankle",     # 15
]

# 统一 16 骨架运动学父子拓扑连接 (用于图卷积邻接矩阵与 3D 渲染)
UNIFIED_SKELETON_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),        # 躯干脊柱中心轴
    (2, 4), (4, 5), (5, 6),        # 左上肢 (neck -> shoulder -> elbow -> wrist)
    (2, 7), (7, 8), (8, 9),        # 右上肢 (neck -> shoulder -> elbow -> wrist)
    (0, 10), (10, 11), (11, 12),   # 左下肢 (pelvis -> hip -> knee -> ankle)
    (0, 13), (13, 14), (14, 15),   # 右下肢 (pelvis -> hip -> knee -> ankle)
]

# 6 大部位关节映射
BODY_PART_GROUPS: Dict[str, List[int]] = {
    "head": [2, 3],
    "torso": [0, 1, 2],
    "left_arm": [4, 5, 6],
    "right_arm": [7, 8, 9],
    "left_leg": [10, 11, 12],
    "right_leg": [13, 14, 15],
}


@dataclass
class EstimatedSkeleton3D:
    """估计的 3D 人体骨架与感知质量结构体。"""
    joints_3d_cam: np.ndarray             # 形状 (16, 3) 相机系坐标
    joints_3d_world: np.ndarray           # 形状 (16, 3) 世界系坐标
    joints_2d: np.ndarray                 # 形状 (16, 2) 图像系像素坐标
    confidence: np.ndarray                # 形状 (16,) 融合置信度 [0.0, 1.0]
    occluded_mask: np.ndarray             # 形状 (16,) bool (True=遮挡/低置信)
    part_confidence: Dict[str, float]     # 6 大部位平均置信度
    joint_names: List[str] = field(default_factory=lambda: list(UNIFIED_JOINT_NAMES))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joints_3d_cam": self.joints_3d_cam.tolist(),
            "joints_3d_world": self.joints_3d_world.tolist(),
            "joints_2d": self.joints_2d.tolist(),
            "confidence": self.confidence.tolist(),
            "occluded_mask": self.occluded_mask.tolist(),
            "part_confidence": {k: float(v) for k, v in self.part_confidence.items()},
            "joint_names": self.joint_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstimatedSkeleton3D":
        return cls(
            joints_3d_cam=np.array(data["joints_3d_cam"], dtype=np.float32),
            joints_3d_world=np.array(data["joints_3d_world"], dtype=np.float32),
            joints_2d=np.array(data["joints_2d"], dtype=np.float32),
            confidence=np.array(data["confidence"], dtype=np.float32),
            occluded_mask=np.array(data["occluded_mask"], dtype=bool),
            part_confidence=data.get("part_confidence", {}),
            joint_names=data.get("joint_names", list(UNIFIED_JOINT_NAMES)),
        )


class SkeletonConverter:
    """骨架格式转换与置信度融合器。"""

    def __init__(self, occlusion_conf_thresh: float = 0.35):
        self.occlusion_conf_thresh = float(occlusion_conf_thresh)

    def convert_and_fuse(
        self,
        pose2d: Pose2DResult,
        depth_res: DepthProjectionResult,
    ) -> EstimatedSkeleton3D:
        """
        融合 2D 姿态检测与 3D 深度反投影，生成标准 16 节点 3D 骨架。

        Args:
            pose2d: 2D 姿态检测结果 (COCO-17)
            depth_res: 深度反投影结果 (COCO-17 对应 3D 坐标与深度置信度)

        Returns:
            EstimatedSkeleton3D
        """
        kpts_2d_coco = pose2d.keypoints          # (17, 2)
        conf_2d_coco = pose2d.confidence         # (17,)
        joints_cam_coco = depth_res.joints_3d_cam # (17, 3)
        joints_world_coco = depth_res.joints_3d_world # (17, 3)
        conf_depth_coco = depth_res.depth_confidence  # (17,)

        # 1. 逐关节复合置信度计算: c = c_2d * c_depth
        conf_composite_coco = conf_2d_coco * conf_depth_coco  # (17,)

        # 2. 映射构建 16 关键点拓扑
        num_joints = len(UNIFIED_JOINT_NAMES)  # 16
        unified_cam = np.zeros((num_joints, 3), dtype=np.float32)
        unified_world = np.zeros((num_joints, 3), dtype=np.float32)
        unified_2d = np.zeros((num_joints, 2), dtype=np.float32)
        unified_conf = np.zeros(num_joints, dtype=np.float32)

        # COCO 索引映射
        # 0: nose, 5: l_sh, 6: r_sh, 7: l_el, 8: r_el, 9: l_wr, 10: r_wr
        # 11: l_hip, 12: r_hip, 13: l_kn, 14: r_kn, 15: l_ank, 16: r_ank

        # 0: pelvis (合成: 11 与 12 的中点)
        unified_cam[0] = (joints_cam_coco[11] + joints_cam_coco[12]) / 2.0
        unified_world[0] = (joints_world_coco[11] + joints_world_coco[12]) / 2.0
        unified_2d[0] = (kpts_2d_coco[11] + kpts_2d_coco[12]) / 2.0
        unified_conf[0] = (conf_composite_coco[11] + conf_composite_coco[12]) / 2.0

        # 2: neck (合成: 5 与 6 的中点)
        unified_cam[2] = (joints_cam_coco[5] + joints_cam_coco[6]) / 2.0
        unified_world[2] = (joints_world_coco[5] + joints_world_coco[6]) / 2.0
        unified_2d[2] = (kpts_2d_coco[5] + kpts_2d_coco[6]) / 2.0
        unified_conf[2] = (conf_composite_coco[5] + conf_composite_coco[6]) / 2.0

        # 1: spine (合成: pelvis 与 neck 的中点)
        unified_cam[1] = (unified_cam[0] + unified_cam[2]) / 2.0
        unified_world[1] = (unified_world[0] + unified_world[2]) / 2.0
        unified_2d[1] = (unified_2d[0] + unified_2d[2]) / 2.0
        unified_conf[1] = (unified_conf[0] + unified_conf[2]) / 2.0

        # 3: head (来自 nose 0)
        unified_cam[3] = joints_cam_coco[0]
        unified_world[3] = joints_world_coco[0]
        unified_2d[3] = kpts_2d_coco[0]
        unified_conf[3] = conf_composite_coco[0]

        # 直接映射的肢体关键点
        direct_map = [
            (4, 5),   # left_shoulder <- COCO 5
            (5, 7),   # left_elbow    <- COCO 7
            (6, 9),   # left_wrist    <- COCO 9
            (7, 6),   # right_shoulder<- COCO 6
            (8, 8),   # right_elbow   <- COCO 8
            (9, 10),  # right_wrist   <- COCO 10
            (10, 11), # left_hip      <- COCO 11
            (11, 13), # left_knee     <- COCO 13
            (12, 15), # left_ankle    <- COCO 15
            (13, 12), # right_hip     <- COCO 12
            (14, 14), # right_knee    <- COCO 14
            (15, 16), # right_ankle   <- COCO 16
        ]

        for u_idx, c_idx in direct_map:
            unified_cam[u_idx] = joints_cam_coco[c_idx]
            unified_world[u_idx] = joints_world_coco[c_idx]
            unified_2d[u_idx] = kpts_2d_coco[c_idx]
            unified_conf[u_idx] = conf_composite_coco[c_idx]

        # 3. 遮挡与低置信度掩码判定
        occluded_mask = unified_conf < self.occlusion_conf_thresh

        # 4. 计算 6 大部位平均置信度
        part_conf: Dict[str, float] = {}
        for part_name, joint_indices in BODY_PART_GROUPS.items():
            sub_confs = [unified_conf[j] for j in joint_indices]
            part_conf[part_name] = float(np.mean(sub_confs))

        return EstimatedSkeleton3D(
            joints_3d_cam=unified_cam,
            joints_3d_world=unified_world,
            joints_2d=unified_2d,
            confidence=unified_conf,
            occluded_mask=occluded_mask,
            part_confidence=part_conf,
            joint_names=list(UNIFIED_JOINT_NAMES),
        )
