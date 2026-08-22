"""
GT 与估计骨架空间对齐与映射器 —— skeleton_alignment.py
=====================================================

职责：
    1. 将 Habitat / AMASS 世界坐标系 GT 骨骼通过相机外参反变换到相机坐标系；
    2. 基于 `configs/skeleton_definition.json` 中的 `gt_joint_mapping` 建立估计关节与 GT 关节的无硬编码动态映射；
    3. 支持根节点相对对齐 (Root-Relative Alignment)；
    4. 支持 Procrustes 分析 (Rigid Procrustes Alignment) 计算最优旋转、平移与尺度缩放，供 PA-MPJPE 评测。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ea_avs_mvp_v10.core.types import CameraPose
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


def transform_gt_to_camera_frame(
    gt_joints_dict: Dict[str, List[float]],
    camera_matrix_4x4: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    将 Habitat 世界坐标系下的 GT 骨骼字典转换至相机坐标系。

    Habitat 相机坐标系约定为 OpenGL 坐标系 (+X: 右, +Y: 上, -Z: 前/视线)，
    ACTIVEVIEW 相机坐标系约定为标准右手深度系 (+X: 右, +Y: 上, +Z: 前/深度)。
    因此 Z 轴取反转换至标准深度系。
    """
    c2w = np.array(camera_matrix_4x4, dtype=np.float32)
    w2c = np.linalg.inv(c2w)

    gt_cam_dict: Dict[str, np.ndarray] = {}
    for name, pos in gt_joints_dict.items():
        p_world_homo = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
        p_c = (w2c @ p_world_homo)[:3]
        # Habitat 视线沿 -Z 轴，转换为标准相机系 +Z 为深度
        p_activeview_cam = np.array([p_c[0], p_c[1], -p_c[2]], dtype=np.float32)
        gt_cam_dict[name] = p_activeview_cam

    return gt_cam_dict


def extract_aligned_joint_pairs(
    estimated_skeleton: EstimatedSkeleton3D,
    gt_joints_dict: Dict[str, List[float]],
    camera_matrix_4x4: np.ndarray,
    skel_def: Optional[SkeletonDefinition] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    基于 Schema 映射提取匹配的估计关节点矩阵与 GT 关节点矩阵 (N, 3)。

    Returns:
        P_est: (N, 3) 估计关节相机坐标
        P_gt: (N, 3) GT 关节相机坐标
        joint_names: 匹配的关节名称列表
    """
    skel_def = skel_def or get_skeleton_definition()
    gt_cam = transform_gt_to_camera_frame(gt_joints_dict, camera_matrix_4x4)
    gt_mapping = skel_def.gt_joint_mapping

    p_est_list: List[np.ndarray] = []
    p_gt_list: List[np.ndarray] = []
    names_list: List[str] = []

    for gt_name, est_target in gt_mapping.items():
        if est_target is None or gt_name not in gt_cam:
            continue

        if est_target == "hip_center":
            # 骨盆中心
            root_idx = skel_def.root_joints
            p_est = np.mean(estimated_skeleton.joints_3d_camera[root_idx], axis=0)
        elif isinstance(est_target, int):
            p_est = estimated_skeleton.joints_3d_camera[est_target]
        else:
            continue

        p_gt = gt_cam[gt_name]
        p_est_list.append(p_est)
        p_gt_list.append(p_gt)
        names_list.append(gt_name)

    return np.array(p_est_list, dtype=np.float32), np.array(p_gt_list, dtype=np.float32), names_list


def compute_procrustes_alignment(
    P_est: np.ndarray,
    P_gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    使用 Umeyama 算法计算估计点集到 GT 点集的最优 Procrustes 刚体变换对齐 (R, t, s)。

    Args:
        P_est: (N, 3) 估计点集
        P_gt: (N, 3) 真值点集

    Returns:
        P_aligned: (N, 3) 对齐后的估计点集 s * P_est @ R.T + t
        R: (3, 3) 正交旋转矩阵
        s: float 尺度缩放因子
        t: (3,) 平移向量
    """
    assert P_est.shape == P_gt.shape, "Point sets must have identical shapes"
    n, m = P_est.shape

    mu_est = np.mean(P_est, axis=0)
    mu_gt = np.mean(P_gt, axis=0)

    P_est_c = P_est - mu_est
    P_gt_c = P_gt - mu_gt

    var_est = np.sum(P_est_c ** 2) / n

    # 计算协方差矩阵
    H = (P_gt_c.T @ P_est_c) / n

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    # 保证右手系旋转行列式为 +1
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U @ Vt

    s = float(np.sum(S) / (var_est + 1e-8))
    t = mu_gt - s * (R @ mu_est)

    P_aligned = (s * (R @ P_est.T)).T + t
    return P_aligned.astype(np.float32), R.astype(np.float32), s, t.astype(np.float32)
