"""
人体状态估计有效性验证模块 —— state_validation.py
=================================================

功能：
    依据配置阈值与数值有效性（Finite / NaN / Inf）对 EstimatedHumanState 进行全面检验，
    不合格时返回明确的 failure_reason，触发安全兜底，严禁静默回退 GT。
"""

from typing import Optional, Tuple
import numpy as np
from .estimated_human_state import EstimatedHumanState
from .action_pose_library import KEYPOINT_GROUPS


def validate_estimated_state(
    state: EstimatedHumanState,
    config: dict,
) -> Tuple[bool, Optional[str]]:
    """验证 EstimatedHumanState 的有效性。

    参数：
        state: 待验证的人体估计状态。
        config: 主配置字典。

    返回：
        (is_valid, failure_reason)
    """
    est_cfg = config.get("human_state_estimation", {})
    min_2d = est_cfg.get("min_2d_keypoints", 6)
    min_3d = est_cfg.get("min_3d_keypoints", 4)
    min_torso_3d = est_cfg.get("min_torso_keypoints_3d", 2)
    req_pos = est_cfg.get("require_human_position", True)
    req_orient = est_cfg.get("require_orientation", True)

    p_cfg = config.get("perception", {})
    min_pose_score = p_cfg.get("min_pose_score", 0.30)

    # 1. 检查检测分数
    if state.pose_detection_score < min_pose_score:
        return False, f"pose_score_too_low_{state.pose_detection_score:.2f}<{min_pose_score}"

    # 2. 检查 2D 可见关键点数量
    if len(state.visible_2d_keypoints) < min_2d:
        return False, f"insufficient_2d_keypoints_{len(state.visible_2d_keypoints)}<{min_2d}"

    # 3. 检查 3D 观测有效关键点数量
    if len(state.observable_3d_keypoints) < min_3d:
        return False, f"insufficient_3d_keypoints_{len(state.observable_3d_keypoints)}<{min_3d}"

    # 4. 检查 3D 躯干关键点数量
    torso_names = set(KEYPOINT_GROUPS["torso"])
    torso_3d_count = sum(1 for k in state.observable_3d_keypoints if k in torso_names)
    if torso_3d_count < min_torso_3d:
        return False, f"insufficient_torso_3d_keypoints_{torso_3d_count}<{min_torso_3d}"

    # 5. 检查人体位置存在性与有限性 (finite check)
    if req_pos:
        if state.human_position_world is None or state.human_position_source == "invalid":
            return False, "human_position_invalid"
        if not np.all(np.isfinite(state.human_position_world)):
            return False, "human_position_non_finite"

    # 6. 检查人体朝向存在性与有限性 (finite check)
    if req_orient:
        if state.human_yaw is None or state.yaw_source == "invalid":
            return False, "human_orientation_invalid"
        if not np.isfinite(state.human_yaw):
            return False, "human_yaw_non_finite"

    # 7. 检查人体尺度比例有限性与合理范围 [0.4, 2.5]
    if state.body_scale is not None:
        if not np.isfinite(state.body_scale) or not (0.4 <= state.body_scale <= 2.5):
            return False, f"body_scale_out_of_range_{state.body_scale}"

    # 8. 检查 Proxy 骨架各关节坐标有限性
    if state.proxy_full_skeleton:
        for k, pt in state.proxy_full_skeleton.items():
            if pt is None or not np.all(np.isfinite(pt)):
                return False, f"proxy_joint_non_finite_{k}"

    return True, None
