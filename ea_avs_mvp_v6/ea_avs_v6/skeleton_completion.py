"""
骨架补全与 Proxy 骨架构建模块 —— skeleton_completion.py
======================================================

功能：
    结合观测到的 3D 关节、估计的人体中心、朝向（Yaw）与尺度，
    以标准站立姿态（canonical standing template）构建完整的 Proxy 骨架。
    严格记录关键点来源，杜绝静默读取 Humanoid GT 补全。
"""

from typing import Dict, Tuple
import numpy as np

from .action_pose_library import POSE_SKELETONS
from .depth_lifter import EstimatedJoint3D
from .keypoint_schema import EA_AVS_15_KEYPOINTS


class SkeletonCompleter:
    """Proxy 骨架补全器。"""

    def __init__(self, template_name: str = "standing"):
        self.template_name = template_name
        self.canonical_template = POSE_SKELETONS.get(template_name, POSE_SKELETONS["standing"])

    def complete_skeleton(
        self,
        joints_3d: Dict[str, EstimatedJoint3D],
        human_base_pos: np.ndarray,
        human_yaw: float,
        body_scale: float = 1.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, EstimatedJoint3D]]:
        """构建 Proxy 完整骨架与仅观测骨架。

        参数：
            joints_3d: 包含各个关键点观测状态的 EstimatedJoint3D 字典。
            human_base_pos: 估计的人体脚底/基座世界坐标 shape=(3,)。
            human_yaw: 估计的人体朝向角（弧度）。
            body_scale: 估计的人体尺度缩放比例。

        返回：
            (proxy_full_skeleton, observed_skeleton, updated_joints_3d)
            - proxy_full_skeleton: 15 个关键点的世界坐标字典
            - observed_skeleton: 仅包含真实观测到 3D 的关键点字典
            - updated_joints_3d: 补全后的 EstimatedJoint3D 字典副本
        """
        proxy_skeleton: Dict[str, np.ndarray] = {}
        observed_skeleton: Dict[str, np.ndarray] = {}
        updated_joints: Dict[str, EstimatedJoint3D] = {}

        cos_th = np.cos(human_yaw)
        sin_th = np.sin(human_yaw)

        for name in EA_AVS_15_KEYPOINTS:
            joint = joints_3d.get(name)

            # 计算模板位置
            loc = self.canonical_template.get(name, [0.0, 1.0, 0.0])
            x_loc = loc[0] * body_scale
            y_loc = loc[1] * body_scale
            z_loc = loc[2] * body_scale

            # 绕 Y 轴旋转到 human_yaw 并平移至 human_base_pos
            x_rot = x_loc * cos_th + z_loc * sin_th
            y_rot = y_loc
            z_rot = -x_loc * sin_th + z_loc * cos_th

            template_world_pos = human_base_pos + np.array([x_rot, y_rot, z_rot], dtype=np.float64)

            if joint is not None and joint.observable_3d and joint.position_world is not None:
                # 真实观测到的 3D 关键点
                proxy_skeleton[name] = joint.position_world.copy()
                observed_skeleton[name] = joint.position_world.copy()
                updated_joints[name] = joint
            else:
                # 模板补全的关键点
                proxy_skeleton[name] = template_world_pos
                # 构造模板补全的 EstimatedJoint3D 记录
                conf = joint.confidence_2d if joint else 0.0
                updated_joints[name] = EstimatedJoint3D(
                    name=name,
                    position_world=template_world_pos,
                    position_camera=None,
                    confidence_2d=conf,
                    depth_valid=False,
                    observable_3d=False,
                    source="template_completion",
                    uncertainty=None,
                )

        return proxy_skeleton, observed_skeleton, updated_joints
