"""
人体状态估计器 —— human_state_estimator.py
=========================================

功能：
    统一封装从当前视角 RGB-D 观测到 EstimatedHumanState 的完整前端流程。

重要科学约束：
    - 严禁在 estimate() 方法及内部实现中接收或读取任何 Humanoid GT 变量。
    - 所有位置、朝向、尺度与骨架均仅由当前观测数据推导。
"""

from typing import Dict, List, Optional
import numpy as np

from .pose_backend import PoseBackend, create_pose_backend
from .depth_lifter import lift_2d_keypoints_to_3d, EstimatedJoint3D
from .orientation_estimator import OrientationEstimator
from .skeleton_completion import SkeletonCompleter
from .estimated_human_state import EstimatedHumanState
from .state_validation import validate_estimated_state
from .keypoint_schema import EA_AVS_15_KEYPOINTS, KEYPOINT_GROUPS


class HumanStateEstimator:
    """当前 RGB-D 人体状态估计器。"""

    def __init__(self, config: dict, pose_backend: Optional[PoseBackend] = None):
        self.config = config
        self.pose_backend = pose_backend or create_pose_backend(config)
        self.orientation_estimator = OrientationEstimator(config)
        template_name = (
            config.get("human_state_estimation", {})
            .get("skeleton_completion", {})
            .get("template", "standing")
        )
        self.skeleton_completer = SkeletonCompleter(template_name=template_name)
        self.est_cfg = config.get("human_state_estimation", {})

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_state: dict,
    ) -> EstimatedHumanState:
        """从当前 RGB-D 观测和相机几何状态估计人体状态。

        参数：
            rgb: shape=(H, W, 3) 的 RGB 图像。
            depth: shape=(H, W) 的公制深度图。
            camera_state: 包含相机位姿与内参的字典：
                {
                    "position": np.ndarray shape=(3,),
                    "yaw": float,
                    "camera_height": float,
                    "intrinsics": dict {"fx", "fy", "cx", "cy", "width", "height"},
                }

        返回：
            EstimatedHumanState 实例。
        """
        camera_base_pos = np.asarray(camera_state["position"], dtype=np.float64)
        camera_yaw = float(camera_state["yaw"])
        camera_height = float(camera_state["camera_height"])
        intrinsics = camera_state["intrinsics"]

        # -------------------------------------------------------------
        # 1. 2D 姿态检测
        # -------------------------------------------------------------
        detections = self.pose_backend.infer(rgb)
        if not detections:
            return EstimatedHumanState(
                valid=False,
                human_position_world=None,
                human_yaw=None,
                body_scale=None,
                failure_reason="no_person_detected",
            )

        best_det = detections[0]  # 取最高置信度检测结果
        keypoints_2d = best_det.keypoints

        # -------------------------------------------------------------
        # 2. 深度采样与 3D 逆投影
        # -------------------------------------------------------------
        joints_3d = lift_2d_keypoints_to_3d(
            keypoints_2d=keypoints_2d,
            depth_img=depth,
            intrinsics=intrinsics,
            camera_base_pos=camera_base_pos,
            camera_yaw=camera_yaw,
            camera_height=camera_height,
            depth_config=self.est_cfg,
        )

        visible_2d = [k for k, v in keypoints_2d.items() if v.detected]
        observable_3d = [k for k, v in joints_3d.items() if v.observable_3d]
        missing = [k for k in EA_AVS_15_KEYPOINTS if k not in observable_3d]

        # -------------------------------------------------------------
        # 3. 人体基座/世界位置估计
        # -------------------------------------------------------------
        human_pos, pos_conf, pos_src = self._estimate_human_position(joints_3d)

        # -------------------------------------------------------------
        # 4. 人体朝向 (Yaw) 估计
        # -------------------------------------------------------------
        orient_res = self.orientation_estimator.estimate_orientation(joints_3d)
        human_yaw = orient_res.yaw_rad
        yaw_conf = orient_res.confidence
        yaw_src = orient_res.source

        # -------------------------------------------------------------
        # 5. 人体尺度比例估计
        # -------------------------------------------------------------
        body_scale, scale_conf, scale_src = self._estimate_body_scale(joints_3d)

        # -------------------------------------------------------------
        # 6. Proxy 骨架补全构建
        # -------------------------------------------------------------
        if human_pos is not None and human_yaw is not None:
            proxy_skel, obs_skel, updated_joints = self.skeleton_completer.complete_skeleton(
                joints_3d=joints_3d,
                human_base_pos=human_pos,
                human_yaw=human_yaw,
                body_scale=body_scale or 1.0,
            )
        else:
            proxy_skel = {}
            obs_skel = {k: v.position_world for k, v in joints_3d.items() if v.position_world is not None}
            updated_joints = joints_3d

        template_completed = [k for k in EA_AVS_15_KEYPOINTS if k not in observable_3d]

        # -------------------------------------------------------------
        # 7. 整体状态置信度综合
        # -------------------------------------------------------------
        state_conf = float(
            0.4 * best_det.score
            + 0.3 * (len(observable_3d) / float(len(EA_AVS_15_KEYPOINTS)))
            + 0.3 * yaw_conf
        )

        state = EstimatedHumanState(
            valid=True,
            human_position_world=human_pos,
            human_position_confidence=pos_conf,
            human_position_source=pos_src,
            human_yaw=human_yaw,
            yaw_confidence=yaw_conf,
            yaw_source=yaw_src,
            body_scale=body_scale,
            body_scale_confidence=scale_conf,
            body_scale_source=scale_src,
            joints=updated_joints,
            observed_skeleton=obs_skel,
            proxy_full_skeleton=proxy_skel,
            visible_2d_keypoints=visible_2d,
            observable_3d_keypoints=observable_3d,
            missing_keypoints=missing,
            template_completed_keypoints=template_completed,
            pose_detection_score=best_det.score,
            state_confidence=state_conf,
            failure_reason=None,
        )

        # -------------------------------------------------------------
        # 8. 状态有效性检验
        # -------------------------------------------------------------
        is_valid, fail_reason = validate_estimated_state(state, self.config)
        state.valid = is_valid
        state.failure_reason = fail_reason

        return state

    def _estimate_human_position(
        self,
        joints_3d: Dict[str, EstimatedJoint3D],
    ) -> tuple:
        """从 3D 关节估计人体脚底/基座世界坐标。"""
        # 1. 优先使用 pelvis (站立标准高约 0.95m)
        pelvis = joints_3d.get("pelvis")
        if pelvis and pelvis.observable_3d and pelvis.position_world is not None:
            base_pos = pelvis.position_world.copy()
            base_pos[1] -= 0.95
            return base_pos, pelvis.confidence_2d, "pelvis"

        # 2. 左右髋中点 (站立标准高约 0.90m)
        l_hip = joints_3d.get("left_hip")
        r_hip = joints_3d.get("right_hip")
        if (
            l_hip and r_hip and l_hip.observable_3d and r_hip.observable_3d
            and l_hip.position_world is not None and r_hip.position_world is not None
        ):
            mid_hip = (l_hip.position_world + r_hip.position_world) / 2.0
            base_pos = mid_hip.copy()
            base_pos[1] -= 0.90
            conf = (l_hip.confidence_2d + r_hip.confidence_2d) / 2.0
            return base_pos, conf, "hip_midpoint"

        # 3. 躯干关节均值 (站立躯干中心约 1.15m)
        torso_pts = [
            joints_3d[k].position_world
            for k in KEYPOINT_GROUPS["torso"]
            if k in joints_3d and joints_3d[k].observable_3d and joints_3d[k].position_world is not None
        ]
        if len(torso_pts) >= 2:
            center = np.mean(torso_pts, axis=0)
            base_pos = center.copy()
            base_pos[1] -= 1.15
            return base_pos, 0.6, "torso_mean"

        # 4. 全体可见关节中位数
        all_pts = [
            j.position_world
            for j in joints_3d.values()
            if j.observable_3d and j.position_world is not None
        ]
        if len(all_pts) >= 4:
            center = np.median(all_pts, axis=0)
            min_y = float(np.min([p[1] for p in all_pts]))
            base_pos = np.array([center[0], min_y, center[2]], dtype=np.float64)
            return base_pos, 0.4, "observable_median"

        return None, 0.0, "invalid"

    def _estimate_body_scale(
        self,
        joints_3d: Dict[str, EstimatedJoint3D],
    ) -> tuple:
        """从观测到的双侧对称关节间距估计人体尺度比例。"""
        scales = []
        l_sh = joints_3d.get("left_shoulder")
        r_sh = joints_3d.get("right_shoulder")
        if (
            l_sh and r_sh and l_sh.observable_3d and r_sh.observable_3d
            and l_sh.position_world is not None and r_sh.position_world is not None
        ):
            w = float(np.linalg.norm(l_sh.position_world - r_sh.position_world))
            if 0.2 < w < 0.8:
                scales.append(w / 0.44)  # 0.44m 为标准肩宽

        l_hip = joints_3d.get("left_hip")
        r_hip = joints_3d.get("right_hip")
        if (
            l_hip and r_hip and l_hip.observable_3d and r_hip.observable_3d
            and l_hip.position_world is not None and r_hip.position_world is not None
        ):
            w = float(np.linalg.norm(l_hip.position_world - r_hip.position_world))
            if 0.15 < w < 0.6:
                scales.append(w / 0.32)  # 0.32m 为标准髋宽

        neck = joints_3d.get("neck")
        pelvis = joints_3d.get("pelvis")
        if (
            neck and pelvis and neck.observable_3d and pelvis.observable_3d
            and neck.position_world is not None and pelvis.position_world is not None
        ):
            h = float(np.linalg.norm(neck.position_world - pelvis.position_world))
            if 0.2 < h < 0.8:
                scales.append(h / 0.45)  # 0.45m 为标准颈部-骨盆长度

        if scales:
            median_scale = float(np.median(scales))
            clamped = float(np.clip(median_scale, 0.7, 1.4))
            return clamped, 0.8, "joint_distances"
        else:
            return 1.0, 0.5, "default_template"
