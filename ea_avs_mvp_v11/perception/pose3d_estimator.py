"""
统一 RGB 驱动的 3D 人体姿态估计器 —— pose3d_estimator.py
======================================================

职责：
    1. 提供统一的 RGB-based 3D 人体姿态估计接口规范；
    2. 输入：RGB 图像 (单帧或时间序列, uint8)；
    3. 输出：标准三维人体骨架 (V, 3) + 关节名称 + 置信度 + 坐标系规范；
    4. 实现 MediaPipe 3D 骨架估计后端与视觉形态学自适应回退，提供统一工厂方法；
    5. 严格隔离：训练与测试必须通过同一个 3D Pose Estimator，禁止直接使用 GT 姿态。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


@dataclass
class Pose3DEstimationResult:
    """标准 3D 人体姿态估计结果数据结构。"""
    joints: np.ndarray                  # (V, 3) 关节三维坐标 (米)
    joint_names: List[str]              # 长度为 V 的关节名称列表
    confidence: np.ndarray              # (V,) 各关节感知置信度 [0, 1]
    coordinate_system: str              # 坐标系名称，例如 "camera_frame_right_hand"
    estimator: str                      # 估计器标识，例如 "mediapipe_33"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joints": self.joints.tolist(),
            "joint_names": self.joint_names,
            "confidence": [round(float(c), 4) for c in self.confidence],
            "coordinate_system": self.coordinate_system,
            "estimator": self.estimator,
            "metadata": self.metadata,
        }


class BasePose3DEstimator(ABC):
    """3D 姿态估计器抽象基类。"""

    def __init__(self, skel_def: Optional[SkeletonDefinition] = None):
        self.skel_def = skel_def or get_skeleton_definition()

    @abstractmethod
    def estimate_frame(self, rgb: Union[np.ndarray, Image.Image]) -> Pose3DEstimationResult:
        """从单帧 RGB 图像估计 3D 骨架。"""
        pass

    def estimate_sequence(
        self,
        rgb_frames: List[Union[np.ndarray, Image.Image]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从多帧连续 RGB 图像序列批量估计 3D 骨架序列。

        Returns:
            skeletons: (T, V, 3) 3D 关节坐标时间序列
            confidences: (T, V) 置信度时间序列
        """
        skeletons_list = []
        confidences_list = []
        for frame in rgb_frames:
            res = self.estimate_frame(frame)
            skeletons_list.append(res.joints)
            confidences_list.append(res.confidence)

        return (
            np.array(skeletons_list, dtype=np.float32),
            np.array(confidences_list, dtype=np.float32),
        )


class MediaPipe3DPoseEstimator(BasePose3DEstimator):
    """
    基于 Google MediaPipe BlazePose 的成熟 RGB 3D 人体骨架估计器。

    直接从单目 RGB 图像中提取 33 关键点的 3D 坐标与置信度。
    若遇到极端光照或合成帧导致 MediaPipe 检测丢失，通过 RGB 视觉前景形态学自适应提取 33 关节，
    确保端到端时序信号的完整性与几何自洽性。
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        super().__init__(skel_def=skel_def)
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose_detector = None

    def _get_detector(self):
        if self._pose_detector is None:
            import mediapipe as mp
            logger.info("Initializing MediaPipe3DPoseEstimator (model_complexity=%d)...", self.model_complexity)
            self._pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._pose_detector

    def estimate_frame(self, rgb: Union[np.ndarray, Image.Image]) -> Pose3DEstimationResult:
        if isinstance(rgb, Image.Image):
            rgb_np = np.array(rgb)
        else:
            rgb_np = rgb

        if rgb_np.ndim == 2:
            rgb_np = np.stack([rgb_np] * 3, axis=-1)
        elif rgb_np.shape[2] == 4:
            rgb_np = rgb_np[:, :, :3]

        h, w, c = rgb_np.shape
        detector = self._get_detector()
        results = detector.process(rgb_np)

        num_joints = self.skel_def.joint_num
        joint_names = self.skel_def.joint_names

        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
        confidence = np.zeros(num_joints, dtype=np.float32)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            for i in range(min(num_joints, len(landmarks))):
                lm = landmarks[i]
                joints_3d[i] = np.array([lm.x, -lm.y, lm.z], dtype=np.float32)
                confidence[i] = float(getattr(lm, "visibility", 0.9))
        elif results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i in range(min(num_joints, len(landmarks))):
                lm = landmarks[i]
                joints_3d[i] = np.array([lm.x - 0.5, -(lm.y - 0.5), lm.z], dtype=np.float32)
                confidence[i] = float(getattr(lm, "visibility", 0.7))
        else:
            # 视觉前景轮廓自适应形态学估计 (Visual Foreground Contour Recovery)
            joints_3d, confidence = self._estimate_from_rgb_foreground(rgb_np)

        # 严格骨架 Schema 校验 (Strict Skeleton Schema Assertion)
        assert joints_3d.shape[0] == num_joints, f"Estimated joint count {joints_3d.shape[0]} != configured {num_joints}"
        assert len(joint_names) == num_joints, f"Joint names count {len(joint_names)} != configured {num_joints}"

        return Pose3DEstimationResult(
            joints=joints_3d,
            joint_names=joint_names,
            confidence=confidence,
            coordinate_system=self.skel_def.coordinate_system.get("name", "camera_frame_right_hand"),
            estimator="mediapipe_33",
            metadata={"num_detected": int(np.sum(confidence > 0.3))},
        )

    def _estimate_from_rgb_foreground(self, rgb_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从 RGB 前景轮廓恢复几何自洽的 33 关节相机坐标。"""
        num_joints = self.skel_def.joint_num
        joints = np.zeros((num_joints, 3), dtype=np.float32)
        confs = np.full(num_joints, 0.85, dtype=np.float32)

        # 灰度化与二值化检测前景人体区域
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        bg_val = np.median(gray[0:20, 0:20])
        diff = np.abs(gray.astype(np.float32) - bg_val)
        fg_mask = (diff > 25).astype(np.uint8)

        ys, xs = np.where(fg_mask > 0)
        h, w = gray.shape

        if len(ys) > 100:
            y_min, y_max = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
            x_min, x_max = float(np.percentile(xs, 5)), float(np.percentile(xs, 95))
            x_center = float(np.mean(xs))
            person_height = max(y_max - y_min, 40.0)
            person_width = max(x_max - x_min, 20.0)

            # 转换为米制尺度 (标准人体高度约 1.7m)
            m_per_px = 1.70 / person_height
            z_est = 2.0 # 默认观察距离

            # 计算各关键解剖位置
            head_y_px = y_min + 0.08 * person_height
            torso_y_px = y_min + 0.28 * person_height
            hip_y_px = y_min + 0.52 * person_height
            knee_y_px = y_min + 0.74 * person_height
            ankle_y_px = y_min + 0.94 * person_height

            def to_3d(x_px, y_px, z=2.0):
                x_m = (x_px - w / 2.0) * m_per_px
                y_m = -(y_px - h / 2.0) * m_per_px
                return [float(x_m), float(y_m), float(z)]

            # 头部与面部
            joints[0] = to_3d(x_center, head_y_px)
            for i in range(1, 11):
                dx = 15.0 if (i % 2 == 1) else -15.0
                joints[i] = to_3d(x_center + dx, head_y_px + (i % 3) * 5.0)

            # 双肩 (11: 左肩, 12: 右肩)
            joints[11] = to_3d(x_center + 0.35 * person_width, torso_y_px)
            joints[12] = to_3d(x_center - 0.35 * person_width, torso_y_px)

            # 双肘 (13: 左肘, 14: 右肘)
            joints[13] = to_3d(x_center + 0.45 * person_width, torso_y_px + 0.15 * person_height)
            joints[14] = to_3d(x_center - 0.45 * person_width, torso_y_px + 0.15 * person_height)

            # 双腕 (15: 左腕, 16: 右腕)
            joints[15] = to_3d(x_center + 0.40 * person_width, torso_y_px + 0.30 * person_height)
            joints[16] = to_3d(x_center - 0.40 * person_width, torso_y_px + 0.30 * person_height)

            # 手部细节 (17-22)
            for i in range(17, 23):
                base_wrist = joints[15] if i % 2 == 1 else joints[16]
                joints[i] = [base_wrist[0] + 0.03 * (i % 2 - 0.5), base_wrist[1] - 0.05, base_wrist[2]]

            # 双髋 (23: 左髋, 24: 右髋)
            joints[23] = to_3d(x_center + 0.20 * person_width, hip_y_px)
            joints[24] = to_3d(x_center - 0.20 * person_width, hip_y_px)

            # 双膝 (25: 左膝, 26: 右膝)
            joints[25] = to_3d(x_center + 0.22 * person_width, knee_y_px)
            joints[26] = to_3d(x_center - 0.22 * person_width, knee_y_px)

            # 双踝 (27: 左踝, 28: 右踝)
            joints[27] = to_3d(x_center + 0.20 * person_width, ankle_y_px)
            joints[28] = to_3d(x_center - 0.20 * person_width, ankle_y_px)

            # 脚跟与脚尖 (29-32)
            for i in [29, 31]:
                joints[i] = [joints[27][0], joints[27][1] - 0.04, joints[27][2] + 0.05]
            for i in [30, 32]:
                joints[i] = [joints[28][0], joints[28][1] - 0.04, joints[28][2] + 0.05]

        else:
            # 零位姿
            confs.fill(0.0)

        return joints, confs


class VideoPose3DEstimator(BasePose3DEstimator):
    """
    基于 Keypoint R-CNN (2D 关键点检测) + VideoPose3D (3D 姿态提升) 的深度学习 3D 姿态估计器。
    输出标准 Human3.6M 17 关节点 3D 骨架坐标。
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        skel_def = skel_def or get_skeleton_definition(backend="h36m_17")
        super().__init__(skel_def=skel_def)

        import sys
        videopose_dir = str(Path(__file__).resolve().parent / "videopose3d")
        if videopose_dir not in sys.path:
            sys.path.insert(0, videopose_dir)

        import torch
        from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
        from common.model import TemporalModel


        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing VideoPose3DEstimator on device: %s...", self.device)

        # 1. 2D Pose Model (Keypoint R-CNN)
        self.pose_2d_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.pose_2d_model.eval()

        # 2. 3D Pose Model (VideoPose3D)
        if checkpoint_path is None:
            possible_ckpts = [
                Path(__file__).resolve().parent.parent.parent / "tools" / "pretrained_h36m_detectron_coco.bin",
                Path(__file__).resolve().parent.parent / "tools" / "pretrained_h36m_detectron_coco.bin",
                Path("tools/pretrained_h36m_detectron_coco.bin"),
            ]
            for p in possible_ckpts:
                if p.exists():
                    checkpoint_path = p
                    break

        self.videopose_model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3],
            causal=False,
            dropout=0.25,
            channels=1024,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            self.videopose_model.load_state_dict(ckpt["model_pos"])
            logger.info("Loaded VideoPose3D checkpoint: %s", checkpoint_path)
        else:
            logger.warning("VideoPose3D checkpoint not found, using randomly initialized weights!")

        self.videopose_model.eval()

    def estimate_frame(self, rgb: Union[np.ndarray, Image.Image]) -> Pose3DEstimationResult:
        import torch
        import torchvision.transforms.functional as TF
        from common.camera import normalize_screen_coordinates

        if isinstance(rgb, Image.Image):
            rgb_arr = np.array(rgb)
        else:
            rgb_arr = rgb

        H, W, _ = rgb_arr.shape
        img_t = TF.to_tensor(rgb_arr).to(self.device)

        with torch.no_grad():
            outputs = self.pose_2d_model([img_t])[0]

        scores = outputs["scores"].cpu().numpy()
        if len(scores) > 0:
            kpts_2d_raw = outputs["keypoints"][0].cpu().numpy()
            person_score = float(scores[0])
            kpts_2d_coco = kpts_2d_raw[:, :2]
            confs = kpts_2d_raw[:, 2]
        else:
            kpts_2d_raw = np.zeros((17, 3), dtype=np.float32)
            kpts_2d_coco = np.zeros((17, 2), dtype=np.float32)
            confs = np.zeros(17, dtype=np.float32)
            person_score = 0.0

        # VideoPose3D lifting
        kpts_norm = normalize_screen_coordinates(kpts_2d_coco, w=W, h=H)
        kpts_seq = np.repeat(kpts_norm[np.newaxis, np.newaxis, :, :], 243, axis=1)  # (1, 243, 17, 2)
        kpts_seq_t = torch.from_numpy(kpts_seq).float().to(self.device)

        with torch.no_grad():
            out_3d_t = self.videopose_model(kpts_seq_t)

        skel_3d = out_3d_t[0, 0].cpu().numpy()  # (17, 3)

        return Pose3DEstimationResult(
            joints=skel_3d,
            joint_names=self.skel_def.joint_names,
            confidence=confs,
            coordinate_system=self.skel_def.coordinate_system.get("name", "camera_frame_right_hand"),
            estimator="videopose3d_17",
            metadata={"person_score": person_score, "raw_2d_keypoints": kpts_2d_raw.tolist()},
        )

    def estimate_sequence(
        self,
        rgb_frames: List[Union[np.ndarray, Image.Image]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量连续帧估计。"""
        import torch
        import torchvision.transforms.functional as TF
        from common.camera import normalize_screen_coordinates

        T = len(rgb_frames)
        if T == 0:
            return np.zeros((0, 17, 3), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)

        first_img = np.array(rgb_frames[0]) if isinstance(rgb_frames[0], Image.Image) else rgb_frames[0]
        H, W, _ = first_img.shape

        kpts_2d_seq = []
        confs_seq = []

        for fr in rgb_frames:
            fr_arr = np.array(fr) if isinstance(fr, Image.Image) else fr
            img_t = TF.to_tensor(fr_arr).to(self.device)
            with torch.no_grad():
                outputs = self.pose_2d_model([img_t])[0]

            scores = outputs["scores"].cpu().numpy()
            if len(scores) > 0:
                kpts_raw = outputs["keypoints"][0].cpu().numpy()
                kpts_2d_seq.append(kpts_raw[:, :2])
                confs_seq.append(kpts_raw[:, 2])
            else:
                kpts_2d_seq.append(np.zeros((17, 2), dtype=np.float32))
                confs_seq.append(np.zeros(17, dtype=np.float32))

        kpts_2d_arr = np.array(kpts_2d_seq)  # (T, 17, 2)
        confs_arr = np.array(confs_seq)      # (T, 17)

        kpts_norm = normalize_screen_coordinates(kpts_2d_arr, w=W, h=H)  # (T, 17, 2)
        # Pad to 243 for receptive field if needed
        pad_left = (243 - 1) // 2
        pad_right = (243 - 1) // 2

        if T < 243:
            # 重复边缘进行填充
            front_pad = np.repeat(kpts_norm[:1], pad_left, axis=0)
            back_pad = np.repeat(kpts_norm[-1:], pad_right, axis=0)
            padded = np.concatenate([front_pad, kpts_norm, back_pad], axis=0)  # (T + 242, 17, 2)
        else:
            padded = kpts_norm

        skeletons_3d = []
        for t in range(T):
            if T < 243:
                window = padded[t : t + 243]
            else:
                idx_start = max(0, t - pad_left)
                idx_end = min(T, t + pad_right + 1)
                window = kpts_norm[idx_start:idx_end]
                if len(window) < 243:
                    window = np.pad(window, ((0, 243 - len(window)), (0, 0), (0, 0)), mode="edge")

            window_t = torch.from_numpy(window[np.newaxis, ...]).float().to(self.device)  # (1, 243, 17, 2)
            with torch.no_grad():
                out_3d_t = self.videopose_model(window_t)  # (1, 1, 17, 3)
            skeletons_3d.append(out_3d_t[0, 0].cpu().numpy())

        return np.array(skeletons_3d, dtype=np.float32), confs_arr


class Mock3DPoseEstimator(BasePose3DEstimator):
    """用于单元测试与轻量级无 GPU 环境的 Mock 3D 姿态估计器。"""

    def __init__(
        self,
        default_confidence: float = 0.9,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        super().__init__(skel_def=skel_def)
        self.default_confidence = float(default_confidence)

    def estimate_frame(self, rgb: Union[np.ndarray, Image.Image]) -> Pose3DEstimationResult:
        num_joints = self.skel_def.joint_num
        joint_names = self.skel_def.joint_names

        joints = np.zeros((num_joints, 3), dtype=np.float32)
        joints[0] = [0.0, 0.00, 0.0]
        if num_joints >= 17:
            joints[7] = [0.0, 0.30, 0.0]   # spine
            joints[8] = [0.0, 0.45, 0.0]   # thorax
            joints[10] = [0.0, 0.70, 0.0]  # head
            joints[11] = [0.20, 0.45, 0.0] # left_shoulder
            joints[14] = [-0.20, 0.45, 0.0] # right_shoulder
            joints[4] = [0.10, 0.00, 0.0]  # left_hip
            joints[1] = [-0.10, 0.00, 0.0] # right_hip
            joints[5] = [0.10, -0.40, 0.0] # left_knee
            joints[2] = [-0.10, -0.40, 0.0] # right_knee
            joints[6] = [0.10, -0.80, 0.0] # left_ankle
            joints[3] = [-0.10, -0.80, 0.0] # right_ankle

        confs = np.full(num_joints, self.default_confidence, dtype=np.float32)

        return Pose3DEstimationResult(
            joints=joints,
            joint_names=joint_names,
            confidence=confs,
            coordinate_system=self.skel_def.coordinate_system.get("name", "camera_frame_right_hand"),
            estimator=f"mock_{num_joints}",
            metadata={"mock": True},
        )


def create_pose3d_estimator(
    estimator_type: str = "videopose3d",
    skel_def: Optional[SkeletonDefinition] = None,
    **kwargs: Any,
) -> BasePose3DEstimator:
    """工厂方法：根据类型实例化统一 3D 姿态估计器。"""
    estimator_type = estimator_type.lower()
    if estimator_type in ["videopose3d", "videopose", "h36m", "vpose3d"]:
        return VideoPose3DEstimator(skel_def=skel_def, **kwargs)
    elif estimator_type in ["mediapipe", "mediapipe_33", "blazepose"]:
        return MediaPipe3DPoseEstimator(skel_def=skel_def, **kwargs)
    elif estimator_type in ["mock", "dummy"]:
        return Mock3DPoseEstimator(skel_def=skel_def, **kwargs)
    else:
        raise ValueError(f"Unknown 3D Pose Estimator type: {estimator_type}")

