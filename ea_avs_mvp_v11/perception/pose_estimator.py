"""
2D 人体姿态估计器封装 —— pose_estimator.py
========================================

职责：
    1. 接收 RGB 图像 (uint8, HxWx3)；
    2. 基于开箱即用轻量姿态估计器 (Torchvision Keypoint R-CNN) 提取 17 个 COCO 关键点 2D 坐标与置信度；
    3. 支持 MockPoseEstimator 供纯 Python 单元测试无网络/无 GPU 运行；
    4. 严格禁止在此处读取或反向依赖任何 Habitat GT 人体姿态真值。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# COCO-17 关键点标准名称定义
COCO_KEYPOINTS: List[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# 骨架连线定义 (COCO Kinematic Tree for visualization)
COCO_SKELETON_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 头部/五官
    (5, 6),                                     # 双肩
    (5, 7), (7, 9),                             # 左臂
    (6, 8), (8, 10),                            # 右臂
    (5, 11), (6, 12),                           # 躯干侧边
    (11, 12),                                   # 骨盆跨部
    (11, 13), (13, 15),                         # 左腿
    (12, 14), (14, 16),                         # 右腿
]


@dataclass
class Pose2DResult:
    """2D 姿态估计结果容器。"""
    keypoints: np.ndarray          # 形状 (17, 2), 每行 [u, v] (像素坐标)
    confidence: np.ndarray         # 形状 (17,), 每个关节置信度 [0.0, 1.0]
    bbox: Optional[np.ndarray] = None      # [x1, y1, x2, y2]
    person_score: float = 1.0              # 人体检测整体置信度
    joint_names: List[str] = field(default_factory=lambda: list(COCO_KEYPOINTS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keypoints": self.keypoints.tolist(),
            "confidence": self.confidence.tolist(),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "person_score": float(self.person_score),
            "joint_names": self.joint_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pose2DResult":
        return cls(
            keypoints=np.array(data["keypoints"], dtype=np.float32),
            confidence=np.array(data["confidence"], dtype=np.float32),
            bbox=np.array(data["bbox"], dtype=np.float32) if data.get("bbox") is not None else None,
            person_score=float(data.get("person_score", 1.0)),
            joint_names=data.get("joint_names", list(COCO_KEYPOINTS)),
        )


class BasePoseEstimator(ABC):
    """2D 姿态估计器抽象基类。"""

    @abstractmethod
    def estimate_pose2d(self, rgb_image: Union[np.ndarray, Image.Image]) -> Pose2DResult:
        """从 RGB 图像估计 2D 人体关键点与置信度。"""
        pass


class TorchvisionPoseEstimator(BasePoseEstimator):
    """基于 Torchvision Pretrained Keypoint R-CNN 的开箱即用 2D 姿态估计器。"""

    def __init__(self, device: Optional[str] = None, min_score_thresh: float = 0.5):
        self.min_score_thresh = float(min_score_thresh)
        self._model = None
        self._device_str = device
        self._device = None
        self._transforms = None

    def _lazy_init(self):
        if self._model is not None:
            return

        import torch
        import torchvision
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn,
            KeypointRCNN_ResNet50_FPN_Weights,
        )

        if self._device_str is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)

        logger.info("Initializing Keypoint R-CNN on device: %s", self._device)
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self._transforms = weights.transforms()
        self._model = keypointrcnn_resnet50_fpn(weights=weights).to(self._device)
        self._model.eval()

    def estimate_pose2d(self, rgb_image: Union[np.ndarray, Image.Image]) -> Pose2DResult:
        self._lazy_init()
        import torch

        if isinstance(rgb_image, np.ndarray):
            pil_img = Image.fromarray(rgb_image.astype(np.uint8))
        else:
            pil_img = rgb_image

        w, h = pil_img.size
        img_tensor = self._transforms(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(img_tensor)[0]

        scores = outputs["scores"].cpu().numpy()
        if len(scores) == 0 or scores[0] < self.min_score_thresh:
            logger.debug("No high-confidence person detected (top score: %s)", scores[0] if len(scores) > 0 else "None")
            return Pose2DResult(
                keypoints=np.zeros((17, 2), dtype=np.float32),
                confidence=np.zeros(17, dtype=np.float32),
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                person_score=0.0,
            )

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_box = outputs["boxes"][best_idx].cpu().numpy()

        kpts_raw = outputs["keypoints"][best_idx].cpu().numpy()  # (17, 3): [x, y, visibility]
        kpt_scores = outputs["keypoints_scores"][best_idx].cpu().numpy()  # (17,)

        # 归一化关键点置信度
        kpts_2d = kpts_raw[:, :2].astype(np.float32)

        # keypoint_scores 可为负或 logits，使用 sigmoid 或截断归一化至 [0, 1]
        conf = 1.0 / (1.0 + np.exp(-np.clip(kpt_scores, -10.0, 10.0)))

        return Pose2DResult(
            keypoints=kpts_2d,
            confidence=conf.astype(np.float32),
            bbox=best_box.astype(np.float32),
            person_score=best_score,
        )


class MockPoseEstimator(BasePoseEstimator):
    """用于单元测试与离线模拟的 Mock 2D 姿态估计器。"""

    def __init__(self, default_confidence: float = 0.95):
        self.default_confidence = default_confidence

    def estimate_pose2d(self, rgb_image: Union[np.ndarray, Image.Image]) -> Pose2DResult:
        if isinstance(rgb_image, np.ndarray):
            h, w = rgb_image.shape[:2]
        else:
            w, h = rgb_image.size

        # 模拟生成位于画面中央的标准 17 关键点
        cx, cy = w / 2.0, h / 2.0
        kpts = np.array([
            [cx, cy - 120],       # 0: nose
            [cx - 10, cy - 130],  # 1: left_eye
            [cx + 10, cy - 130],  # 2: right_eye
            [cx - 25, cy - 125],  # 3: left_ear
            [cx + 25, cy - 125],  # 4: right_ear
            [cx - 40, cy - 80],   # 5: left_shoulder
            [cx + 40, cy - 80],   # 6: right_shoulder
            [cx - 60, cy - 20],   # 7: left_elbow
            [cx + 60, cy - 20],   # 8: right_elbow
            [cx - 70, cy + 30],   # 9: left_wrist
            [cx + 70, cy + 30],   # 10: right_wrist
            [cx - 30, cy + 20],   # 11: left_hip
            [cx + 30, cy + 20],   # 12: right_hip
            [cx - 35, cy + 90],   # 13: left_knee
            [cx + 35, cy + 90],   # 14: right_knee
            [cx - 40, cy + 160],  # 15: left_ankle
            [cx + 40, cy + 160],  # 16: right_ankle
        ], dtype=np.float32)

        conf = np.full(17, self.default_confidence, dtype=np.float32)
        bbox = np.array([cx - 80, cy - 140, cx + 80, cy + 180], dtype=np.float32)

        return Pose2DResult(
            keypoints=kpts,
            confidence=conf,
            bbox=bbox,
            person_score=1.0,
        )
