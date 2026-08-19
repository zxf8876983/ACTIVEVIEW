"""
2D 人体姿态估计后端适配器 —— pose_backend.py
=============================================

功能：
    提供通用 2D 人体姿态检测后端接口与实现，包括基于 TorchVision KeypointRCNN 的实现
    以及离线测试使用的 Mock 后端。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .keypoint_schema import (
    COCO_17_KEYPOINTS,
    Keypoint2D,
    coco17_to_ea_avs_15,
)


@dataclass
class Pose2DDetection:
    """单个人体 2D 检测结果。"""
    keypoints: Dict[str, Keypoint2D]
    bbox_xyxy: Optional[Tuple[float, float, float, float]]
    score: float
    backend_name: str


class PoseBackend(ABC):
    """2D 人体姿态检测器抽象基类。"""

    @abstractmethod
    def infer(self, rgb: np.ndarray) -> List[Pose2DDetection]:
        """对输入的 RGB 图像执行 2D 姿态估计。

        参数：
            rgb: shape=(H, W, 3), dtype=uint8 的 RGB 图像。

        返回：
            Pose2DDetection 列表（按置信度从高到低排序）。
        """
        pass


class TorchvisionPoseBackend(PoseBackend):
    """基于 TorchVision Keypoint R-CNN 的 2D 姿态检测后端。"""

    def __init__(
        self,
        device: str = "cuda:0",
        min_pose_score: float = 0.30,
        min_keypoint_confidence: float = 0.30,
        model_path: Optional[str] = None,
    ):
        import torch
        import torchvision
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn,
            KeypointRCNN_ResNet50_FPN_Weights,
        )

        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.min_pose_score = min_pose_score
        self.min_keypoint_confidence = min_keypoint_confidence
        self.backend_name = "torchvision_keypointrcnn"

        # 加载官方预训练权重
        if model_path is not None:
            self.model = keypointrcnn_resnet50_fpn(weights=None)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = keypointrcnn_resnet50_fpn(weights=weights)

        self.model.to(self.device)
        self.model.eval()

    def infer(self, rgb: np.ndarray) -> List[Pose2DDetection]:
        import torch
        import torchvision.transforms.functional as F

        if rgb is None or rgb.size == 0:
            return []

        # Habitat render 返回的图像可能为 RGBA (H, W, 4)，切片前 3 通道
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        height, width = rgb.shape[:2]
        img_tensor = F.to_tensor(rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        scores = outputs["scores"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        keypoints = outputs["keypoints"].cpu().numpy()  # (N, 17, 3) [x, y, visibility]
        keypoint_scores = outputs["keypoints_scores"].cpu().numpy() if "keypoints_scores" in outputs else None

        detections: List[Pose2DDetection] = []
        for i in range(len(scores)):
            pose_score = float(scores[i])
            if pose_score < self.min_pose_score:
                continue

            box = tuple(map(float, boxes[i]))
            kpts_raw = keypoints[i]  # (17, 3)

            coco_dict: Dict[str, Tuple[float, float, float]] = {}
            for j, name in enumerate(COCO_17_KEYPOINTS):
                u = float(kpts_raw[j, 0])
                v = float(kpts_raw[j, 1])
                # 如果有专门的 keypoint_scores 使用之，否则使用 visibility / 1.0
                if keypoint_scores is not None:
                    conf = float(keypoint_scores[i, j])
                else:
                    conf = float(kpts_raw[j, 2])
                    if conf > 1.0:  # normalize if necessary
                        conf = 1.0 if conf > 0 else 0.0
                coco_dict[name] = (u, v, conf)

            ea_kpts = coco17_to_ea_avs_15(
                coco_dict,
                img_width=width,
                img_height=height,
                min_confidence=self.min_keypoint_confidence,
            )

            detections.append(
                Pose2DDetection(
                    keypoints=ea_kpts,
                    bbox_xyxy=box,
                    score=pose_score,
                    backend_name=self.backend_name,
                )
            )

        # 按置信度排序
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


class MockPoseBackend(PoseBackend):
    """测试与无模型环境专用的 Mock 姿态检测后端。"""

    def __init__(self, preset_detections: Optional[List[Pose2DDetection]] = None):
        self.preset_detections = preset_detections or []
        self.backend_name = "mock_pose_backend"

    def set_preset_detections(self, detections: List[Pose2DDetection]):
        self.preset_detections = detections

    def infer(self, rgb: np.ndarray) -> List[Pose2DDetection]:
        return self.preset_detections


def create_pose_backend(config: dict) -> PoseBackend:
    """根据配置创建 PoseBackend 实例。"""
    p_cfg = config.get("perception", {})
    backend_type = p_cfg.get("pose_backend", "torchvision").lower()

    if backend_type == "mock":
        return MockPoseBackend()
    elif backend_type in ("torchvision", "keypointrcnn"):
        return TorchvisionPoseBackend(
            device=p_cfg.get("device", "cuda:0"),
            min_pose_score=p_cfg.get("min_pose_score", 0.30),
            min_keypoint_confidence=p_cfg.get("min_keypoint_confidence", 0.30),
            model_path=p_cfg.get("model_path"),
        )
    else:
        # 默认回退 torchvision
        return TorchvisionPoseBackend(
            device=p_cfg.get("device", "cuda:0"),
            min_pose_score=p_cfg.get("min_pose_score", 0.30),
            min_keypoint_confidence=p_cfg.get("min_keypoint_confidence", 0.30),
        )
