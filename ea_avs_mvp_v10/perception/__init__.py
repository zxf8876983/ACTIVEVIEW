"""
Perception pipeline module for v10.0.
RGB-D -> 2D Pose -> Depth Projection -> Estimated 3D Skeleton (COCO17) -> Normalization.
"""

from .depth_projection import DepthProjectionResult, DepthProjector
from .pose_estimator import (
    COCO_KEYPOINTS,
    COCO_SKELETON_PAIRS,
    BasePoseEstimator,
    MockPoseEstimator,
    Pose2DResult,
    TorchvisionPoseEstimator,
)
from .skeleton_adapter import BaseSkeletonAdapter, COCO17ToNTU25Adapter, NTU_25_JOINT_NAMES
from .skeleton_converter import (
    COCO_BODY_PART_GROUPS,
    EstimatedSkeleton3D,
    SkeletonConverter,
)
from .skeleton_normalizer import SkeletonNormalizer

__all__ = [
    "BasePoseEstimator",
    "TorchvisionPoseEstimator",
    "MockPoseEstimator",
    "Pose2DResult",
    "COCO_KEYPOINTS",
    "COCO_SKELETON_PAIRS",
    "DepthProjector",
    "DepthProjectionResult",
    "EstimatedSkeleton3D",
    "SkeletonConverter",
    "COCO_BODY_PART_GROUPS",
    "SkeletonNormalizer",
    "BaseSkeletonAdapter",
    "COCO17ToNTU25Adapter",
    "NTU_25_JOINT_NAMES",
]
