"""
Perception pipeline module for v10.0.
RGB-D -> 2D Pose -> Depth Projection -> Estimated 3D Skeleton.
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
from .skeleton_converter import (
    BODY_PART_GROUPS,
    UNIFIED_JOINT_NAMES,
    UNIFIED_SKELETON_PAIRS,
    EstimatedSkeleton3D,
    SkeletonConverter,
)

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
    "UNIFIED_JOINT_NAMES",
    "UNIFIED_SKELETON_PAIRS",
    "BODY_PART_GROUPS",
]
