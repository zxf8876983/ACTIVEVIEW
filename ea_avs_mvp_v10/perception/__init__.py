"""
Perception pipeline module for v10.0.
RGB-D Observation -> RGB-D Skeleton Extractor -> 3D Skeleton -> Normalizer -> Validator -> ST-GCN ready.
"""

from .coordinate_validator import CoordinateValidator, ValidationResult
from .depth_projection import DepthProjectionResult, DepthProjector  # Retained for legacy/deprecated support
from .pose_estimator import (
    COCO_KEYPOINTS,
    COCO_SKELETON_PAIRS,
    BasePoseEstimator,
    MockPoseEstimator,
    Pose2DResult,
    TorchvisionPoseEstimator,
)
from .rgbd_skeleton_extractor import (
    BaseRGBDSkeletonExtractor,
    MediaPipeRGBDSkeletonExtractor,
    MockRGBDSkeletonExtractor,
    RGBDSkeletonExtractor,
)
from .skeleton_adapter import (
    BaseSkeletonAdapter,
    COCO17ToNTU25Adapter,
    MediaPipe33ToCOCO17Adapter,
    MediaPipe33ToNTU25Adapter,
    NTU_25_JOINT_NAMES,
)
from .skeleton_converter import (
    COCO_BODY_PART_GROUPS,
    EstimatedSkeleton3D,
    SkeletonConverter,
)
from .skeleton_definition import (
    JointDef,
    SkeletonDefinition,
    get_skeleton_definition,
)
from .skeleton_normalizer import SkeletonNormalizer

__all__ = [
    # Skeleton Definition
    "SkeletonDefinition",
    "JointDef",
    "get_skeleton_definition",
    # RGB-D Extractor
    "BaseRGBDSkeletonExtractor",
    "MediaPipeRGBDSkeletonExtractor",
    "MockRGBDSkeletonExtractor",
    "RGBDSkeletonExtractor",
    # Core Structures & Adapters
    "EstimatedSkeleton3D",
    "SkeletonConverter",
    "SkeletonNormalizer",
    "BaseSkeletonAdapter",
    "MediaPipe33ToCOCO17Adapter",
    "MediaPipe33ToNTU25Adapter",
    "COCO17ToNTU25Adapter",
    "NTU_25_JOINT_NAMES",
    # Validation
    "CoordinateValidator",
    "ValidationResult",
    # Legacy / Deprecated
    "BasePoseEstimator",
    "TorchvisionPoseEstimator",
    "MockPoseEstimator",
    "Pose2DResult",
    "DepthProjector",
    "DepthProjectionResult",
    "COCO_KEYPOINTS",
    "COCO_SKELETON_PAIRS",
    "COCO_BODY_PART_GROUPS",
]
