"""文件用途：组织 RGB、pose 与 skeleton normalization 感知组件。

主要输入：Habitat RGB 帧和感知模型权重。
主要输出：3D pose、骨架定义、归一化结果及 RGB embedding。
项目角色：连接仿真观测与冻结动作识别器的感知层。
"""

from .pose.estimator import BasePose3DEstimator, Pose3DEstimationResult
from .skeleton import JointDef, SkeletonDefinition, get_skeleton_definition
from .normalization import SkeletonNormalizer
from .pose.ultralytics import UltralyticsPose3DEstimator

__all__ = [
    "BasePose3DEstimator",
    "Pose3DEstimationResult",
    "JointDef",
    "SkeletonDefinition",
    "get_skeleton_definition",
    "SkeletonNormalizer",
    "UltralyticsPose3DEstimator",
]
