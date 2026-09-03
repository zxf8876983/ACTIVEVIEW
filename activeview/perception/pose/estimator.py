"""文件用途：
    提供 RGB 姿态估计接口与实现。

主要输入：
    - RGB 帧序列和模型权重。
主要输出：
    - 3D pose estimation 结果。
项目角色：
    - 属于 perception.pose 感知模块。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class Pose3DEstimationResult:
    """Single-frame 3D pose and detector confidence."""

    joints: np.ndarray
    joint_names: List[str]
    confidence: np.ndarray
    coordinate_system: str
    estimator: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joints": self.joints.tolist(),
            "joint_names": self.joint_names,
            "confidence": [round(float(value), 4) for value in self.confidence],
            "coordinate_system": self.coordinate_system,
            "estimator": self.estimator,
            "metadata": self.metadata,
        }


class BasePose3DEstimator(ABC):
    """Interface shared by RGB pose detectors followed by VideoPose3D."""

    def __init__(self, skel_def: Any) -> None:
        self.skel_def = skel_def

    @abstractmethod
    def estimate_frame(self, rgb: Union[np.ndarray, Image.Image]) -> Pose3DEstimationResult:
        """Estimate a single RGB frame."""

    def estimate_sequence(
        self, rgb_frames: Sequence[Union[np.ndarray, Image.Image]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate a sequence, returning ``(T,V,3)`` and ``(T,V)`` arrays."""
        results = [self.estimate_frame(frame) for frame in rgb_frames]
        if not results:
            return (
                np.zeros((0, self.skel_def.joint_num, 3), dtype=np.float32),
                np.zeros((0, self.skel_def.joint_num), dtype=np.float32),
            )
        return (
            np.asarray([result.joints for result in results], dtype=np.float32),
            np.asarray([result.confidence for result in results], dtype=np.float32),
        )
