"""
EA-AVS-MVP v7.0 观测与数据记录子模块
"""

from .metadata import FrameMetadata, SequenceMetadata
from .recorder import ObservationRecorder

__all__ = [
    "FrameMetadata",
    "SequenceMetadata",
    "ObservationRecorder",
]
