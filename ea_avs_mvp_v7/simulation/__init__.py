"""
EA-AVS-MVP v7.0 仿真场景与机器人观测生成子模块
"""

from .scene_loader import HabitatSceneLoader
from .robot_sensor import RobotSensorRig
from .observation_generator import ObservationGenerator, ObservationRecord

__all__ = [
    "HabitatSceneLoader",
    "RobotSensorRig",
    "ObservationGenerator",
    "ObservationRecord",
]
