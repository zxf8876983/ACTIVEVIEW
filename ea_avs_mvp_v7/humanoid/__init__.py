"""
EA-AVS-MVP v7.0 Humanoid 实体与代理子模块
"""

from .humanoid_loader import (
    HumanoidAssetBundle,
    resolve_humanoid_assets,
    validate_humanoid_assets,
)
from .humanoid_agent import HumanoidAgent

__all__ = [
    "HumanoidAssetBundle",
    "resolve_humanoid_assets",
    "validate_humanoid_assets",
    "HumanoidAgent",
]
