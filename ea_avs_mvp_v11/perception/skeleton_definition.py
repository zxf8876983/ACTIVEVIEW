"""
统一骨架定义与关节拓扑规范 —— skeleton_definition.py
===================================================

职责：
    1. 作为 ACTIVEVIEW v10.0 全局唯一的人体骨架拓扑、关节定义与坐标系规范标准；
    2. 加载并维护 `configs/skeleton_definition.json`；
    3. 提供所有关节索引、名称、父子关系与官方骨骼连接边；
    4. 提供 GT 关节与估计关节之间的动态映射标准；
    5. 禁止任何业务代码硬编码关节索引。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class JointDef:
    id: int
    name: str
    part: str
    parent: Optional[int] = None


@dataclass
class SkeletonDefinition:
    backend: str
    joint_num: int
    root_joints: List[int]
    root_name: str
    torso_joints: List[int]
    coordinate_system: Dict[str, str]
    joints: List[JointDef]
    edges: List[Tuple[int, int]]
    bone_symmetry_pairs: List[Dict[str, Any]] = field(default_factory=list)
    gt_joint_mapping: Dict[str, Any] = field(default_factory=dict)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {j.name: j.id for j in self.joints}

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {j.id: j.name for j in self.joints}

    @property
    def part_groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for j in self.joints:
            groups.setdefault(j.part, []).append(j.id)
        return groups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "joint_num": self.joint_num,
            "root_joints": self.root_joints,
            "root_name": self.root_name,
            "torso_joints": self.torso_joints,
            "coordinate_system": self.coordinate_system,
            "joints": [{"id": j.id, "name": j.name, "part": j.part, "parent": j.parent} for j in self.joints],
            "edges": self.edges,
            "bone_symmetry_pairs": self.bone_symmetry_pairs,
            "gt_joint_mapping": self.gt_joint_mapping,
        }


# 全局单例定义缓存字典
_SKELETON_DEF_CACHE: Dict[str, SkeletonDefinition] = {}


def get_skeleton_definition(
    config_path: Optional[Path] = None,
    backend: str = "h36m_17",
) -> SkeletonDefinition:
    """获取指定拓扑后端的骨架定义对象 (默认为 h36m_17 17 关节标准定义)。"""
    global _SKELETON_DEF_CACHE

    cache_key = str(config_path) if config_path is not None else backend
    if cache_key in _SKELETON_DEF_CACHE:
        return _SKELETON_DEF_CACHE[cache_key]

    # 优先从文件加载
    if config_path is None:
        target_file = "skeleton_definition_h36m.json" if backend.startswith("h36m") else "skeleton_definition.json"
        possible_paths = [
            Path(__file__).resolve().parent.parent.parent / "configs" / target_file,
            Path(__file__).resolve().parent.parent / "configs" / target_file,
            Path("configs") / target_file,
            Path(__file__).resolve().parent.parent.parent / "configs" / "skeleton_definition.json",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        joints = [JointDef(id=j["id"], name=j["name"], part=j["part"], parent=j.get("parent")) for j in data["joints"]]
        edges = [tuple(e) for e in data["edges"]]
        skel_def = SkeletonDefinition(
            backend=data.get("backend", backend),
            joint_num=data.get("joint_num", len(joints)),
            root_joints=data.get("root_joints", [0] if backend.startswith("h36m") else [23, 24]),
            root_name=data.get("root_name", "pelvis" if backend.startswith("h36m") else "hip_center"),
            torso_joints=data.get("torso_joints", [0, 7, 8, 11, 14] if backend.startswith("h36m") else [11, 12, 23, 24]),
            coordinate_system=data.get("coordinate_system", {
                "name": "camera_frame_right_hand",
                "x_axis": "right (+X)",
                "y_axis": "up (+Y)",
                "z_axis": "forward/depth (+Z)",
                "unit": "meter",
            }),
            joints=joints,
            edges=edges,
            bone_symmetry_pairs=data.get("bone_symmetry_pairs", []),
            gt_joint_mapping=data.get("gt_joint_mapping", {}),
        )
        _SKELETON_DEF_CACHE[cache_key] = skel_def
        return skel_def

    raise FileNotFoundError(f"Skeleton definition config not found for backend {backend} / path {config_path}")

