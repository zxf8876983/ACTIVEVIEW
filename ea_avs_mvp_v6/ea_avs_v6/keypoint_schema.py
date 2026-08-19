"""
关键点 Schema 映射与定义模块 —— keypoint_schema.py
=================================================

功能：
    统一定义 COCO-17 与 EA-AVS 15 关键点格式，并提供两者的双向/转换映射。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# COCO-17 标准关键点名称列表（顺序与主流检测模型一致）
COCO_17_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# EA-AVS 15 关键点名称元组（与 v5.0 完全一致）
EA_AVS_15_KEYPOINTS = (
    "head",
    "neck",
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# 关键点所属部位分组
KEYPOINT_GROUPS = {
    "torso": ["neck", "pelvis", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "lower_body": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
    "head": ["head"],
    "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
}


@dataclass
class Keypoint2D:
    name: str
    u: float
    v: float
    confidence: float
    detected: bool
    source: str  # "observed_2d" | "derived_2d" | "missing"


def coco17_to_ea_avs_15(
    coco_kpts: Dict[str, Tuple[float, float, float]],
    img_width: int,
    img_height: int,
    min_confidence: float = 0.30,
) -> Dict[str, Keypoint2D]:
    """将 COCO-17 关键点字典转换为 EA-AVS 15-keypoint 字典。

    参数：
        coco_kpts: 字典 {name: (u, v, conf)}
        img_width: 图像宽度
        img_height: 图像高度
        min_confidence: 关键点最小有效置信度

    返回：
        字典 {ea_avs_name: Keypoint2D}
    """
    res: Dict[str, Keypoint2D] = {}

    def is_inside(u: float, v: float) -> bool:
        return 0.0 <= u < float(img_width) and 0.0 <= v < float(img_height)

    # 1. 基础直接映射肢体关节
    direct_mappings = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    for kpt_name in direct_mappings:
        if kpt_name in coco_kpts:
            u, v, conf = coco_kpts[kpt_name]
            detected = (conf >= min_confidence) and is_inside(u, v)
            res[kpt_name] = Keypoint2D(
                name=kpt_name,
                u=float(u),
                v=float(v),
                confidence=float(conf),
                detected=detected,
                source="observed_2d" if detected else "missing",
            )
        else:
            res[kpt_name] = Keypoint2D(
                name=kpt_name, u=0.0, v=0.0, confidence=0.0, detected=False, source="missing"
            )

    # 2. head：优先使用 nose，若 nose 置信度不足，尝试使用 eyes/ears 的中心
    head_u, head_v, head_conf, head_src = 0.0, 0.0, 0.0, "missing"
    if "nose" in coco_kpts and coco_kpts["nose"][2] >= min_confidence and is_inside(coco_kpts["nose"][0], coco_kpts["nose"][1]):
        u, v, conf = coco_kpts["nose"]
        head_u, head_v, head_conf, head_src = float(u), float(v), float(conf), "observed_2d"
    else:
        # fallback to facial features mean
        face_pts = [coco_kpts[k] for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"] if k in coco_kpts and coco_kpts[k][2] >= min_confidence]
        if face_pts:
            head_u = float(np.mean([p[0] for p in face_pts]))
            head_v = float(np.mean([p[1] for p in face_pts]))
            head_conf = float(np.mean([p[2] for p in face_pts]))
            head_src = "derived_2d"

    head_detected = (head_conf >= min_confidence) and is_inside(head_u, head_v)
    res["head"] = Keypoint2D(
        name="head",
        u=head_u,
        v=head_v,
        confidence=head_conf,
        detected=head_detected,
        source=head_src if head_detected else "missing",
    )

    # 3. neck：midpoint(left_shoulder, right_shoulder)
    l_sh = res.get("left_shoulder")
    r_sh = res.get("right_shoulder")
    if l_sh and r_sh and l_sh.detected and r_sh.detected:
        neck_u = (l_sh.u + r_sh.u) / 2.0
        neck_v = (l_sh.v + r_sh.v) / 2.0
        neck_conf = min(l_sh.confidence, r_sh.confidence)
        neck_det = is_inside(neck_u, neck_v)
        res["neck"] = Keypoint2D(
            name="neck",
            u=neck_u,
            v=neck_v,
            confidence=neck_conf,
            detected=neck_det,
            source="derived_2d" if neck_det else "missing",
        )
    else:
        res["neck"] = Keypoint2D(
            name="neck", u=0.0, v=0.0, confidence=0.0, detected=False, source="missing"
        )

    # 4. pelvis：midpoint(left_hip, right_hip)
    l_hip = res.get("left_hip")
    r_hip = res.get("right_hip")
    if l_hip and r_hip and l_hip.detected and r_hip.detected:
        pelvis_u = (l_hip.u + r_hip.u) / 2.0
        pelvis_v = (l_hip.v + r_hip.v) / 2.0
        pelvis_conf = min(l_hip.confidence, r_hip.confidence)
        pelvis_det = is_inside(pelvis_u, pelvis_v)
        res["pelvis"] = Keypoint2D(
            name="pelvis",
            u=pelvis_u,
            v=pelvis_v,
            confidence=pelvis_conf,
            detected=pelvis_det,
            source="derived_2d" if pelvis_det else "missing",
        )
    else:
        res["pelvis"] = Keypoint2D(
            name="pelvis", u=0.0, v=0.0, confidence=0.0, detected=False, source="missing"
        )

    return res
