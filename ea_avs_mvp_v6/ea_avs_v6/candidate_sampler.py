"""
候选视角采样模块 —— candidate_sampler.py
==========================================

功能：
    围绕目标人体位置生成候选观察位姿，并通过导航网格与测地距离进行可达性过滤。
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .geometry import compute_look_at_yaw


@dataclass
class CandidateView:
    """候选观察位姿数据结构。"""
    candidate_id: int
    position: np.ndarray
    yaw: float
    geodesic_distance: float
    euclidean_distance_to_human: float
    is_valid: bool
    invalid_reason: str = ""
    pred_score: dict = field(default_factory=dict)
    true_score: dict = field(default_factory=dict)
    selected_by: list = field(default_factory=list)


class CandidateSampler:
    """候选视角采样器。"""

    def __init__(self, config: dict):
        self.config = config

    def sample(
        self,
        human_pos: np.ndarray,
        robot_pos: np.ndarray,
        runner,
    ) -> List[CandidateView]:
        """围绕人体采样候选位姿，过滤不可达点。

        参数：
            human_pos: 人体世界位置（在估计状态路径中为 estimated_human_position，在 GT 路径中为 gt_human_pos）。
            robot_pos: 机器人起始位置，shape=(3,)。
            runner: HabitatRunner 实例。

        返回：
            CandidateView 列表。
        """
        sampling_cfg = self.config["candidate_sampling"]
        radii = sampling_cfg["radii"]
        angles_deg = sampling_cfg["angles_deg"]
        min_dist = sampling_cfg["min_distance_to_human"]
        max_dist = sampling_cfg["max_distance_to_human"]
        max_geo = sampling_cfg["max_geodesic_distance"]

        candidates: List[CandidateView] = []
        candidate_id = 0
        angles_rad = np.deg2rad(angles_deg)

        for r in radii:
            for theta in angles_rad:
                cx = human_pos[0] + r * np.cos(theta)
                cz = human_pos[2] + r * np.sin(theta)
                cy = human_pos[1]
                candidate = np.array([cx, cy, cz], dtype=np.float32)

                snapped = runner.snap_point(candidate)

                if not runner.is_navigable(snapped):
                    candidates.append(CandidateView(
                        candidate_id=candidate_id, position=snapped, yaw=0.0,
                        geodesic_distance=float("inf"),
                        euclidean_distance_to_human=float("inf"),
                        is_valid=False, invalid_reason="not_navigable",
                    ))
                    candidate_id += 1
                    continue

                euclidean_dist = float(np.linalg.norm(snapped - human_pos))
                geo_dist = runner.geodesic_distance(robot_pos, snapped)

                if euclidean_dist < min_dist or euclidean_dist > max_dist:
                    candidates.append(CandidateView(
                        candidate_id=candidate_id, position=snapped, yaw=0.0,
                        geodesic_distance=geo_dist,
                        euclidean_distance_to_human=euclidean_dist,
                        is_valid=False, invalid_reason="out_of_distance_range",
                    ))
                    candidate_id += 1
                    continue

                if geo_dist == float("inf") or geo_dist > max_geo:
                    reason = ("no_geodesic_path" if geo_dist == float("inf")
                              else "geodesic_distance_too_large")
                    candidates.append(CandidateView(
                        candidate_id=candidate_id, position=snapped, yaw=0.0,
                        geodesic_distance=geo_dist,
                        euclidean_distance_to_human=euclidean_dist,
                        is_valid=False, invalid_reason=reason,
                    ))
                    candidate_id += 1
                    continue

                yaw = compute_look_at_yaw(snapped, human_pos)

                candidates.append(CandidateView(
                    candidate_id=candidate_id, position=snapped, yaw=yaw,
                    geodesic_distance=geo_dist,
                    euclidean_distance_to_human=euclidean_dist,
                    is_valid=True,
                ))
                candidate_id += 1

        return candidates
