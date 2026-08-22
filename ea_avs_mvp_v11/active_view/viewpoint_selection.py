"""
主动视点选择策略与多基准评测器 —— viewpoint_selection.py (v11.3)
============================================================

职责：
    1. 实现基于 Utility Predictor 效用排序的最优视点选择决策器；
    2. 实现 3 大经典对比基线策略：
       - Random View Selection (随机视角选择)
       - Nearest View Selection (最近距离视角选择 / 最小导航代价)
       - Oracle View Selection (全知最优离线上界)
    3. 提供多基准对比评测函数，计算选择后动作识别熵、熵降低量、Oracle Gap 与 Top-1 命中率。
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("viewpoint_selection")


class ActiveViewSelector:
    """主动视角选择器。"""

    def __init__(
        self,
        predictor: Optional[ViewpointUtilityPredictor] = None,
        strategy: str = "utility_predictor",
    ):
        self.predictor = predictor
        self.strategy = strategy

    def select_viewpoint(
        self,
        current_observation: Dict[str, Any],
        candidate_viewpoints: List[Viewpoint],
        strategy: Optional[str] = None,
    ) -> Tuple[Viewpoint, float]:
        """
        根据指定策略从可行候选点中选出最佳观察视角。

        返回:
            (selected_viewpoint, predicted_or_heuristic_score)
        """
        if not candidate_viewpoints:
            raise ValueError("Candidate viewpoints list is empty.")

        strat = strategy or self.strategy

        if strat == "random":
            chosen = random.choice(candidate_viewpoints)
            return chosen, 0.0

        elif strat == "nearest":
            # 选取导航移动代价最小的视点
            chosen = min(candidate_viewpoints, key=lambda v: v.navigation_cost)
            return chosen, -chosen.navigation_cost

        elif strat == "utility_predictor":
            if self.predictor is None:
                raise RuntimeError("ViewpointUtilityPredictor is not initialized for strategy='utility_predictor'.")
            scores = self.predictor.predict_utility(current_observation, candidate_viewpoints)
            best_idx = int(np.argmax(scores))
            return candidate_viewpoints[best_idx], scores[best_idx]

        else:
            raise ValueError(f"Unknown viewpoint selection strategy: {strat}")


def evaluate_viewpoint_selection_benchmarks(
    test_samples: List[Dict[str, Any]],
    predictor: ViewpointUtilityPredictor,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    在测试集上全面评测 4 大主动视角选择策略:
        1. Random
        2. Nearest
        3. Utility Predictor (Ours)
        4. Oracle (Ground Truth Upper Bound)
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按照 motion_instance_id 聚合每个动作实例的候选视角集合
    instances: Dict[str, List[Dict[str, Any]]] = {}
    for s in test_samples:
        m_id = s.get("motion_instance_id", s.get("motion_id", ""))
        instances.setdefault(m_id, []).append(s)

    logger.info("Evaluating viewpoint selection across %d test motion instances (%d total samples)...",
                len(instances), len(test_samples))

    policies = ["random", "nearest", "utility_predictor", "oracle"]
    results: Dict[str, Dict[str, List[float]]] = {
        p: {
            "selected_entropies": [],
            "entropy_reductions": [],
            "oracle_gaps": [],
            "top1_matches": [],
            "navigation_costs": [],
        }
        for p in policies
    }

    for m_id, sample_list in instances.items():
        if not sample_list:
            continue

        h_curr = float(sample_list[0].get("current_entropy", 0.40))
        curr_obs = {"current_viewpoint": sample_list[0]["current_viewpoint"]}

        # 构造 Viewpoint 对象列表与对应的真实后验熵
        vps: List[Viewpoint] = []
        true_entropies: List[float] = []

        for s in sample_list:
            vp_dict = s["candidate_viewpoint"] if "candidate_viewpoint" in s else s["viewpoint"]
            vps.append(Viewpoint(
                id=int(vp_dict["id"]),
                position=list(vp_dict["position"]),
                rotation=list(vp_dict["rotation"]),
                yaw=float(vp_dict["yaw"]),
                pitch=float(vp_dict["pitch"]),
                distance=float(vp_dict["distance"]),
                angle=float(vp_dict["angle"]),
                camera_height=float(vp_dict["camera_height"]),
                navigation_cost=float(vp_dict.get("navigation_cost", 0.0)),
            ))
            true_entropies.append(float(s.get("candidate_entropy", s.get("entropy", 0.0))))

        # 1. Oracle: 真实后验熵最小的视角
        oracle_idx = int(np.argmin(true_entropies))
        h_oracle = true_entropies[oracle_idx]
        oracle_vp = vps[oracle_idx]

        # 2. Random
        rand_idx = random.randint(0, len(vps) - 1)
        # 3. Nearest
        nearest_idx = int(np.argmin([v.navigation_cost for v in vps]))
        # 4. Ours (Utility Predictor)
        pred_scores = predictor.predict_utility(curr_obs, vps)
        ours_idx = int(np.argmax(pred_scores))

        policy_indices = {
            "random": rand_idx,
            "nearest": nearest_idx,
            "utility_predictor": ours_idx,
            "oracle": oracle_idx,
        }

        for p_name, idx in policy_indices.items():
            sel_h = true_entropies[idx]
            sel_vp = vps[idx]

            results[p_name]["selected_entropies"].append(sel_h)
            results[p_name]["entropy_reductions"].append(h_curr - sel_h)
            results[p_name]["oracle_gaps"].append(sel_h - h_oracle)
            results[p_name]["top1_matches"].append(1.0 if idx == oracle_idx else 0.0)
            results[p_name]["navigation_costs"].append(sel_vp.navigation_cost)

    # 汇总各策略平均指标
    summary: Dict[str, Dict[str, float]] = {}
    for p_name in policies:
        summary[p_name] = {
            "mean_selected_entropy": round(float(np.mean(results[p_name]["selected_entropies"])), 4),
            "mean_entropy_reduction": round(float(np.mean(results[p_name]["entropy_reductions"])), 4),
            "mean_oracle_gap": round(float(np.mean(results[p_name]["oracle_gaps"])), 4),
            "top1_accuracy": round(float(np.mean(results[p_name]["top1_matches"])), 4),
            "mean_navigation_cost": round(float(np.mean(results[p_name]["navigation_costs"])), 4),
        }

    return {
        "num_test_instances": len(instances),
        "policy_summary": summary,
        "raw_results": results,
    }
