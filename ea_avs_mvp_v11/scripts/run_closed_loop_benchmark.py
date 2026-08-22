#!/usr/bin/env python3
"""
ACTIVEVIEW v11.4 Closed-Loop Active Perception Benchmark CLI Script
===================================================================

职责：
    1. 跨 10 个场景运行 100 个闭环主动感知仿真 Episode；
    2. 全程记录 Initial Observation -> Navigation -> Post-Observation 完整链路；
    3. 评测对比 4 大策略：Random, Nearest, Fixed Front, Utility Predictor (Ours), Oracle；
    4. 统计并输出 Entropy Reduction, Accuracy Improvement, Efficiency, Nav Distance 指标；
    5. 保存完整数据集与评测结果 JSON。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.closed_loop_episode import ClosedLoopActivePerceptionRunner, ACTION_CLASSES
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_closed_loop_benchmark")


def run_benchmark(
    num_episodes: int = 100,
    num_scenes: int = 10,
    output_dir: str = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """运行多场景闭环主动感知评测。"""
    data_root = get_data_root()
    out_dir = Path(output_dir) if output_dir else (data_root / "v11_closed_loop_dataset")
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing ClosedLoopActivePerceptionRunner across %d scenes...", num_scenes)
    runner = ClosedLoopActivePerceptionRunner(data_root=data_root, seed=seed)

    primary_scenes = runner.scene_mgr.get_primary_scenes(count=num_scenes)
    scene_ids = [s.scene_id for s in primary_scenes]
    logger.info("Selected %d Primary Scenes: %s", len(scene_ids), scene_ids)

    all_episode_records: List[Dict[str, Any]] = []

    # 累加各策略指标
    policies = ["random", "nearest", "fixed_front", "utility_predictor", "oracle"]
    metrics_by_policy = {
        p: {
            "entropy_before": [],
            "entropy_after": [],
            "entropy_reduction": [],
            "confidence_before": [],
            "confidence_after": [],
            "accuracy_before": [],
            "accuracy_after": [],
            "accuracy_improved_count": 0,
            "navigation_distance": [],
            "navigation_steps": [],
            "navigation_efficiency": [],
            "navigation_success": [],
            "oracle_gap": [],
        }
        for p in policies
    }

    logger.info("Starting execution of %d Closed-Loop Episodes...", num_episodes)

    for ep_idx in range(num_episodes):
        scene_id = scene_ids[ep_idx % len(scene_ids)]
        action_id = ep_idx % len(ACTION_CLASSES)
        instance_idx = ep_idx // len(ACTION_CLASSES)

        ep_res = runner.run_single_episode(
            episode_idx=ep_idx,
            scene_id=scene_id,
            action_id=action_id,
            instance_idx=instance_idx,
        )

        ep_dict = ep_res.to_dict()
        all_episode_records.append(ep_dict)

        # 保存单 Episode JSON
        with open(episodes_dir / f"{ep_res.episode_id}.json", "w", encoding="utf-8") as f:
            json.dump(ep_dict, f, indent=2)

        init_obs = ep_res.initial_observation
        h_init = init_obs["entropy"]
        conf_init = init_obs["confidence"]
        is_corr_init = init_obs["is_correct"]

        for p_name in policies:
            p_res = ep_res.policy_results[p_name]
            m = metrics_by_policy[p_name]

            m["entropy_before"].append(h_init)
            m["entropy_after"].append(p_res.entropy_after)
            m["entropy_reduction"].append(p_res.entropy_reduction)
            m["confidence_before"].append(conf_init)
            m["confidence_after"].append(p_res.confidence_after)
            m["accuracy_before"].append(1 if is_corr_init else 0)
            m["accuracy_after"].append(1 if p_res.is_correct_after else 0)
            if p_res.accuracy_improved:
                m["accuracy_improved_count"] += 1
            m["navigation_distance"].append(p_res.navigation_distance)
            m["navigation_steps"].append(p_res.navigation_steps)
            m["navigation_efficiency"].append(p_res.navigation_efficiency)
            m["navigation_success"].append(1 if p_res.navigation_success else 0)
            m["oracle_gap"].append(p_res.oracle_gap)

        if (ep_idx + 1) % 20 == 0 or ep_idx == num_episodes - 1:
            logger.info("Progress: [%03d/%03d] Episodes Finished", ep_idx + 1, num_episodes)

    # 统计并格式化汇总报告
    benchmark_summary = {}
    for p_name in policies:
        m = metrics_by_policy[p_name]
        mean_acc_before = float(np.mean(m["accuracy_before"])) * 100
        mean_acc_after = float(np.mean(m["accuracy_after"])) * 100
        delta_acc = mean_acc_after - mean_acc_before

        benchmark_summary[p_name] = {
            "mean_entropy_before": round(float(np.mean(m["entropy_before"])), 4),
            "mean_entropy_after": round(float(np.mean(m["entropy_after"])), 4),
            "mean_entropy_reduction": round(float(np.mean(m["entropy_reduction"])), 4),
            "mean_confidence_before": round(float(np.mean(m["confidence_before"])), 4),
            "mean_confidence_after": round(float(np.mean(m["confidence_after"])), 4),
            "mean_accuracy_before_pct": round(mean_acc_before, 2),
            "mean_accuracy_after_pct": round(mean_acc_after, 2),
            "accuracy_improvement_pct": round(delta_acc, 2),
            "mean_navigation_distance_m": round(float(np.mean(m["navigation_distance"])), 4),
            "mean_navigation_efficiency": round(float(np.mean(m["navigation_efficiency"])), 4),
            "navigation_success_rate_pct": round(float(np.mean(m["navigation_success"])) * 100, 2),
            "mean_oracle_gap": round(float(np.mean(m["oracle_gap"])), 4),
        }

    logger.info("=========================================================================================")
    logger.info("  ACTIVEVIEW v11.4 CLOSED-LOOP ACTIVE PERCEPTION BENCHMARK RESULTS                       ")
    logger.info("=========================================================================================")
    logger.info("  Total Episodes: %d across %d Scenes", num_episodes, len(scene_ids))
    logger.info("  ---------------------------------------------------------------------------------------")
    logger.info("  Strategy            | H_after | ΔH (Gain) | Conf_after | Acc_after | Nav Dist | Efficiency")
    logger.info("  --------------------+---------+-----------+------------+-----------+----------+-----------")
    for p_name, s in benchmark_summary.items():
        logger.info("  %-19s | %7.4f | %9.4f | %10.4f | %8.2f%% | %7.2fm | %9.4f",
                    p_name, s["mean_entropy_after"], s["mean_entropy_reduction"],
                    s["mean_confidence_after"], s["mean_accuracy_after_pct"],
                    s["mean_navigation_distance_m"], s["mean_navigation_efficiency"])
    logger.info("=========================================================================================")

    results_payload = {
        "num_episodes": num_episodes,
        "num_scenes": len(scene_ids),
        "scenes": scene_ids,
        "benchmark_summary": benchmark_summary,
    }

    # 保存全量数据集元数据与评估结果
    with open(out_dir / "episodes_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_episode_records, f, indent=2)
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Saved closed-loop dataset and evaluation results to: %s", out_dir)
    return results_payload


def main():
    parser = argparse.ArgumentParser(description="Run ACTIVEVIEW v11.4 Closed-Loop Active Perception Benchmark")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of closed-loop episodes")
    parser.add_argument("--num_scenes", type=int, default=10, help="Number of primary scenes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_benchmark(
        num_episodes=args.num_episodes,
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
