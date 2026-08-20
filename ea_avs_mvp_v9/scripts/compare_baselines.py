"""
多动作类别批量基线评测脚本 —— compare_baselines.py
=================================================

功能：
    遍历支持的全部 5 类核心动作 (fall, sitting, standing, bending, reaching)，
    在相同场景与候选视点池下评测视角偏移 (Viewpoint Preference Shift) 与增益量化。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v9.action.action_encoder import ALL_ACTION_CLASSES, ActionEncoder
from ea_avs_mvp_v9.core.config import load_v9_config
from ea_avs_mvp_v9.core.paths import get_data_root
from ea_avs_mvp_v9.evaluation.baseline_comparison import compare_all_baselines
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer
from ea_avs_mvp_v9.visualization.action_view_plotter import format_comparison_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_baselines")


def main():
    parser = argparse.ArgumentParser(description="Multi-action active viewpoint comparison")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_v9_config(args.config)
    vis_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "visualizations/v9_multi_action_comparison"
    vis_dir.mkdir(parents=True, exist_ok=True)

    human_pos = cfg.human.get("initial_pose", {}).get("position", [1.5, -1.60, 4.0])

    # 1. 生成视点池
    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pos,
        human_yaw_deg=0.0,
        ground_height=-1.60,
    )

    # 模拟 16 关节位置
    mock_joints = {
        "pelvis": [1.5, -0.80, 4.0],
        "head": [1.5, -0.05, 4.0],
        "left_shoulder": [1.3, -0.30, 4.0],
        "right_shoulder": [1.7, -0.30, 4.0],
        "left_elbow": [1.2, -0.55, 4.0],
        "right_elbow": [1.8, -0.55, 4.0],
        "left_wrist": [1.1, -0.80, 4.0],
        "right_wrist": [1.9, -0.80, 4.0],
        "spine": [1.5, -0.50, 4.0],
        "left_hip": [1.4, -0.90, 4.0],
        "right_hip": [1.6, -0.90, 4.0],
        "left_knee": [1.4, -1.25, 4.0],
        "right_knee": [1.6, -1.25, 4.0],
        "left_ankle": [1.4, -1.55, 4.0],
        "right_ankle": [1.6, -1.55, 4.0],
    }

    # 2. 基础几何打分
    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    ranked_pairs = geom_evaluator.rank_viewpoints(raw_candidates, mock_joints, human_yaw_deg=0.0)
    geom_qualities = [q for _, q in ranked_pairs]
    geom_map = {q.viewpoint_id: q.visibility_score for q in geom_qualities}

    # 3. 视点特征提取
    feat_extractor = ViewFeatureExtractor(cfg.camera)
    features = feat_extractor.extract_batch(raw_candidates, mock_joints, human_yaw_deg=0.0)

    # 4. 遍历所有动作对比
    encoder = ActionEncoder(cfg.action_weights)
    scorer = ActionConditionedScorer(cfg.scoring)

    batch_summary = {}
    for act_class in ALL_ACTION_CLASSES:
        act_embed = encoder.encode(act_class)
        scores = scorer.score_batch(features, act_embed, geom_map)
        comp_rep = compare_all_baselines(
            viewpoints=raw_candidates,
            action_scores=scores,
            features=features,
            geometry_qualities=geom_qualities,
            action=act_embed,
            human_position=human_pos,
        )
        batch_summary[act_class.value] = comp_rep
        print("\n" + format_comparison_table(comp_rep))

    summary_file = vis_dir / "all_actions_comparison.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll action comparisons saved to: {summary_file}")


if __name__ == "__main__":
    main()
