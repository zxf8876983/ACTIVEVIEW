#!/usr/bin/env python
"""
EA-AVS-MVP v3.0 主入口 —— run_mvp30_action_aware.py
=====================================================

中文名称：面向老人动作感知的移动机器人主动视角选择 v3.0 —— 动作感知导向版本

v3.0 核心改进（相比 v2.0）：
    1. 多姿态骨架：standing、sitting、lying_fallen、bending
    2. 人体朝向建模：human_yaw，骨架按朝向旋转
    3. 动作关键部位评分：不同姿态使用不同身体部位权重
    4. 朝向感知评分：偏好侧前方 45° 观察角度

运行命令：
    python scripts/run_mvp30_action_aware.py \\
        --config configs/mvp30_action_aware.yaml \\
        --episodes 20 \\
        --output-dir outputs/mvp30_test_run

主流程：
    1.  采样人体位置、姿态类型、人体朝向
    2.  生成带有朝向的世界坐标骨架
    3.  采样机器人起始位置
    4.  构造 current_view，计算 pred_score（含动作部位 + 朝向评分）
    5.  采样候选位姿，计算 pred_score
    6.  四种策略根据 Q_pred 选择位姿
    7.  渲染 current_view 和各策略选中位姿，计算 true_score
    8.  写入 metrics.csv 和 episodes.jsonl
"""

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v3.config import load_config
from ea_avs_v3.habitat_runner import HabitatRunner
from ea_avs_v3.skeleton import get_skeleton
from ea_avs_v3.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v3.predictive_evaluator import PredictiveEvaluator
from ea_avs_v3.true_evaluator import TrueEvaluator
from ea_avs_v3.policies import FixedPolicy, RandomPolicy, NearestPolicy, OursPolicy
from ea_avs_v3.metrics import MetricsWriter
from ea_avs_v3.visualization import save_rgb_image, save_candidate_debug_json
from ea_avs_v3.geometry import compute_look_at_yaw
from ea_avs_v3.orientation import compute_relative_view_angle


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EA-AVS-MVP v3.0: 动作感知导向主动视角选择"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖配置文件中的 episode 数量")
    parser.add_argument("--output-dir", type=str, default="outputs/mvp30_test_run",
                        help="输出目录")
    return parser.parse_args()


# =============================================================================
# 辅助函数
# =============================================================================

def set_random_seed(seed: int):
    np.random.seed(seed)
    import random
    random.seed(seed)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def sample_valid_human_position(runner: HabitatRunner, config: dict) -> np.ndarray:
    """从导航网格采样人体位置。"""
    max_tries = config["episode"]["max_sampling_tries"]
    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if runner.is_navigable(pt):
            return pt
    raise RuntimeError(f"无法采样人体位置（{max_tries} 次尝试后）")


def sample_pose_type(config: dict) -> str:
    """根据配置随机采样一个姿态类型。

    如果 randomize_pose_type=True，从 pose_types 列表中随机选择。
    否则使用 default_pose_type。
    """
    human_cfg = config["human"]
    if config["episode"].get("randomize_pose_type", True):
        return str(np.random.choice(human_cfg["pose_types"]))
    return human_cfg["default_pose_type"]


def sample_human_yaw(config: dict) -> float:
    """根据配置采样人体朝向角。

    如果 randomize_human_yaw=True，从 yaw_candidates_deg 中随机选择。
    否则使用 0（朝向 +Z）。
    """
    human_cfg = config["human"]
    if config["episode"].get("randomize_human_yaw", True):
        yaw_deg = float(np.random.choice(human_cfg["yaw_candidates_deg"]))
        return np.deg2rad(yaw_deg)
    return 0.0


def sample_robot_start_position_around_human(
    runner: HabitatRunner, human_pos: np.ndarray, config: dict,
) -> np.ndarray:
    """围绕人体采样机器人起始位置。"""
    ep_cfg = config["episode"]
    min_dist = ep_cfg["min_robot_human_distance"]
    max_dist = ep_cfg["max_robot_human_distance"]
    max_tries = ep_cfg["max_sampling_tries"]

    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if not runner.is_navigable(pt):
            continue
        dist = euclidean_distance(pt, human_pos)
        if dist < min_dist or dist > max_dist:
            continue
        if runner.geodesic_distance(pt, human_pos) == float("inf"):
            continue
        return pt
    raise RuntimeError(f"无法采样机器人起始位置（{max_tries} 次尝试后）")


def compute_gains(
    pred_score_sel: dict, pred_score_cur: dict,
    true_score_sel: dict, true_score_cur: dict,
):
    """计算动作关键部位增益和可见性增益。"""
    ap_pred_sel = pred_score_sel.get("S_action_part_pred", 0.0) if pred_score_sel else 0.0
    ap_pred_cur = pred_score_cur.get("S_action_part_pred", 0.0) if pred_score_cur else 0.0
    ap_true_sel = true_score_sel.get("S_action_part_true", 0.0) if true_score_sel else 0.0
    ap_true_cur = true_score_cur.get("S_action_part_true", 0.0) if true_score_cur else 0.0
    gain_ap_pred = ap_pred_sel - ap_pred_cur
    gain_ap_true = ap_true_sel - ap_true_cur

    skp_pred_sel = pred_score_sel.get("S_kp_pred", 0.0) if pred_score_sel else 0.0
    skp_pred_cur = pred_score_cur.get("S_kp_pred", 0.0) if pred_score_cur else 0.0
    gain_vis_pred = skp_pred_sel - skp_pred_cur

    skp_true_sel = true_score_sel.get("S_kp_true", 0.0) if true_score_sel else 0.0
    skp_true_cur = true_score_cur.get("S_kp_true", 0.0) if true_score_cur else 0.0
    gain_vis_true = skp_true_sel - skp_true_cur

    return gain_ap_pred, gain_ap_true, gain_vis_pred, gain_vis_true


# =============================================================================
# 单个 Episode
# =============================================================================

def run_one_episode(
    episode_id, config, runner, sampler,
    pred_evaluator, true_evaluator, policies,
    output_dir, metrics_writer,
) -> bool:
    """运行单个 episode。"""
    scene_id = os.path.splitext(os.path.basename(config["habitat"]["scene_path"]))[0]

    images_dir = os.path.join(output_dir, "images")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # =====================================================================
    # 1. 采样人体位置、姿态类型、人体朝向
    # =====================================================================
    human_pos = sample_valid_human_position(runner, config)
    pose_type = sample_pose_type(config)
    human_yaw = sample_human_yaw(config)
    print(f"  人体位置: {human_pos}")
    print(f"  姿态类型: {pose_type}")
    print(f"  人体朝向: {human_yaw:.3f} rad ({np.rad2deg(human_yaw):.0f}°)")

    # =====================================================================
    # 2. 生成带有朝向的世界坐标骨架
    # =====================================================================
    skeleton = get_skeleton(
        human_base_pos=human_pos,
        pose_type=pose_type,
        human_yaw=human_yaw,
    )

    # =====================================================================
    # 3. 采样机器人起始位置
    # =====================================================================
    robot_start_pos = sample_robot_start_position_around_human(
        runner=runner, human_pos=human_pos, config=config,
    )
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)
    print(f"  机器人起始: {robot_start_pos}, 朝向: {robot_start_yaw:.3f}")

    # =====================================================================
    # 4. 构造 current_view，计算 pred_score
    # =====================================================================
    current_view = CandidateView(
        candidate_id=-1, position=robot_start_pos, yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
    )

    current_view.pred_score = pred_evaluator.score_view_pred(
        view_pos=current_view.position, view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos, human_base_pos=human_pos,
        human_yaw=human_yaw, pose_type=pose_type,
        human_skeleton=skeleton, geodesic_distance=0.0,
    )

    # =====================================================================
    # 5. 采样候选位姿，计算 pred_score
    # =====================================================================
    candidates = sampler.sample(human_pos=human_pos, robot_pos=robot_start_pos, runner=runner)
    valid_candidates = [c for c in candidates if c.is_valid]
    print(f"  候选位姿: {len(candidates)} 个（有效 {len(valid_candidates)} 个）")

    if len(valid_candidates) == 0:
        metrics_writer.write_episode_summary({
            "episode_id": episode_id, "scene_id": scene_id,
            "status": "failed", "reason": "no_valid_candidates",
            "pose_type": pose_type, "human_yaw": human_yaw,
            "human_base_pos": human_pos.tolist(),
            "robot_start_pos": robot_start_pos.tolist(),
            "valid_candidate_count": 0,
        })
        return False

    for cand in valid_candidates:
        cand.pred_score = pred_evaluator.score_view_pred(
            view_pos=cand.position, view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos, human_base_pos=human_pos,
            human_yaw=human_yaw, pose_type=pose_type,
            human_skeleton=skeleton, geodesic_distance=cand.geodesic_distance,
        )

    # =====================================================================
    # 6. 策略根据 Q_pred 选择位姿
    # =====================================================================
    selected_by_policy = {}
    for policy in policies:
        selected = policy.select(current_view, candidates)
        selected_by_policy[policy.name] = selected
        if selected is not current_view:
            selected.selected_by.append(policy.name)

    # =====================================================================
    # 7. 渲染 current_view，计算 true_score
    # =====================================================================
    print(f"  渲染 current_view...")
    obs_current = runner.render_at(current_view.position, current_view.yaw)
    if obs_current["rgb"] is not None:
        save_rgb_image(obs_current["rgb"],
                       os.path.join(images_dir, f"ep_{episode_id:03d}_current.png"))

    current_view.true_score = true_evaluator.score_view_true(
        obs=obs_current, view_pos=current_view.position,
        view_yaw=current_view.yaw, human_base_pos=human_pos,
        human_yaw=human_yaw, pose_type=pose_type,
        human_skeleton=skeleton,
    )

    # =====================================================================
    # 8. 渲染各策略选中位姿，计算 true_score，写指标行
    # =====================================================================
    for policy in policies:
        selected = selected_by_policy[policy.name]
        is_current = (selected is current_view)

        try:
            obs = runner.render_at(selected.position, selected.yaw)
            if obs["rgb"] is not None:
                fname = f"ep_{episode_id:03d}_{policy.name.lower()}.png"
                save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        except Exception as e:
            print(f"    ⚠ 渲染失败（{policy.name}）: {e}")
            obs = obs_current

        if selected is current_view:
            true_score = current_view.true_score
        else:
            if not selected.true_score:
                selected.true_score = true_evaluator.score_view_true(
                    obs=obs, view_pos=selected.position, view_yaw=selected.yaw,
                    human_base_pos=human_pos, human_yaw=human_yaw,
                    pose_type=pose_type, human_skeleton=skeleton,
                )
            true_score = selected.true_score

        pred_score = selected.pred_score if selected.pred_score else current_view.pred_score

        # 计算差异指标
        gap = true_score.get("Q_true", 0.0) - pred_score.get("Q_pred", 0.0)
        gain_ap_pred, gain_ap_true, gain_vis_pred, gain_vis_true = compute_gains(
            selected.pred_score if selected.pred_score else current_view.pred_score,
            current_view.pred_score,
            true_score, current_view.true_score,
        )

        rel_angle = pred_score.get("relative_view_angle",
                     true_score.get("relative_view_angle_true", 0.0))

        row = {
            "episode_id": episode_id, "scene_id": scene_id,
            "policy": policy.name, "status": "success",
            "num_candidates": len(valid_candidates),
            "selected_is_current": int(is_current),
            "pose_type": pose_type, "human_yaw": human_yaw,
            "relative_view_angle": rel_angle,
            "human_x": human_pos[0], "human_y": human_pos[1], "human_z": human_pos[2],
            "robot_start_x": robot_start_pos[0], "robot_start_y": robot_start_pos[1],
            "robot_start_z": robot_start_pos[2],
            "selected_x": selected.position[0], "selected_y": selected.position[1],
            "selected_z": selected.position[2], "selected_yaw": selected.yaw,
            "geodesic_distance": selected.geodesic_distance,
            # 预测指标
            "S_action_part_pred": pred_score.get("S_action_part_pred", 0.0),
            "S_kp_pred": pred_score.get("S_kp_pred", 0.0),
            "S_orient_pred": pred_score.get("S_orient_pred", 0.0),
            "S_center_pred": pred_score.get("S_center_pred", 0.0),
            "S_dist_pred": pred_score.get("S_dist_pred", 0.0),
            "C_move": pred_score.get("C_move", 0.0),
            "Q_pred": pred_score.get("Q_pred", 0.0),
            "torso_visibility_pred": pred_score.get("torso_visibility_pred", 0.0),
            "lower_body_visibility_pred": pred_score.get("lower_body_visibility_pred", 0.0),
            "head_visibility_pred": pred_score.get("head_visibility_pred", 0.0),
            "arms_visibility_pred": pred_score.get("arms_visibility_pred", 0.0),
            # 真实指标
            "S_action_part_true": true_score.get("S_action_part_true", 0.0),
            "S_kp_true": true_score.get("S_kp_true", 0.0),
            "S_orient_true": true_score.get("S_orient_true", 0.0),
            "S_center_true": true_score.get("S_center_true", 0.0),
            "S_dist_true": true_score.get("S_dist_true", 0.0),
            "Q_true": true_score.get("Q_true", 0.0),
            "torso_visibility_true": true_score.get("torso_visibility_true", 0.0),
            "lower_body_visibility_true": true_score.get("lower_body_visibility_true", 0.0),
            "head_visibility_true": true_score.get("head_visibility_true", 0.0),
            "arms_visibility_true": true_score.get("arms_visibility_true", 0.0),
            # 差异指标
            "pred_true_gap": gap,
            "action_part_gain_pred": gain_ap_pred,
            "action_part_gain_true": gain_ap_true,
            "visibility_gain_pred": gain_vis_pred,
            "visibility_gain_true": gain_vis_true,
        }
        metrics_writer.write_metric_row(row)

    # =====================================================================
    # 9. 保存调试信息 + 写入 episode 摘要
    # =====================================================================
    debug_path = os.path.join(debug_dir, f"ep_{episode_id:03d}_candidates.json")
    episode_info = {
        "episode_id": episode_id, "scene_id": scene_id,
        "pose_type": pose_type, "human_yaw": human_yaw,
        "human_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
    }
    save_candidate_debug_json(candidates, debug_path, episode_info=episode_info)

    # 摘要
    metrics_writer.write_episode_summary({
        "episode_id": episode_id, "scene_id": scene_id,
        "status": "success", "pose_type": pose_type, "human_yaw": human_yaw,
        "human_base_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "valid_candidate_count": len(valid_candidates),
        "fixed_Q_pred": selected_by_policy["Fixed"].pred_score.get("Q_pred", 0.0),
        "ours_Q_pred": selected_by_policy["Ours"].pred_score.get("Q_pred", 0.0),
        "ours_selected_is_current": (selected_by_policy["Ours"] is current_view),
    })

    return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.episodes is not None:
        config["episode"]["num_episodes"] = args.episodes

    set_random_seed(config["project"]["seed"])

    num_episodes = config["episode"]["num_episodes"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"EA-AVS-MVP v3.0: 运行 {num_episodes} 个 episodes")
    print(f"  配置文件: {args.config}")
    print(f"  输出目录: {output_dir}")
    print(f"  场景文件: {config['habitat']['scene_path']}")
    print(f"  支持姿态: {config['human']['pose_types']}")
    print()

    runner = HabitatRunner(config)
    sampler = CandidateSampler(config)
    pred_evaluator = PredictiveEvaluator(config)
    true_evaluator = TrueEvaluator(config)
    metrics_writer = MetricsWriter(output_dir)

    policies = [
        FixedPolicy(),
        RandomPolicy(seed=config["project"]["seed"]),
        NearestPolicy(),
        OursPolicy(),
    ]

    success_count = 0
    failed_count = 0

    for episode_id in range(num_episodes):
        print(f"Episode {episode_id + 1}/{num_episodes}:")
        try:
            success = run_one_episode(
                episode_id, config, runner, sampler,
                pred_evaluator, true_evaluator, policies,
                output_dir, metrics_writer,
            )
            if success:
                success_count += 1
                print(f"  -> 成功 ✅")
            else:
                failed_count += 1
                print(f"  -> 失败（无有效候选点）❌")
        except Exception as e:
            failed_count += 1
            print(f"  -> 错误: {e} ❌")
            traceback.print_exc()
            metrics_writer.write_episode_summary({
                "episode_id": episode_id,
                "scene_id": os.path.splitext(
                    os.path.basename(config["habitat"]["scene_path"])
                )[0],
                "status": "failed", "reason": str(e),
            })
        print()

    metrics_writer.close()
    runner.close()

    print("=" * 60)
    print(f"实验完成: {success_count} 成功, {failed_count} 失败")
    print(f"输出目录: {output_dir}")
    print(f"  metrics.csv    —— 含动作感知和朝向指标")
    print(f"  episodes.jsonl —— Episode 摘要")
    print(f"  images/        —— 渲染图像")
    print(f"  debug/         —— 候选点调试（含姿态/朝向信息）")
    print("=" * 60)


if __name__ == "__main__":
    main()
