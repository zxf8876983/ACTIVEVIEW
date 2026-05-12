#!/usr/bin/env python
"""
EA-AVS-MVP v2.0 主入口 —— run_mvp20_visibility.py
====================================================

中文名称：面向老人动作感知的移动机器人主动视角选择 v2.0 —— 规范化实验

v2.0 核心改进：
    - 严格区分预测评分（pred_score）和真实评估（true_score）
    - 选择阶段禁止使用候选点渲染图像
    - OursPolicy 必须允许不移动（将 current_view 一起比较 Q_pred）
    - true_score 在策略选择并渲染后才能计算
    - metrics 同时输出 pred 和 true 指标

运行命令：
    python scripts/run_mvp20_visibility.py \\
        --config configs/mvp20_visibility.yaml \\
        --episodes 20 \\
        --output-dir outputs/mvp20_test_run

主流程：
    1.  采样人体位置 + 生成骨架
    2.  采样机器人起始位置
    3.  构造 current_view，计算 pred_score（几何预测）
    4.  采样候选位姿，计算 pred_score（几何预测）
    5.  四种策略根据 pred_score 选择位姿
    6.  渲染 current_view 图像，计算 true_score
    7.  渲染各策略选中位姿图像，计算 true_score
    8.  写入 metrics.csv 和 episodes.jsonl
"""

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v2.config import load_config
from ea_avs_v2.habitat_runner import HabitatRunner
from ea_avs_v2.skeleton import get_skeleton
from ea_avs_v2.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v2.predictive_evaluator import PredictiveEvaluator
from ea_avs_v2.true_evaluator import TrueEvaluator
from ea_avs_v2.policies import FixedPolicy, RandomPolicy, NearestPolicy, OursPolicy
from ea_avs_v2.metrics import MetricsWriter
from ea_avs_v2.visualization import save_rgb_image, save_candidate_debug_json
from ea_avs_v2.geometry import compute_look_at_yaw


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EA-AVS-MVP v2.0: 主动视角选择规范化实验"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置文件路径"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="覆盖配置文件中的 episode 数量"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/mvp20_test_run",
        help="输出目录"
    )
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


def sample_valid_human_position(
    runner: HabitatRunner, config: dict
) -> np.ndarray:
    """从导航网格采样人体位置。"""
    max_tries = config["episode"]["max_sampling_tries"]
    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if runner.is_navigable(pt):
            return pt
    raise RuntimeError(f"无法采样人体位置（{max_tries} 次尝试后）")


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
        geo = runner.geodesic_distance(pt, human_pos)
        if geo == float("inf"):
            continue
        return pt
    raise RuntimeError(f"无法采样机器人起始位置（{max_tries} 次尝试后）")


def compute_visibility_gain(selected_pred_score: dict, current_pred_score: dict,
                            selected_true_score: dict, current_true_score: dict):
    """计算可见性增益。

    返回：
        (visibility_gain_pred, visibility_gain_true)
        表示选中位姿相比当前位置的 S_kp 提升（可能为负）。
    """
    skp_pred_sel = selected_pred_score.get("S_kp_pred", 0.0) if selected_pred_score else 0.0
    skp_pred_cur = current_pred_score.get("S_kp_pred", 0.0) if current_pred_score else 0.0
    gain_pred = skp_pred_sel - skp_pred_cur

    skp_true_sel = selected_true_score.get("S_kp_true", 0.0) if selected_true_score else 0.0
    skp_true_cur = current_true_score.get("S_kp_true", 0.0) if current_true_score else 0.0
    gain_true = skp_true_sel - skp_true_cur

    return gain_pred, gain_true


# =============================================================================
# 单个 Episode
# =============================================================================

def run_one_episode(
    episode_id: int,
    config: dict,
    runner: HabitatRunner,
    sampler: CandidateSampler,
    pred_evaluator: PredictiveEvaluator,
    true_evaluator: TrueEvaluator,
    policies: list,
    output_dir: str,
    metrics_writer: MetricsWriter,
) -> bool:
    """运行单个 episode。

    v2.0 严格流程：
        1. 采样人体位置 + 骨架（第 3 步前无需渲染）
        2. 采样机器人起点
        3. 构造 current_view，只算 pred_score（几何预测，不渲染）
        4. 采样候选位姿，只算 pred_score（几何预测，不渲染）
        5. 策略根据 pred_score 选择位姿
        6. 渲染 current_view，算 true_score
        7. 渲染各策略选中位姿，算 true_score
        8. 写入指标
    """
    scene_path = config["habitat"]["scene_path"]
    scene_id = os.path.splitext(os.path.basename(scene_path))[0]

    images_dir = os.path.join(output_dir, "images")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # =====================================================================
    # 1. 采样人体位置 + 生成骨架
    # =====================================================================
    human_pos = sample_valid_human_position(runner, config)
    print(f"  人体位置: {human_pos}")

    skeleton = get_skeleton(
        human_base_pos=human_pos,
        pose_type=config["human"]["pose_type"],
    )

    # =====================================================================
    # 2. 采样机器人起始位置
    # =====================================================================
    robot_start_pos = sample_robot_start_position_around_human(
        runner=runner, human_pos=human_pos, config=config,
    )
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)
    print(f"  机器人起始: {robot_start_pos}, 朝向: {robot_start_yaw:.3f}")

    # =====================================================================
    # 3. 构造 current_view，计算 pred_score
    #    ⚠ 这里不调用 render_at()，只做几何预测
    # =====================================================================
    current_view = CandidateView(
        candidate_id=-1,
        position=robot_start_pos,
        yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
    )

    current_view.pred_score = pred_evaluator.score_view_pred(
        view_pos=current_view.position,
        view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos,
        human_base_pos=human_pos,
        human_skeleton=skeleton,
        geodesic_distance=0.0,
    )

    # =====================================================================
    # 4. 采样候选位姿，计算 pred_score
    #    ⚠ 这里不调用 render_at()，只做几何预测
    # =====================================================================
    candidates = sampler.sample(
        human_pos=human_pos,
        robot_pos=robot_start_pos,
        runner=runner,
    )

    valid_candidates = [c for c in candidates if c.is_valid]
    print(f"  候选位姿: {len(candidates)} 个（有效 {len(valid_candidates)} 个）")

    if len(valid_candidates) == 0:
        # 无有效候选点，写失败摘要
        metrics_writer.write_episode_summary({
            "episode_id": episode_id,
            "scene_id": scene_id,
            "status": "failed",
            "reason": "no_valid_candidates",
            "human_base_pos": human_pos.tolist(),
            "robot_start_pos": robot_start_pos.tolist(),
            "valid_candidate_count": 0,
        })
        return False

    for cand in valid_candidates:
        cand.pred_score = pred_evaluator.score_view_pred(
            view_pos=cand.position,
            view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos,
            human_base_pos=human_pos,
            human_skeleton=skeleton,
            geodesic_distance=cand.geodesic_distance,
        )

    # =====================================================================
    # 5. 策略根据 pred_score 选择位姿
    #    ⚠ 策略只能使用 pred_score，不能使用 true_score
    #    ⚠ OursPolicy 将 current_view 与候选点一起比较 Q_pred
    # =====================================================================
    selected_by_policy = {}
    for policy in policies:
        selected = policy.select(current_view, candidates)
        selected_by_policy[policy.name] = selected
        # 记录选中策略名
        if selected not in [current_view]:
            selected.selected_by.append(policy.name)

    # =====================================================================
    # 6. 渲染 current_view，计算 true_score
    # =====================================================================
    print(f"  渲染 current_view...")
    obs_current = runner.render_at(current_view.position, current_view.yaw)
    if obs_current["rgb"] is not None:
        save_rgb_image(obs_current["rgb"],
                       os.path.join(images_dir, f"ep_{episode_id:03d}_current.png"))

    current_view.true_score = true_evaluator.score_view_true(
        obs=obs_current,
        view_pos=current_view.position,
        view_yaw=current_view.yaw,
        human_base_pos=human_pos,
        human_skeleton=skeleton,
    )

    # =====================================================================
    # 7. 渲染各策略选中位姿，计算 true_score，写指标行
    # =====================================================================
    for policy in policies:
        selected = selected_by_policy[policy.name]
        is_current = (selected is current_view)

        # 渲染选中位姿
        try:
            obs = runner.render_at(selected.position, selected.yaw)
            if obs["rgb"] is not None:
                fname = f"ep_{episode_id:03d}_{policy.name.lower()}.png"
                save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        except Exception as e:
            print(f"    ⚠ 渲染失败（{policy.name}）: {e}")
            obs = obs_current  # fallback

        # 计算 true_score（如果尚未计算）
        # 注：current_view 的 true_score 已在第 6 步计算
        if selected is current_view:
            true_score = current_view.true_score
        else:
            if not selected.true_score:
                selected.true_score = true_evaluator.score_view_true(
                    obs=obs,
                    view_pos=selected.position,
                    view_yaw=selected.yaw,
                    human_base_pos=human_pos,
                    human_skeleton=skeleton,
                )
            true_score = selected.true_score

        # 获取 pred_score
        pred_score = selected.pred_score if selected.pred_score else current_view.pred_score

        # 计算差异指标
        pred_true_gap = true_score.get("Q_true", 0.0) - pred_score.get("Q_pred", 0.0)
        gain_pred, gain_true = compute_visibility_gain(
            selected.pred_score if selected.pred_score else current_view.pred_score,
            current_view.pred_score,
            true_score,
            current_view.true_score,
        )

        # 写指标行
        row = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "policy": policy.name,
            "status": "success",
            "num_candidates": len(valid_candidates),
            "selected_is_current": int(is_current),
            "human_x": human_pos[0],
            "human_y": human_pos[1],
            "human_z": human_pos[2],
            "robot_start_x": robot_start_pos[0],
            "robot_start_y": robot_start_pos[1],
            "robot_start_z": robot_start_pos[2],
            "selected_x": selected.position[0],
            "selected_y": selected.position[1],
            "selected_z": selected.position[2],
            "selected_yaw": selected.yaw,
            "geodesic_distance": selected.geodesic_distance,
            # 预测指标
            "S_kp_pred": pred_score.get("S_kp_pred", 0.0),
            "S_center_pred": pred_score.get("S_center_pred", 0.0),
            "S_dist_pred": pred_score.get("S_dist_pred", 0.0),
            "C_move": pred_score.get("C_move", 0.0),
            "Q_pred": pred_score.get("Q_pred", 0.0),
            "torso_visibility_pred": pred_score.get("torso_visibility_pred", 0.0),
            "lower_body_visibility_pred": pred_score.get("lower_body_visibility_pred", 0.0),
            "head_visibility_pred": pred_score.get("head_visibility_pred", 0.0),
            # 真实指标
            "S_kp_true": true_score.get("S_kp_true", 0.0),
            "S_center_true": true_score.get("S_center_true", 0.0),
            "S_dist_true": true_score.get("S_dist_true", 0.0),
            "Q_true": true_score.get("Q_true", 0.0),
            "torso_visibility_true": true_score.get("torso_visibility_true", 0.0),
            "lower_body_visibility_true": true_score.get("lower_body_visibility_true", 0.0),
            "head_visibility_true": true_score.get("head_visibility_true", 0.0),
            # 差异指标
            "pred_true_gap": pred_true_gap,
            "visibility_gain_pred": gain_pred,
            "visibility_gain_true": gain_true,
        }
        metrics_writer.write_metric_row(row)

    # =====================================================================
    # 8. 保存调试信息 + 写入 episode 摘要
    # =====================================================================
    debug_path = os.path.join(debug_dir, f"ep_{episode_id:03d}_candidates.json")
    save_candidate_debug_json(candidates, debug_path)

    # 收集摘要数据
    fixed_q_pred = selected_by_policy["Fixed"].pred_score.get("Q_pred", 0.0) \
        if selected_by_policy["Fixed"].pred_score else 0.0
    ours_selected = selected_by_policy["Ours"]
    ours_q_pred = ours_selected.pred_score.get("Q_pred", 0.0) \
        if ours_selected.pred_score else 0.0
    ours_q_true = ours_selected.true_score.get("Q_true", 0.0) \
        if ours_selected.true_score else 0.0

    metrics_writer.write_episode_summary({
        "episode_id": episode_id,
        "scene_id": scene_id,
        "status": "success",
        "human_base_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "valid_candidate_count": len(valid_candidates),
        "fixed_Q_pred": fixed_q_pred,
        "ours_Q_pred": ours_q_pred,
        "ours_Q_true": ours_q_true,
        "ours_selected_is_current": (ours_selected is current_view),
        "ours_improved_pred": ours_q_pred > fixed_q_pred,
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

    print(f"EA-AVS-MVP v2.0: 运行 {num_episodes} 个 episodes")
    print(f"  配置文件: {args.config}")
    print(f"  输出目录: {output_dir}")
    print(f"  场景文件: {config['habitat']['scene_path']}")
    print()

    # 初始化
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
                episode_id=episode_id,
                config=config,
                runner=runner,
                sampler=sampler,
                pred_evaluator=pred_evaluator,
                true_evaluator=true_evaluator,
                policies=policies,
                output_dir=output_dir,
                metrics_writer=metrics_writer,
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
                "status": "failed",
                "reason": str(e),
            })
        print()

    metrics_writer.close()
    runner.close()

    print("=" * 60)
    print(f"实验完成: {success_count} 成功, {failed_count} 失败")
    print(f"输出目录: {output_dir}")
    print(f"  metrics.csv    —— 含 pred 和 true 双指标")
    print(f"  episodes.jsonl —— Episode 摘要")
    print(f"  images/        —— 渲染图像（含 current_view）")
    print(f"  debug/         —— 候选点调试信息")
    print("=" * 60)


if __name__ == "__main__":
    main()
