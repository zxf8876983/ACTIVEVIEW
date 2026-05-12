#!/usr/bin/env python
"""
EA-AVS-MVP0.1 主入口脚本 —— run_mvp_visibility.py
====================================================

中文名称：面向老人动作感知的移动机器人主动视角选择 - 最小可运行版本

功能：
    串联所有模块，运行完整的主动视角选择对比实验。

运行命令：
    python scripts/run_mvp_visibility.py \\
        --config configs/mvp_visibility.yaml \\
        --episodes 20 \\
        --output-dir outputs/test_run

主流程（每个 episode）：
    1. 从 navmesh 采样一个人体位置
    2. 生成 standing 抽象骨架
    3. 围绕人体采样机器人起始位置（yaw 朝向人体）
    4. 对初始视角（current_view）计算视角质量评分
    5. 围绕人体采样候选视角点（CandidateSampler）
    6. 对每个有效候选点计算视角质量评分
    7. 四种策略（Fixed/Random/Nearest/Ours）各选一个视角
    8. 渲染并保存每种策略选中视角的 RGB 图像
    9. 输出 metrics.csv、episodes.jsonl、candidates debug json
    10. 遇到单个 episode 失败时记录失败并继续，不崩溃

验收标准：
    - 20 个 episodes 全部跑完，不崩溃
    - 成功 episode ≥ 14 个
    - metrics.csv 行数 ≥ 56
    - mean(S_kp_ours) > mean(S_kp_fixed)
    - mean(Q_ours) > mean(Q_fixed)
"""

import argparse
import os
import sys
import traceback

import numpy as np

# ---------- 将项目根目录加入 Python 路径 ----------
# 这样可以直接 from ea_avs.xxx import ... 而不需要安装包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs.config import load_config
from ea_avs.habitat_runner import HabitatRunner
from ea_avs.skeleton import get_skeleton
from ea_avs.candidate_sampler import CandidateSampler, CandidateView
from ea_avs.evaluator import ViewpointEvaluator
from ea_avs.policies import FixedPolicy, RandomPolicy, NearestPolicy, OursPolicy
from ea_avs.metrics import MetricsWriter
from ea_avs.visualization import (
    save_rgb_image,
    save_candidate_debug_json,
)
from ea_avs.geometry import compute_look_at_yaw


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args():
    """
    解析命令行参数。

    支持三个参数：
        --config:    YAML 配置文件路径（必需）
        --episodes:  覆盖配置文件中的 episode 数量（可选）
        --output-dir: 输出目录（可选，默认 outputs/test_run）
    """
    parser = argparse.ArgumentParser(
        description="EA-AVS-MVP0.1: 主动视角选择实验"
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
        "--output-dir", type=str, default="outputs/test_run",
        help="输出目录（默认 outputs/test_run）"
    )
    return parser.parse_args()


# =============================================================================
# 辅助函数
# =============================================================================

def set_random_seed(seed: int):
    """
    设置随机种子，确保实验可复现。

    同时设置 numpy 和 Python random 模块的种子。
    注意：Habitat-Sim 内部的随机采样使用独立的种子机制，
    不受此函数影响。
    """
    np.random.seed(seed)
    import random
    random.seed(seed)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两点之间的欧氏距离（直线距离）。

    参数：
        a, b: 三维坐标点，shape=(3,)。

    返回：
        欧氏距离（米）。
    """
    return float(np.linalg.norm(a - b))


# =============================================================================
# 采样函数
# =============================================================================

def sample_valid_human_position(
    runner: HabitatRunner, config: dict
) -> np.ndarray:
    """
    在导航网格上采样一个可用的"人体位置"。

    采样策略：
        随机从 navmesh 上采样可达点，直到找到一个有效的。
        人体位置只是抽象的"站立点"，不需要在导航网格上，
        但此处使用 navmesh 点确保位置在场景中可达的区域内。

    参数：
        runner: HabitatRunner 实例，提供 navmesh 查询功能。
        config: 配置字典（需要 max_sampling_tries）。

    返回：
        shape=(3,) 的 numpy 数组，表示人体基座位置。

    抛出异常：
        RuntimeError: 超过最大尝试次数后仍未找到有效位置。
    """
    max_tries = config["episode"]["max_sampling_tries"]
    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if runner.is_navigable(pt):
            return pt
    raise RuntimeError(
        f"在 {max_tries} 次尝试后仍无法采样到有效的人体位置"
    )


def sample_robot_start_position_around_human(
    runner: HabitatRunner,
    human_pos: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    围绕人体位置采样一个机器人的起始位置。

    约束条件：
        1. 必须在导航网格上（可达）
        2. 到人体的欧氏距离在 [min_robot_human_distance, max_robot_human_distance] 范围
        3. 从机器人位置到人体位置的测地路径必须存在

    参数：
        runner: HabitatRunner 实例。
        human_pos: 人体基座位置，shape=(3,)。
        config: 配置字典（需要 episode 配置段）。

    返回：
        shape=(3,) 的 numpy 数组，表示机器人起始位置。

    抛出异常：
        RuntimeError: 超过最大尝试次数后仍未找到有效位置。
    """
    ep_cfg = config["episode"]
    min_dist = ep_cfg["min_robot_human_distance"]   # 最小距离（默认 1.5m）
    max_dist = ep_cfg["max_robot_human_distance"]   # 最大距离（默认 4.0m）
    max_tries = ep_cfg["max_sampling_tries"]          # 最大尝试次数

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

    raise RuntimeError(
        f"在 {max_tries} 次尝试后仍无法采样到机器人起始位置"
    )


# =============================================================================
# 单个 Episode 的运行逻辑
# =============================================================================

def run_one_episode(
    episode_id: int,
    config: dict,
    runner: HabitatRunner,
    sampler: CandidateSampler,
    evaluator: ViewpointEvaluator,
    policies: list,
    output_dir: str,
    metrics_writer: MetricsWriter,
) -> bool:
    """
    运行单个完整的实验 Episode。

    参数：
        episode_id:   episode 编号（从 0 开始）。
        config:       配置字典。
        runner:       Habitat 运行器。
        sampler:      候选视角采样器。
        evaluator:    视角质量评估器。
        policies:     策略列表 [Fixed, Random, Nearest, Ours]。
        output_dir:   输出目录。
        metrics_writer: 指标写入器。

    返回：
        True 表示 episode 成功完成，False 表示失败
        （如没有有效候选点）。

    Episode 详细流程：
        Step  1: 采样人体位置
        Step  2: 生成 standing 骨架
        Step  3: 采样机器人起始位置并计算朝向
        Step  4: 创建初始视角对象
        Step  5: 评估初始视角质量
        Step  6: 采样候选视角点
        Step  7: 评估所有候选视角质量
        Step  8: 四种策略各选最佳视角，渲染并保存图像
        Step  9: 保存候选点调试 JSON
        Step 10: 写入 episode 摘要（JSONL）
    """
    # ---------- 获取场景名称（用于记录） ----------
    scene_path = config["habitat"]["scene_path"]
    scene_id = os.path.splitext(os.path.basename(scene_path))[0]

    # ---------- 创建输出子目录 ----------
    images_dir = os.path.join(output_dir, "images")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # =====================================================================
    # Step 1: 从导航网格采样人体位置
    # =====================================================================
    human_pos = sample_valid_human_position(runner, config)
    print(f"  人体位置: {human_pos}")

    # =====================================================================
    # Step 2: 生成 standing 抽象骨架
    # =====================================================================
    skeleton = get_skeleton(
        human_base_pos=human_pos,
        pose_type=config["human"]["pose_type"],
    )

    # =====================================================================
    # Step 3: 采样机器人起始位置
    # =====================================================================
    robot_start_pos = sample_robot_start_position_around_human(
        runner=runner,
        human_pos=human_pos,
        config=config,
    )
    # 计算机器人朝向，使其"看向"人体
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)
    print(f"  机器人起始: {robot_start_pos}, 朝向角: {robot_start_yaw:.3f}")

    # =====================================================================
    # Step 4: 创建初始视角（current_view）
    # =====================================================================
    # candidate_id=-1 表示这不是候选点，而是初始视角
    # geodesic_distance=0.0 表示不需要移动
    current_view = CandidateView(
        candidate_id=-1,
        position=robot_start_pos,
        yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
        invalid_reason="",
        score={},
    )

    # =====================================================================
    # Step 5: 评估初始视角质量
    # =====================================================================
    current_view.score = evaluator.score_view(
        view_pos=current_view.position,
        view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos,
        human_base_pos=human_pos,
        human_skeleton=skeleton,
        geodesic_distance=0.0,
    )

    # =====================================================================
    # Step 6: 采样候选视角
    # =====================================================================
    candidates = sampler.sample(
        human_pos=human_pos,
        robot_pos=robot_start_pos,
        runner=runner,
    )

    valid_candidates = [c for c in candidates if c.is_valid]
    print(f"  候选视角: {len(candidates)} 个（有效 {len(valid_candidates)} 个）")

    # =====================================================================
    # 检查：如果没有有效候选点，记录失败并返回
    # =====================================================================
    if len(valid_candidates) == 0:
        metrics_writer.write_episode_summary({
            "episode_id": episode_id,
            "scene_id": scene_id,
            "status": "failed",
            "reason": "no_valid_candidates",  # 没有有效候选点
            "human_base_pos": human_pos.tolist(),
            "robot_start_pos": robot_start_pos.tolist(),
            "valid_candidate_count": 0,
        })
        return False

    # =====================================================================
    # Step 7: 对所有有效候选点评估视角质量
    # =====================================================================
    for cand in valid_candidates:
        cand.score = evaluator.score_view(
            view_pos=cand.position,
            view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos,
            human_base_pos=human_pos,
            human_skeleton=skeleton,
            geodesic_distance=cand.geodesic_distance,
        )

    # =====================================================================
    # Step 8: 四种策略各选最佳视角，渲染并保存图像
    # =====================================================================
    for policy in policies:
        # 策略选择最佳视角
        selected = policy.select(current_view, candidates)
        is_fallback = selected is current_view  # 是否退回到初始视角

        # 渲染选中视角的 RGB 图像
        try:
            obs = runner.render_at(selected.position, selected.yaw)
            if obs["rgb"] is not None:
                fname = f"ep_{episode_id:03d}_{policy.name.lower()}.png"
                save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        except Exception as e:
            print(f"    ⚠ 渲染失败（{policy.name}）: {e}")

        # 获取评分
        score = selected.score if selected.score else current_view.score
        geo_dist = selected.geodesic_distance

        # 写入 metrics.csv 的一行
        row = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "policy": policy.name,
            "status": "success",
            "num_candidates": len(valid_candidates),
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
            "geodesic_distance": geo_dist,
            "S_kp": score.get("S_kp", 0.0),
            "S_center": score.get("S_center", 0.0),
            "S_dist": score.get("S_dist", 0.0),
            "C_move": score.get("C_move", 0.0),
            "Q": score.get("Q", 0.0),
            "torso_visibility": score.get("torso_visibility", 0.0),
            "lower_body_visibility": score.get("lower_body_visibility", 0.0),
            "head_visibility": score.get("head_visibility", 0.0),
        }
        metrics_writer.write_metric_row(row)

    # =====================================================================
    # Step 9: 保存候选点调试 JSON
    # =====================================================================
    debug_path = os.path.join(debug_dir, f"ep_{episode_id:03d}_candidates.json")
    save_candidate_debug_json(candidates, debug_path)

    # =====================================================================
    # Step 10: 写入 episode 摘要（JSONL）
    # =====================================================================
    fixed_score = None
    ours_score = None
    for policy in policies:
        selected = policy.select(current_view, candidates)
        score = selected.score if selected.score else current_view.score
        if policy.name == "Fixed":
            fixed_score = score.get("Q", 0.0)
        elif policy.name == "Ours":
            ours_score = score.get("Q", 0.0)

    metrics_writer.write_episode_summary({
        "episode_id": episode_id,
        "scene_id": scene_id,
        "status": "success",
        "human_base_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "valid_candidate_count": len(valid_candidates),
        "fixed_score": fixed_score,
        "ours_score": ours_score,
        "ours_improved": ours_score is not None and fixed_score is not None
                         and ours_score > fixed_score,
    })

    return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数 —— 串联所有模块运行完整的主动视角选择实验。

    主流程：
        1. 解析命令行参数
        2. 加载 YAML 配置文件
        3. 设置随机种子
        4. 初始化 Habitat 模拟器、采样器、评估器
        5. 创建四种策略实例和指标写入器
        6. 循环运行指定数量的 episodes
        7. 输出实验总结
    """
    # ---------- 1. 解析参数 ----------
    args = parse_args()

    # ---------- 2. 加载配置 ----------
    config = load_config(args.config)

    # 如果命令行指定了 --episodes，覆盖配置文件中的值
    if args.episodes is not None:
        config["episode"]["num_episodes"] = args.episodes

    # ---------- 3. 设置随机种子 ----------
    set_random_seed(config["project"]["seed"])

    num_episodes = config["episode"]["num_episodes"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"EA-AVS-MVP0.1: 运行 {num_episodes} 个 episodes")
    print(f"  配置文件: {args.config}")
    print(f"  输出目录: {output_dir}")
    print(f"  场景文件: {config['habitat']['scene_path']}")
    print()

    # ---------- 4. 初始化核心模块 ----------
    runner = HabitatRunner(config)      # Habitat 模拟器
    sampler = CandidateSampler(config)  # 候选视角采样器
    evaluator = ViewpointEvaluator(config)  # 视角质量评估器
    metrics_writer = MetricsWriter(output_dir)  # 指标写入器

    # ---------- 5. 创建四种策略 ----------
    policies = [
        FixedPolicy(),                                   # 固定视角（基线）
        RandomPolicy(seed=config["project"]["seed"]),    # 随机选择
        NearestPolicy(),                                 # 最近距离
        OursPolicy(),                                    # 最优评分（核心）
    ]

    # ---------- 6. 运行所有 Episodes ----------
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
                evaluator=evaluator,
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
            # 关键设计：单个 episode 失败不崩溃整个实验
            failed_count += 1
            print(f"  -> 错误: {e} ❌")
            traceback.print_exc()

            # 即使失败也写入 JSONL 记录
            metrics_writer.write_episode_summary({
                "episode_id": episode_id,
                "scene_id": os.path.splitext(
                    os.path.basename(config["habitat"]["scene_path"])
                )[0],
                "status": "failed",
                "reason": str(e),
                "human_base_pos": None,
                "robot_start_pos": None,
                "valid_candidate_count": 0,
            })

        print()

    # ---------- 关闭所有资源 ----------
    metrics_writer.close()
    runner.close()

    # ---------- 7. 输出实验总结 ----------
    print("=" * 60)
    print(f"实验完成: {success_count} 成功, {failed_count} 失败")
    print(f"输出目录: {output_dir}")
    print(f"  metrics.csv    —— 结构化指标数据")
    print(f"  episodes.jsonl —— Episode 摘要")
    print(f"  images/        —— 渲染图像")
    print(f"  debug/         —— 候选点调试信息")
    print("=" * 60)

    # 检查 metrics.csv 是否生成
    metrics_path = os.path.join(output_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            line_count = sum(1 for _ in f) - 1  # 减 1 排除表头
        print(f"metrics.csv 数据行数（不含表头）: {line_count}")
    else:
        print("⚠ 警告: metrics.csv 未生成！")


if __name__ == "__main__":
    main()
