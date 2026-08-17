#!/usr/bin/env python
"""
EA-AVS-MVP v4.0 主入口 —— run_mvp40_occlusion_aware.py
=======================================================

中文名称：面向老人动作感知的移动机器人主动视角选择 v4.0 —— 遮挡感知版本

v4.0 核心升级（相比 v3.0）：
    1. Habitat 场景几何 ray casting（需开启 physics）
    2. 关键点遮挡判断：visible = in_fov AND NOT occluded
    3. 遮挡感知动作关键部位评分 S_action_occ_pred 参与 Q_pred
    4. 模块化消融策略与 Oracle 离线上界

运行命令：
    python scripts/run_mvp40_occlusion_aware.py \\
        --config configs/mvp40_occlusion_aware.yaml \\
        --episodes 20 \\
        --output-dir outputs/mvp40_test_run

主流程（严格遵循 pred/true 分离）：
    1.  采样人体位置 / 姿态类型 / 人体朝向并生成骨架
    2.  采样机器人起始位置，构造 current_view 并计算 pred_score（含 ray casting）
    3.  采样候选位姿并计算 pred_score
    4.  【策略选择阶段】各类策略仅使用 pred_score 完成选择（禁止未来 RGB/depth/Q_true）
    5.  渲染 current_view 和各策略选中位姿，计算 true_score
    6.  【评估阶段】离线计算所有候选位姿 true_score，Oracle 选上界
    7.  写入 metrics.csv / episodes.jsonl / debug JSON
"""

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v4.config import load_config
from ea_avs_v4.habitat_runner import HabitatRunner
from ea_avs_v4.skeleton import get_skeleton
from ea_avs_v4.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v4.predictive_evaluator import PredictiveEvaluator
from ea_avs_v4.true_evaluator import TrueEvaluator
from ea_avs_v4.policies import FixedPolicy, RandomPolicy, NearestPolicy, OursPolicy
from ea_avs_v4.ablation_policies import (
    VisibilityOnlyPolicy, ActionPartOnlyPolicy, OrientationOnlyPolicy,
    OcclusionOnlyPolicy, ActionOrientationPolicy,
)
from ea_avs_v4.oracle_policy import OraclePolicy, compute_oracle_gap
from ea_avs_v4.metrics import MetricsWriter
from ea_avs_v4.visualization import save_rgb_image, save_candidate_debug_json
from ea_avs_v4.geometry import compute_look_at_yaw


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EA-AVS-MVP v4.0: 遮挡感知动作导向主动视角选择"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖配置文件中的 episode 数量")
    parser.add_argument("--output-dir", type=str, default="outputs/mvp40_test_run",
                        help="输出目录")
    return parser.parse_args()


# =============================================================================
# 辅助函数
# =============================================================================

def set_random_seed(seed: int):
    """设置随机种子（numpy + python random）。"""
    np.random.seed(seed)
    import random
    random.seed(seed)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离。"""
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
    """根据配置随机采样一个姿态类型。"""
    human_cfg = config["human"]
    if config["episode"].get("randomize_pose_type", True):
        return str(np.random.choice(human_cfg["pose_types"]))
    return human_cfg["default_pose_type"]


def sample_human_yaw(config: dict) -> float:
    """根据配置采样人体朝向角（弧度）。"""
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
    """计算动作关键部位与可见性增益。"""
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

    # v4.0 遮挡增益：遮挡感知动作部位得分的真实变化
    occ_true_sel = true_score_sel.get("S_action_occ_true", 0.0) if true_score_sel else 0.0
    occ_true_cur = true_score_cur.get("S_action_occ_true", 0.0) if true_score_cur else 0.0
    gain_occ_true = occ_true_sel - occ_true_cur

    return gain_ap_pred, gain_ap_true, gain_vis_pred, gain_vis_true, gain_occ_true


def count_occlusion_errors(pred_score: dict) -> int:
    """统计 pred_score 中 ray cast 失败的关键点数量（含 "error" 字段）。"""
    occ = pred_score.get("occlusion_result", {}) if pred_score else {}
    if not isinstance(occ, dict):
        return 0
    return sum(1 for v in occ.values()
               if isinstance(v, dict) and "error" in v)


def _render_and_score(
    runner, true_evaluator, view, human_pos, human_yaw, pose_type,
    skeleton, images_dir, ep_str, save_image: bool,
) -> dict:
    """渲染单个视角并计算 true_score（带异常保护）。

    参数：
        runner: HabitatRunner 实例。
        true_evaluator: TrueEvaluator 实例。
        view: 待处理的 CandidateView。
        ...骨架相关参数...
        images_dir: 图像输出目录。
        ep_str: 用于文件名的 episode 前缀，如 "ep_000"。
        save_image: 是否保存渲染图像。

    返回：
        渲染的观测字典（失败时返回 None）。
    """
    try:
        obs = runner.render_at(view.position, view.yaw)
        if save_image and obs["rgb"] is not None:
            fname = f"{ep_str}_view_{view.candidate_id}.png"
            save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        view.true_score = true_evaluator.score_view_true(
            runner=runner, obs=obs, view_pos=view.position,
            view_yaw=view.yaw, human_base_pos=human_pos,
            human_yaw=human_yaw, pose_type=pose_type,
            human_skeleton=skeleton,
        )
        return obs
    except Exception as e:
        print(f"    ⚠ 渲染/评估失败 (candidate {view.candidate_id}): {e}")
        return None


# =============================================================================
# 单个 Episode
# =============================================================================

def run_one_episode(
    episode_id, config, runner, sampler,
    pred_evaluator, true_evaluator, policies, oracle_policy,
    output_dir, metrics_writer, oracle_enabled,
) -> bool:
    """运行单个 episode。"""
    scene_id = os.path.splitext(os.path.basename(config["habitat"]["scene_path"]))[0]
    ep_str = f"ep_{episode_id:03d}"

    images_dir = os.path.join(output_dir, "images")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    save_images = config["output"].get("save_images", True)

    # =====================================================================
    # 1. 采样人体位置 / 姿态类型 / 人体朝向
    # =====================================================================
    human_pos = sample_valid_human_position(runner, config)
    pose_type = sample_pose_type(config)
    human_yaw = sample_human_yaw(config)
    print(f"  人体位置: {np.round(human_pos, 3)}  姿态: {pose_type}  "
          f"朝向: {np.rad2deg(human_yaw):.0f}°")

    # =====================================================================
    # 2. 生成世界坐标骨架
    # =====================================================================
    skeleton = get_skeleton(
        human_base_pos=human_pos, pose_type=pose_type, human_yaw=human_yaw,
    )

    # =====================================================================
    # 3. 采样机器人起始位置
    # =====================================================================
    robot_start_pos = sample_robot_start_position_around_human(
        runner=runner, human_pos=human_pos, config=config,
    )
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)

    # =====================================================================
    # 4. 构造 current_view 并计算 pred_score（含 ray casting）
    # =====================================================================
    current_view = CandidateView(
        candidate_id=-1, position=robot_start_pos, yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
    )
    current_view.pred_score = pred_evaluator.score_view_pred(
        runner=runner, view_pos=current_view.position, view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos, human_base_pos=human_pos,
        human_yaw=human_yaw, pose_type=pose_type,
        human_skeleton=skeleton, geodesic_distance=0.0,
    )

    # 记录当前视角 ray cast 失败数（若为 0 但场景无遮挡仍正常，仅当 >0 告警）
    occlusion_errors = count_occlusion_errors(current_view.pred_score)
    if occlusion_errors > 0:
        print(f"    ⚠ 当前视角 ray cast 失败关键点: {occlusion_errors} 个")

    # =====================================================================
    # 5. 采样候选位姿并计算 pred_score
    # =====================================================================
    candidates = sampler.sample(human_pos=human_pos, robot_pos=robot_start_pos,
                                runner=runner)
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
            runner=runner, view_pos=cand.position, view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos, human_base_pos=human_pos,
            human_yaw=human_yaw, pose_type=pose_type,
            human_skeleton=skeleton, geodesic_distance=cand.geodesic_distance,
        )

    # =====================================================================
    # 6. 【策略选择阶段】各类策略仅使用 Q_pred 等 pred_score 完成选择
    #    ⚠ 此阶段禁止使用候选点未来 RGB / depth / Q_true
    # =====================================================================
    selected_by_policy = {}
    for policy in policies:
        selected = policy.select(current_view, candidates)
        selected_by_policy[policy.name] = selected
        if selected is not current_view:
            selected.selected_by.append(policy.name)

    # =====================================================================
    # 7. 渲染 current_view 与各策略选中位姿，计算 true_score
    #    （同一视角被多个策略选中时只渲染一次）
    # =====================================================================
    print(f"  渲染 current_view 与选中位姿...")
    obs_current = _render_and_score(
        runner, true_evaluator, current_view, human_pos, human_yaw,
        pose_type, skeleton, images_dir, ep_str,
        save_image=False,   # 当前视角图像在下方单独命名保存
    )

    # current_view 图像 / true_score 单独处理（obs_current 可能为 None）
    try:
        if save_images and obs_current is not None and obs_current["rgb"] is not None:
            save_rgb_image(obs_current["rgb"],
                           os.path.join(images_dir, f"{ep_str}_current.png"))
    except Exception as e:
        print(f"    ⚠ 保存 current 图像失败: {e}")

    if not current_view.true_score:
        # render 失败时退化为离线 ray cast 评估
        current_view.true_score = true_evaluator.score_view_true(
            runner=runner, obs=None, view_pos=current_view.position,
            view_yaw=current_view.yaw, human_base_pos=human_pos,
            human_yaw=human_yaw, pose_type=pose_type, human_skeleton=skeleton,
        )

    # 对每个策略选中且非 current 的位姿渲染（去重）
    rendered_candidate_ids = set()
    for policy in policies:
        selected = selected_by_policy[policy.name]
        if selected is current_view or selected.candidate_id in rendered_candidate_ids:
            continue
        obs = _render_and_score(
            runner, true_evaluator, selected, human_pos, human_yaw,
            pose_type, skeleton, images_dir, ep_str, save_image=save_images,
        )
        if not selected.true_score:
            selected.true_score = true_evaluator.score_view_true(
                runner=runner, obs=obs, view_pos=selected.position,
                view_yaw=selected.yaw, human_base_pos=human_pos,
                human_yaw=human_yaw, pose_type=pose_type,
                human_skeleton=skeleton,
            )
        rendered_candidate_ids.add(selected.candidate_id)

    # =====================================================================
    # 8. 【评估阶段】Oracle 离线上界
    #    离线计算所有有效候选位姿 true_score（ray cast，无需渲染图像）
    #    ⚠ Oracle 仅在评估阶段运行，不参与任何策略选择
    # =====================================================================
    oracle_eval = None
    oracle_q_true = 0.0
    oracle_gap = 0.0
    if oracle_enabled and oracle_policy is not None:
        for cand in valid_candidates:
            if not cand.true_score:
                cand.true_score = true_evaluator.score_view_true(
                    runner=runner, obs=None, view_pos=cand.position,
                    view_yaw=cand.yaw, human_base_pos=human_pos,
                    human_yaw=human_yaw, pose_type=pose_type,
                    human_skeleton=skeleton,
                )
        oracle_view = oracle_policy.select(current_view, candidates)
        if oracle_view is not current_view:
            oracle_view.selected_by.append("Oracle-offline")

        oracle_q_true = float(oracle_view.true_score.get("Q_true", 0.0))
        ours_view = selected_by_policy["Ours"]
        oracle_gap = compute_oracle_gap(oracle_view, ours_view, current_view)

        oracle_eval = {
            "oracle_selected_candidate_id": oracle_view.candidate_id,
            "oracle_selected_is_current": (oracle_view is current_view),
            "oracle_Q_true": oracle_q_true,
            "ours_candidate_id": ours_view.candidate_id,
            "ours_Q_true": float(ours_view.true_score.get("Q_true", 0.0))
            if ours_view.true_score else 0.0,
            "oracle_gap": oracle_gap,
        }

    # =====================================================================
    # 9. 写 metrics.csv 行（每策略一行）
    # =====================================================================
    for policy in policies:
        selected = selected_by_policy[policy.name]
        is_current = (selected is current_view)

        pred_score = selected.pred_score if selected.pred_score else current_view.pred_score

        # true_score 兜底：若选中位姿缺失（渲染失败等），离线 ray cast 补充，
        # 而不是静默使用 current_view 的 true_score
        if not selected.true_score:
            if not is_current:
                try:
                    selected.true_score = true_evaluator.score_view_true(
                        runner=runner, obs=None, view_pos=selected.position,
                        view_yaw=selected.yaw, human_base_pos=human_pos,
                        human_yaw=human_yaw, pose_type=pose_type,
                        human_skeleton=skeleton,
                    )
                except Exception as e:
                    print(f"    ⚠ {policy.name} 选中位姿 true_score 兜底失败: {e}")
            if not selected.true_score:
                selected.true_score = dict(current_view.true_score) if current_view.true_score else {}
                print(f"    ⚠ {policy.name} 使用 current_view 的 true_score 兜底")
        true_score = selected.true_score

        gap = true_score.get("Q_true", 0.0) - pred_score.get("Q_pred", 0.0)
        (gain_ap_pred, gain_ap_true, gain_vis_pred, gain_vis_true,
         gain_occ_true) = compute_gains(
            selected.pred_score if selected.pred_score else current_view.pred_score,
            current_view.pred_score,
            true_score, current_view.true_score,
        )

        rel_angle = pred_score.get(
            "relative_view_angle",
            true_score.get("relative_view_angle_true", 0.0),
        )

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
            # ---- 预测指标（v3.0）----
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
            # ---- v4.0 遮挡感知预测指标 ----
            "S_action_occ_pred": pred_score.get("S_action_occ_pred", 0.0),
            "S_kp_occ_pred": pred_score.get("S_kp_occ_pred", 0.0),
            "occlusion_rate_pred": pred_score.get("occlusion_rate_pred", 0.0),
            "occluded_keypoint_count_pred": pred_score.get("occluded_keypoint_count_pred", 0),
            "torso_visibility_occ_pred": pred_score.get("torso_visibility_occ_pred", 0.0),
            "lower_body_visibility_occ_pred": pred_score.get("lower_body_visibility_occ_pred", 0.0),
            "head_visibility_occ_pred": pred_score.get("head_visibility_occ_pred", 0.0),
            "arms_visibility_occ_pred": pred_score.get("arms_visibility_occ_pred", 0.0),
            # ---- 真实指标（v3.0）----
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
            # ---- v4.0 遮挡感知真实指标 ----
            "S_action_occ_true": true_score.get("S_action_occ_true", 0.0),
            "S_kp_occ_true": true_score.get("S_kp_occ_true", 0.0),
            "occlusion_rate_true": true_score.get("occlusion_rate_true", 0.0),
            "occluded_keypoint_count_true": true_score.get("occluded_keypoint_count_true", 0),
            "torso_visibility_occ_true": true_score.get("torso_visibility_occ_true", 0.0),
            "lower_body_visibility_occ_true": true_score.get("lower_body_visibility_occ_true", 0.0),
            "head_visibility_occ_true": true_score.get("head_visibility_occ_true", 0.0),
            "arms_visibility_occ_true": true_score.get("arms_visibility_occ_true", 0.0),
            # ---- 差异 / 增益指标 ----
            "pred_true_gap": gap,
            "action_part_gain_pred": gain_ap_pred,
            "action_part_gain_true": gain_ap_true,
            "visibility_gain_pred": gain_vis_pred,
            "visibility_gain_true": gain_vis_true,
            "occlusion_gain_true": gain_occ_true,
            # ---- Oracle 指标 ----
            "oracle_Q_true": oracle_q_true,
            "oracle_gap": oracle_gap,
        }
        metrics_writer.write_metric_row(row)

    # =====================================================================
    # 10. 保存调试信息 + 写入 episode 摘要
    # =====================================================================
    debug_path = os.path.join(debug_dir, f"{ep_str}_candidates.json")
    episode_info = {
        "episode_id": episode_id, "scene_id": scene_id,
        "pose_type": pose_type, "human_yaw": human_yaw,
        "human_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "occlusion_enabled": config["occlusion"].get("enabled", True),
    }
    save_candidate_debug_json(candidates, debug_path,
                              episode_info=episode_info, oracle_eval=oracle_eval)

    metrics_writer.write_episode_summary({
        "episode_id": episode_id, "scene_id": scene_id,
        "status": "success", "pose_type": pose_type, "human_yaw": human_yaw,
        "human_base_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "valid_candidate_count": len(valid_candidates),
        "occlusion_ray_errors": occlusion_errors,
        "fixed_Q_pred": selected_by_policy["Fixed"].pred_score.get("Q_pred", 0.0),
        "ours_Q_pred": selected_by_policy["Ours"].pred_score.get("Q_pred", 0.0),
        "ours_selected_is_current": (selected_by_policy["Ours"] is current_view),
        "oracle_Q_true": oracle_q_true,
        "oracle_gap": oracle_gap,
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

    print(f"EA-AVS-MVP v4.0: 运行 {num_episodes} 个 episodes")
    print(f"  配置文件: {args.config}")
    print(f"  输出目录: {output_dir}")
    print(f"  场景文件: {config['habitat']['scene_path']}")
    print(f"  遮挡感知: {config['occlusion'].get('enabled', True)}")
    print(f"  支持姿态: {config['human']['pose_types']}")
    print()

    runner = HabitatRunner(config)
    sampler = CandidateSampler(config)
    pred_evaluator = PredictiveEvaluator(config)
    true_evaluator = TrueEvaluator(config)
    metrics_writer = MetricsWriter(output_dir)

    # 策略列表：基线 + 消融 + 主策略（Ours == FullOurs）
    policies = [
        FixedPolicy(),
        RandomPolicy(seed=config["project"]["seed"]),
        NearestPolicy(),
        # 模块化消融
        VisibilityOnlyPolicy(),
        ActionPartOnlyPolicy(),
        OrientationOnlyPolicy(),
        OcclusionOnlyPolicy(),
        ActionOrientationPolicy(config),
        # 主策略（v4.0 完整版）
        OursPolicy(),
    ]

    oracle_enabled = config.get("oracle", {}).get("enabled", True)
    oracle_policy = OraclePolicy() if oracle_enabled else None

    success_count = 0
    failed_count = 0

    for episode_id in range(num_episodes):
        print(f"Episode {episode_id + 1}/{num_episodes}:")
        try:
            success = run_one_episode(
                episode_id, config, runner, sampler,
                pred_evaluator, true_evaluator, policies, oracle_policy,
                output_dir, metrics_writer, oracle_enabled,
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
    print(f"  metrics.csv    —— 含遮挡感知与 Oracle 指标")
    print(f"  episodes.jsonl —— Episode 摘要")
    print(f"  images/        —— 渲染图像")
    print(f"  debug/         —— 候选点调试（含遮挡信息与 Oracle 评估）")
    print("=" * 60)


if __name__ == "__main__":
    main()