#!/usr/bin/env python
"""
EA-AVS-MVP v5.0 主入口 —— run_mvp50_humanoid.py
=================================================

中文名称：真实 Humanoid + RGB-D 的 one-shot 主动观察位姿选择

v5.0 = v4.0 主动观察位姿选择框架 + Habitat Humanoid + 真实人体 RGB-D 渲染
        + Humanoid GT 状态接口

流程（保持 v2.0 建立的 pred/true 信息边界）：
    Episode 开始
      → 采样 human base position
      → 创建/重置 Humanoid 并放置
      → 设置 pose（默认 standing）
      → 读取 Humanoid GT skeleton（link transforms，非手工坐标）
      → 生成 robot current + candidate observation poses
      → 使用 v4.0 pred evaluator（GT Humanoid skeleton + map geometry）评分
      → 在线策略选择（Fixed/Random/Nearest/消融/Ours，仅用 Q_pred）
      → 【Evaluation Phase】渲染 current + 各策略选中位姿（含 Humanoid 的 RGB-D）
      → 计算几何/depth true metrics
      → Oracle 离线上界（同口径 depth Q_true）
      → 保存 Humanoid GT state + RGB-D + metrics/debug JSON

禁止：
    - 候选点未来 RGB/depth 进入在线策略选择
    - 用旧 action_pose_library 手工坐标作为主 GT（仅兜底）
    - 用 standing 静默替代缺失的 research 动作
"""

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.humanoid_assets import (
    resolve_humanoid_assets, validate_humanoid_assets,
)
from ea_avs_v5.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v5.action_pose_library import POSE_SKELETONS
from ea_avs_v5.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v5.predictive_evaluator import PredictiveEvaluator
from ea_avs_v5.true_evaluator import TrueEvaluator
from ea_avs_v5.policies import FixedPolicy, RandomPolicy, NearestPolicy, OursPolicy
from ea_avs_v5.ablation_policies import (
    VisibilityOnlyPolicy, ActionPartOnlyPolicy, OrientationOnlyPolicy,
    OcclusionOnlyPolicy, ActionOrientationPolicy,
)
from ea_avs_v5.oracle_policy import OraclePolicy, compute_oracle_gap
from ea_avs_v5.metrics import MetricsWriter
from ea_avs_v5.visualization import save_rgb_image, save_candidate_debug_json
from ea_avs_v5.geometry import compute_look_at_yaw, compute_camera_intrinsics


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EA-AVS-MVP v5.0: 真实 Humanoid + RGB-D 主动视角选择")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖配置文件中的 episode 数量")
    parser.add_argument("--output-dir", type=str, default="outputs/mvp50_run",
                        help="输出目录")
    return parser.parse_args()


# =============================================================================
# 辅助函数
# =============================================================================

def set_random_seed(seed: int):
    np.random.seed(seed)
    import random
    random.seed(seed)


def euclidean_distance(a, b) -> float:
    return float(np.linalg.norm(a - b))


def sample_valid_human_position(runner, config) -> np.ndarray:
    max_tries = config["episode"]["max_sampling_tries"]
    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if runner.is_navigable(pt):
            return pt
    raise RuntimeError(f"无法采样人体位置（{max_tries} 次尝试后）")


def sample_human_yaw(config) -> float:
    human_cfg = config["human"] if "human" in config else None
    if human_cfg and config["episode"].get("randomize_human_yaw", True):
        yaw_deg = float(np.random.choice(human_cfg["yaw_candidates_deg"]))
        return np.deg2rad(yaw_deg)
    return 0.0


def sample_robot_start_position_around_human(runner, human_pos, config):
    ep_cfg = config["episode"]
    min_dist = ep_cfg.get("min_robot_human_distance", 1.5)
    max_dist = ep_cfg.get("max_robot_human_distance", 4.0)
    max_tries = ep_cfg["max_sampling_tries"]
    for _ in range(max_tries):
        pt = runner.sample_navigable_point()
        if not runner.is_navigable(pt):
            continue
        if not (min_dist <= euclidean_distance(pt, human_pos) <= max_dist):
            continue
        if runner.geodesic_distance(pt, human_pos) == float("inf"):
            continue
        return pt
    raise RuntimeError(f"无法采样机器人起始位置（{max_tries} 次尝试后）")


def compute_gains(pred_sel, pred_cur, true_sel, true_cur):
    ap_pred_sel = pred_sel.get("S_action_part_pred", 0.0) if pred_sel else 0.0
    ap_pred_cur = pred_cur.get("S_action_part_pred", 0.0) if pred_cur else 0.0
    gain_ap_pred = ap_pred_sel - ap_pred_cur
    ap_true_sel = true_sel.get("S_action_part_true", 0.0) if true_sel else 0.0
    ap_true_cur = true_cur.get("S_action_part_true", 0.0) if true_cur else 0.0
    gain_ap_true = ap_true_sel - ap_true_cur
    skp_pred_sel = pred_sel.get("S_kp_pred", 0.0) if pred_sel else 0.0
    skp_pred_cur = pred_cur.get("S_kp_pred", 0.0) if pred_cur else 0.0
    gain_vis_pred = skp_pred_sel - skp_pred_cur
    skp_true_sel = true_sel.get("S_kp_true", 0.0) if true_sel else 0.0
    skp_true_cur = true_cur.get("S_kp_true", 0.0) if true_cur else 0.0
    gain_vis_true = skp_true_sel - skp_true_cur
    occ_true_sel = true_sel.get("S_action_occ_true", 0.0) if true_sel else 0.0
    occ_true_cur = true_cur.get("S_action_occ_true", 0.0) if true_cur else 0.0
    gain_occ_true = occ_true_sel - occ_true_cur
    return gain_ap_pred, gain_ap_true, gain_vis_pred, gain_vis_true, gain_occ_true


def ensure_true_score(
    runner, true_evaluator, view, human_pos, human_yaw, pose_type,
    skeleton, images_dir, ep_str, save_image, save_depth,
):
    """渲染一个视角并计算 depth 口径 true_score（含缓存）。"""
    if (view.true_score
            and view.true_score.get("true_evaluation_source") == "depth"):
        return
    try:
        obs = runner.render_at(view.position, view.yaw)
        if save_image and obs["rgb"] is not None:
            fname = f"{ep_str}_view_{view.candidate_id}.png"
            save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        if save_depth and obs["depth"] is not None:
            os.makedirs(images_dir, exist_ok=True)
            import numpy as _np
            _np.save(os.path.join(images_dir, f"{ep_str}_view_{view.candidate_id}_depth.npy"),
                     _np.asarray(obs["depth"], dtype=_np.float32))
        view.true_score = true_evaluator.score_view_true(
            runner=runner, obs=obs, view_pos=view.position,
            view_yaw=view.yaw, human_base_pos=human_pos, human_yaw=human_yaw,
            pose_type=pose_type, human_skeleton=skeleton,
        )
    except Exception as e:
        print(f"    ⚠ 渲染失败 (candidate {view.candidate_id}): {e}")
        try:
            view.true_score = true_evaluator.score_view_true(
                runner=runner, obs=None, view_pos=view.position,
                view_yaw=view.yaw, human_base_pos=human_pos,
                human_yaw=human_yaw, pose_type=pose_type,
                human_skeleton=skeleton,
            )
        except Exception as e2:
            print(f"    ⚠ true_score 兜底失败: {e2}")
            view.true_score = {}


# =============================================================================
# 单个 Episode
# =============================================================================

def run_one_episode(
    episode_id, config, runner, sampler, pred_evaluator, true_evaluator,
    policies, oracle_policy, humanoid_manager, output_dir, metrics_writer,
    oracle_enabled,
) -> bool:
    scene_id = os.path.splitext(os.path.basename(config["habitat"]["scene_path"]))[0]
    ep_str = f"ep_{episode_id:03d}"
    images_dir = os.path.join(output_dir, "images")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    save_images = config["output"].get("save_images", True)
    save_depth = config["output"].get("save_depth", True)

    # ---- 1. 采样 human base position + yaw ----
    human_pos = sample_valid_human_position(runner, config)
    human_yaw = sample_human_yaw(config)

    # ---- 2. 放置 / 重置 Humanoid（standing，episode 生命周期内持续存在）----
    humanoid_manager.set_base_pose(human_pos, human_yaw)
    try:
        humanoid_manager.set_pose("standing")
    except Exception as e:
        metrics_writer.write_episode_summary({
            "episode_id": episode_id, "scene_id": scene_id, "status": "failed",
            "reason": f"humanoid_pose_failed: {e}",
        })
        return False
    hstate = humanoid_manager.get_state()

    # ---- 3. 读取 Humanoid GT skeleton（link transforms 优先）----
    legacy_skeleton = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in POSE_SKELETONS["standing"].items()
    }
    gt_result = get_humanoid_gt_skeleton(humanoid_manager, legacy_skeleton)
    skeleton = gt_result["skeleton"]
    gt_skeleton_source = gt_result["source"]

    # ---- 4. 生成 robot current pose ----
    robot_start_pos = sample_robot_start_position_around_human(
        runner, human_pos, config)
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)

    # ---- 5. current_view + v4.0 pred evaluator（含 ray casting，无 render）----
    current_view = CandidateView(
        candidate_id=-1, position=robot_start_pos, yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
    )
    current_view.pred_score = pred_evaluator.score_view_pred(
        runner=runner, view_pos=current_view.position, view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos, human_base_pos=human_pos,
        human_yaw=human_yaw, pose_type="standing",
        human_skeleton=skeleton, geodesic_distance=0.0,
    )

    # ---- 6. 生成候选位姿 + pred_score ----
    candidates = sampler.sample(human_pos=human_pos, robot_pos=robot_start_pos,
                                runner=runner)
    valid_candidates = [c for c in candidates if c.is_valid]
    if len(valid_candidates) == 0:
        metrics_writer.write_episode_summary({
            "episode_id": episode_id, "scene_id": scene_id, "status": "failed",
            "reason": "no_valid_candidates",
            "humanoid": hstate.to_dict(),
        })
        return False

    for cand in valid_candidates:
        cand.pred_score = pred_evaluator.score_view_pred(
            runner=runner, view_pos=cand.position, view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos, human_base_pos=human_pos,
            human_yaw=human_yaw, pose_type="standing",
            human_skeleton=skeleton, geodesic_distance=cand.geodesic_distance,
        )

    # ---- 7. 在线策略选择（仅使用 Q_pred，禁止未来 RGB/depth/Q_true）----
    selected_by_policy = {}
    for policy in policies:
        selected = policy.select(current_view, candidates)
        selected_by_policy[policy.name] = selected
        if selected is not current_view:
            selected.selected_by.append(policy.name)

    # ---- 8. 【Evaluation Phase】渲染 current + 策略选中位姿（含 Humanoid RGB-D）----
    humanoid_render_success = True
    ensure_true_score(
        runner, true_evaluator, current_view, human_pos, human_yaw, "standing",
        skeleton, images_dir, ep_str, save_images, save_depth,
    )
    for policy in policies:
        selected = selected_by_policy[policy.name]
        if selected is current_view:
            continue
        ensure_true_score(
            runner, true_evaluator, selected, human_pos, human_yaw, "standing",
            skeleton, images_dir, ep_str, save_images, save_depth,
        )
    # Oracle 补齐其余候选（depth 口径）
    if oracle_enabled and oracle_policy is not None:
        for cand in valid_candidates:
            ensure_true_score(
                runner, true_evaluator, cand, human_pos, human_yaw, "standing",
                skeleton, images_dir, ep_str, False, False,
            )

    # ---- 9. Oracle 离线上界（同口径 depth Q_true）----
    oracle_q_true, oracle_gap, oracle_valid, oracle_sel_id = 0.0, 0.0, 0, None
    oracle_eval = None
    if oracle_enabled and oracle_policy is not None:
        oracle_view, oracle_valid = oracle_policy.select(current_view, candidates)
        if oracle_view is not None:
            oracle_q_true = float(oracle_view.true_score.get("Q_true", 0.0))
            oracle_sel_id = oracle_view.candidate_id
            ours_view = selected_by_policy["Ours"]
            oracle_gap = compute_oracle_gap(oracle_view, ours_view, current_view)
            oracle_eval = {
                "oracle_selected_candidate_id": oracle_view.candidate_id,
                "oracle_Q_true": oracle_q_true,
                "oracle_valid_true_candidate_count": oracle_valid,
                "oracle_gap": oracle_gap,
                "note": "offline evaluation (upper bound, not deployable)",
            }

    # ---- 10. metrics.csv 行 ----
    for policy in policies:
        selected = selected_by_policy[policy.name]
        is_current = (selected is current_view)
        pred_score = selected.pred_score if selected.pred_score else current_view.pred_score
        true_score = selected.true_score if selected.true_score else current_view.true_score
        if not true_score:
            true_score = {}
        gap = true_score.get("Q_true", 0.0) - pred_score.get("Q_pred", 0.0)
        (g_ap_pred, g_ap_true, g_vis_pred, g_vis_true, g_occ_true) = compute_gains(
            selected.pred_score if selected.pred_score else current_view.pred_score,
            current_view.pred_score, true_score, current_view.true_score,
        )
        rel_angle = pred_score.get("relative_view_angle",
                     true_score.get("relative_view_angle_true", 0.0))

        row = {
            "episode_id": episode_id, "scene_id": scene_id,
            "policy": policy.name, "status": "success",
            "num_candidates": len(valid_candidates),
            "selected_is_current": int(is_current),
            "pose_type": "standing", "human_yaw": human_yaw,
            "relative_view_angle": rel_angle,
            "human_x": human_pos[0], "human_y": human_pos[1], "human_z": human_pos[2],
            "robot_start_x": robot_start_pos[0], "robot_start_y": robot_start_pos[1],
            "robot_start_z": robot_start_pos[2],
            "selected_x": selected.position[0], "selected_y": selected.position[1],
            "selected_z": selected.position[2], "selected_yaw": selected.yaw,
            "geodesic_distance": selected.geodesic_distance,
            # 预测指标（v4.0）
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
            "S_action_occ_pred": pred_score.get("S_action_occ_pred", 0.0),
            "S_kp_occ_pred": pred_score.get("S_kp_occ_pred", 0.0),
            "occlusion_rate_pred": pred_score.get("occlusion_rate_pred", 0.0),
            "occluded_keypoint_count_pred": pred_score.get("occluded_keypoint_count_pred", 0),
            "occlusion_valid_keypoint_count_pred": pred_score.get(
                "occlusion_valid_keypoint_count_pred", 0),
            "raycast_error_count_pred": pred_score.get("raycast_error_count_pred", 0),
            "raycast_error_rate_pred": pred_score.get("raycast_error_rate_pred", 0.0),
            "is_occlusion_valid_pred": int(pred_score.get("is_occlusion_valid_pred", True)),
            "torso_visibility_occ_pred": pred_score.get("torso_visibility_occ_pred", 0.0),
            "lower_body_visibility_occ_pred": pred_score.get("lower_body_visibility_occ_pred", 0.0),
            "head_visibility_occ_pred": pred_score.get("head_visibility_occ_pred", 0.0),
            "arms_visibility_occ_pred": pred_score.get("arms_visibility_occ_pred", 0.0),
            # 真实指标（v4.0）
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
            "S_action_occ_true": true_score.get("S_action_occ_true", 0.0),
            "S_kp_occ_true": true_score.get("S_kp_occ_true", 0.0),
            "occlusion_rate_true": true_score.get("occlusion_rate_true", 0.0),
            "occluded_keypoint_count_true": true_score.get("occluded_keypoint_count_true", 0),
            "occlusion_valid_keypoint_count_true": true_score.get(
                "occlusion_valid_keypoint_count_true", 0),
            "raycast_error_count_true": true_score.get("raycast_error_count_true", 0),
            "raycast_error_rate_true": true_score.get("raycast_error_rate_true", 0.0),
            "depth_valid_keypoint_count_true": true_score.get(
                "depth_valid_keypoint_count_true", 0),
            "depth_invalid_keypoint_count_true": true_score.get(
                "depth_invalid_keypoint_count_true", 0),
            "true_evaluation_source": true_score.get("true_evaluation_source", ""),
            "torso_visibility_occ_true": true_score.get("torso_visibility_occ_true", 0.0),
            "lower_body_visibility_occ_true": true_score.get("lower_body_visibility_occ_true", 0.0),
            "head_visibility_occ_true": true_score.get("head_visibility_occ_true", 0.0),
            "arms_visibility_occ_true": true_score.get("arms_visibility_occ_true", 0.0),
            # 差异/增益
            "pred_true_gap": gap,
            "action_part_gain_pred": g_ap_pred,
            "action_part_gain_true": g_ap_true,
            "visibility_gain_pred": g_vis_pred,
            "visibility_gain_true": g_vis_true,
            "occlusion_gain_true": g_occ_true,
            # Oracle
            "oracle_Q_true": oracle_q_true,
            "oracle_gap": oracle_gap,
            "oracle_valid_true_candidate_count": oracle_valid,
            "oracle_selected_candidate_id": oracle_sel_id,
            # v5.0 Humanoid 状态
            "humanoid_enabled": int(config["humanoid"].get("enabled", True)),
            "humanoid_avatar_name": config["humanoid"]["avatar_name"],
            "humanoid_pose_name": hstate.pose_name,
            "humanoid_motion_frame": hstate.motion_frame,
            "humanoid_base_x": hstate.base_position[0],
            "humanoid_base_y": hstate.base_position[1],
            "humanoid_base_z": hstate.base_position[2],
            "humanoid_yaw": hstate.base_yaw,
            "humanoid_gt_skeleton_source": gt_skeleton_source,
            "humanoid_render_success": int(humanoid_render_success),
            "humanoid_self_occlusion_supported_pred": 0,
        }
        metrics_writer.write_metric_row(row)

    # ---- 11. debug JSON（含 Humanoid GT 状态）----
    debug_path = os.path.join(debug_dir, f"{ep_str}_candidates.json")
    camera_intrinsics = compute_camera_intrinsics(
        config["camera"]["width"], config["camera"]["height"],
        config["camera"]["hfov_deg"])
    episode_info = {
        "episode_id": episode_id, "scene_id": scene_id,
        "pose_type": "standing", "human_yaw": human_yaw,
        "human_pos": human_pos.tolist(),
        "robot_start_pos": robot_start_pos.tolist(),
        "humanoid": {
            **hstate.to_dict(),
            "avatar_name": config["humanoid"]["avatar_name"],
            "asset_paths": {
                "glb": resolve_humanoid_assets(config).glb_path,
                "urdf": resolve_humanoid_assets(config).urdf_path,
            },
            "gt_skeleton_source": gt_skeleton_source,
        },
        "camera": camera_intrinsics,
    }
    save_candidate_debug_json(candidates, debug_path,
                              episode_info=episode_info, oracle_eval=oracle_eval)

    metrics_writer.write_episode_summary({
        "episode_id": episode_id, "scene_id": scene_id, "status": "success",
        "pose_type": "standing", "human_yaw": human_yaw,
        "human_pos": human_pos.tolist(),
        "humanoid": hstate.to_dict(),
        "valid_candidate_count": len(valid_candidates),
        "humanoid_gt_skeleton_source": gt_skeleton_source,
        "humanoid_render_success": humanoid_render_success,
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

    print(f"EA-AVS-MVP v5.0: 运行 {num_episodes} 个 episodes")
    print(f"  输出目录: {output_dir}")
    print(f"  场景: {config['habitat']['scene_path']}")
    print(f"  Humanoid: {config['humanoid']['avatar_name']} ("
          f"motion_mode={config['humanoid'].get('motion_mode','official')})")
    print()

    # 资源校验（缺失即报错，不静默退化）
    validate_humanoid_assets(resolve_humanoid_assets(config))

    runner = HabitatRunner(config)
    humanoid_manager = HumanoidManager(runner, config)
    humanoid_manager.load()
    runner.attach_humanoid_manager(humanoid_manager)

    sampler = CandidateSampler(config)
    pred_evaluator = PredictiveEvaluator(config)
    true_evaluator = TrueEvaluator(config)
    metrics_writer = MetricsWriter(output_dir)

    policies = [
        FixedPolicy(),
        RandomPolicy(seed=config["project"]["seed"]),
        NearestPolicy(),
        VisibilityOnlyPolicy(),
        ActionPartOnlyPolicy(),
        OrientationOnlyPolicy(),
        OcclusionOnlyPolicy(),
        ActionOrientationPolicy(config),
        OursPolicy(),
    ]
    oracle_enabled = config.get("oracle", {}).get("enabled", True)
    oracle_policy = OraclePolicy() if oracle_enabled else None

    success_count = 0
    failed_count = 0
    for episode_id in range(num_episodes):
        print(f"Episode {episode_id + 1}/{num_episodes}:")
        try:
            ok = run_one_episode(
                episode_id, config, runner, sampler, pred_evaluator,
                true_evaluator, policies, oracle_policy,
                humanoid_manager, output_dir, metrics_writer, oracle_enabled,
            )
            if ok:
                success_count += 1
                print(f"  -> 成功 ✅")
            else:
                failed_count += 1
                print(f"  -> 失败 ❌")
        except Exception as e:
            failed_count += 1
            print(f"  -> 错误: {e} ❌")
            traceback.print_exc()
            metrics_writer.write_episode_summary({
                "episode_id": episode_id,
                "scene_id": os.path.splitext(
                    os.path.basename(config["habitat"]["scene_path"]))[0],
                "status": "failed", "reason": str(e),
            })
        print()

    metrics_writer.close()
    humanoid_manager.close()
    runner.close()

    print("=" * 60)
    print(f"实验完成: {success_count} 成功, {failed_count} 失败")
    print("输出: metrics.csv / episodes.jsonl / images(RGB+Depth) / debug")
    print("=" * 60)


if __name__ == "__main__":
    main()