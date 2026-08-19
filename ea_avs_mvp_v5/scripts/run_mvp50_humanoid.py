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
from ea_avs_v5.humanoid_validation import validate_gt_skeleton
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
    """从 humanoid 配置采样 yaw（8 方向随机化）。

    v5.0：yaw 候选来自 humanoid.yaw_candidates_deg（v5 配置无 human: 段）。
    """
    hcfg = config.get("humanoid", {})
    candidates = hcfg.get(
        "yaw_candidates_deg", [0, 45, 90, 135, 180, -135, -90, -45])
    if config["episode"].get("randomize_human_yaw", True):
        yaw_deg = float(np.random.choice(candidates))
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
    skeleton, keypoint_meta, images_dir, ep_str, save_image, save_depth,
    humanoid_object_ids, semantic_enabled,
):
    """渲染一个视角并计算 depth 口径 true_score（含缓存），返回 obs。"""
    if (view.true_score
            and view.true_score.get("true_evaluation_source") == "depth"):
        return None
    obs = None
    try:
        obs = runner.render_at(view.position, view.yaw)
        if save_image and obs["rgb"] is not None:
            fname = f"{ep_str}_view_{view.candidate_id}.png"
            save_rgb_image(obs["rgb"], os.path.join(images_dir, fname))
        if save_depth and obs["depth"] is not None:
            os.makedirs(images_dir, exist_ok=True)
            _np = np
            _np.save(os.path.join(
                images_dir, f"{ep_str}_view_{view.candidate_id}_depth.npy"),
                _np.asarray(obs["depth"], dtype=_np.float32))
        view.true_score = true_evaluator.score_view_true(
            runner=runner, obs=obs, view_pos=view.position,
            view_yaw=view.yaw, human_base_pos=human_pos, human_yaw=human_yaw,
            pose_type=pose_type, human_skeleton=skeleton,
            humanoid_object_ids=humanoid_object_ids,
            keypoint_meta=keypoint_meta,
        )
    except Exception as e:
        print(f"    ⚠ 渲染失败 (candidate {view.candidate_id}): {e}")
        try:
            view.true_score = true_evaluator.score_view_true(
                runner=runner, obs=None, view_pos=view.position,
                view_yaw=view.yaw, human_base_pos=human_pos,
                human_yaw=human_yaw, pose_type=pose_type,
                human_skeleton=skeleton, humanoid_object_ids=humanoid_object_ids,
                keypoint_meta=keypoint_meta,
            )
        except Exception as e2:
            print(f"    ⚠ true_score 兜底失败: {e2}")
            view.true_score = {}
    return obs


def compute_humanoid_render_stats(
    config, obs, view_pos, view_yaw, human_skeleton, humanoid_semantic_ids,
    semantic_assignment_ok=False, semantic_assignment_count=0,
):
    """优先级感知的 Humanoid 渲染统计（semantic -> GT-depth proxy -> unavailable）。

    ⚠ 仅用于验收/metrics，绝不进入 Q_pred/策略选择。
    """
    from ea_avs_v5.humanoid_validation import compute_humanoid_render_stats as _stats
    return _stats(obs, config, view_pos, view_yaw, human_skeleton,
                  humanoid_semantic_ids, semantic_assignment_ok,
                  semantic_assignment_count)


# ---- render stats 提取辅助（兼容 semantic 与 gt_depth_proxy 两种口径）----

def _rs_visible(rs):
    """Humanoid 是否可见（semantic 或 proxy 口径统一）。"""
    if rs.get("humanoid_validation_source") == "semantic":
        return 1 if rs.get("humanoid_semantic_visible") else 0
    return 1 if rs.get("humanoid_proxy_visible") else 0


def _rs_pixels(rs):
    """Humanoid 像素数（semantic 用 pixel_count；proxy 用 depth_match_pixel_count）。"""
    if rs.get("humanoid_validation_source") == "semantic":
        return int(rs.get("humanoid_semantic_pixel_count", 0))
    return int(rs.get("humanoid_proxy_pixel_count", 0))


def _rs_pixel_ratio(rs, config):
    """Humanoid 像素占比。"""
    h = config["camera"]["height"]
    w = config["camera"]["width"]
    if rs.get("humanoid_validation_source") == "semantic":
        return float(rs.get("humanoid_semantic_pixel_ratio", 0.0))
    cnt = _rs_pixels(rs)
    return float(cnt / (h * w)) if h * w > 0 else 0.0


def _rs_match_ratio(rs):
    """匹配率（semantic 下用 depth_valid_ratio 近似；proxy 用 depth_match_ratio）。"""
    if rs.get("humanoid_validation_source") == "semantic":
        return float(rs.get("humanoid_depth_valid_ratio", 0.0))
    return float(rs.get("humanoid_proxy_match_ratio", 0.0))


def _rs_bbox(rs, idx):
    """bbox 坐标（semantic 的 humanoid_semantic_bbox 或 proxy 的 humanoid_proxy_bbox）。"""
    bbox = None
    if rs.get("humanoid_validation_source") == "semantic":
        bbox = rs.get("humanoid_semantic_bbox")
    else:
        bbox = rs.get("humanoid_proxy_bbox")
    if bbox and idx < len(bbox):
        return bbox[idx]
    return None


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
    requested_yaw = sample_human_yaw(config)

    # ---- 2. 放置 / 重置 Humanoid（standing，episode 生命周期内持续存在）----
    humanoid_manager.set_base_pose(human_pos, requested_yaw)
    try:
        humanoid_manager.set_pose("standing")
    except Exception as e:
        metrics_writer.write_episode_summary({
            "episode_id": episode_id, "scene_id": scene_id, "status": "failed",
            "reason": f"humanoid_pose_failed: {e}",
        })
        return False
    hstate = humanoid_manager.get_state()
    human_yaw = hstate.base_yaw          # 实际设置的 humanoid yaw
    actual_yaw = hstate.actual_base_yaw  # 从 agent 读回的朝向

    # ---- 3. 读取 Humanoid GT skeleton（link transforms 优先，严格 15/15）----
    strict_gt = config["humanoid"].get("strict_gt_skeleton", True)
    legacy_skeleton = None
    if not strict_gt:
        legacy_skeleton = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in POSE_SKELETONS["standing"].items()
        }
    gt_result = get_humanoid_gt_skeleton(
        humanoid_manager, legacy_skeleton,
        human_base_pos=human_pos, human_yaw=human_yaw, strict=strict_gt,
    )
    skeleton = gt_result["skeleton"]
    keypoint_meta = gt_result.get("keypoint_meta", {})
    gt_validation = validate_gt_skeleton(gt_result, strict=strict_gt)
    if not gt_validation["valid"]:
        # 严格模式已抛错；此处为 debug 模式的显式失败处理
        metrics_writer.write_episode_summary({
            "episode_id": episode_id, "scene_id": scene_id, "status": "failed",
            "reason": f"invalid_humanoid_gt_skeleton: {gt_validation['reason']}",
        })
        return False
    gt_skeleton_source = gt_result["source"]

    # ---- 3b. Humanoid object ids（用于 environment/self occlusion 区分）----
    humanoid_object_ids = humanoid_manager.get_humanoid_object_ids()

    # ---- 3c. 显式给 Humanoid link/visual node 设置 semantic id（验证用）----
    semantic_enabled = config["humanoid"].get("semantic_enabled", True)
    semantic_id = config["humanoid"].get("semantic_id", 100)
    semantic_assignment_count = 0
    if semantic_enabled:
        semantic_assignment_count = humanoid_manager.assign_semantic_id_to_links(
            semantic_id)
    semantic_assignment_ok = bool(semantic_enabled and semantic_assignment_count > 0)
    humanoid_semantic_ids = [semantic_id] if semantic_enabled else []

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
        humanoid_object_ids=humanoid_object_ids,
        keypoint_meta=keypoint_meta,
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
            humanoid_object_ids=humanoid_object_ids,
            keypoint_meta=keypoint_meta,
        )

    # ---- 7. 在线策略选择（仅使用 Q_pred，禁止未来 RGB/depth/Q_true）----
    selected_by_policy = {}
    ours_sel_stats = None
    # Ours 过滤 invalid-occlusion 候选，需读取其选后统计
    for policy in policies:
        selected = policy.select(current_view, candidates)
        if policy.name == "Ours":
            ours_sel_stats = getattr(policy, "last_selection_stats", None)
        selected_by_policy[policy.name] = selected
        if selected is not current_view:
            selected.selected_by.append(policy.name)
    ours_excluded_invalid = (ours_sel_stats or {}).get(
        "excluded_invalid_occ_count", 0)
    ours_fell_back = bool((ours_sel_stats or {}).get(
        "fell_back_to_current", False))

    # ---- 8. 【Evaluation Phase】渲染 current + 策略选中位姿（含 Humanoid RGB-D）
    #        计算当前视角 + 各策略选中视角的 Humanoid 渲染统计（不同视角分开）。
    #        用 render_cache 按 candidate_id 缓存 obs，避免同一候选被多策略重复渲染。
    # ⚠ 在线策略选择已在第 7 步完成；此处渲染不构成信息泄漏。
    # =====================================================================
    render_cache = {}  # candidate_id -> obs

    def _ensure_obs(view):
        """渲染并缓存某一视图的 obs + true_score（depth 口径），返回 obs。"""
        if view.candidate_id in render_cache:
            return render_cache[view.candidate_id]
        obs = ensure_true_score(
            runner, true_evaluator, view, human_pos, human_yaw, "standing",
            skeleton, keypoint_meta, images_dir, ep_str,
            False, False, humanoid_object_ids, semantic_enabled,
        )
        render_cache[view.candidate_id] = obs
        return obs

    # 8.1 渲染 current_view（保存 current.png + depth）
    obs_curr = _ensure_obs(current_view)
    if save_images and obs_curr is not None and obs_curr["rgb"] is not None:
        save_rgb_image(obs_curr["rgb"], os.path.join(images_dir, f"{ep_str}_current.png"))
    if save_depth and obs_curr is not None and obs_curr["depth"] is not None:
        np.save(os.path.join(images_dir, f"{ep_str}_current_depth.npy"),
                np.asarray(obs_curr["depth"], dtype=np.float32))

    # 8.2 current 渲染统计
    rs_curr = compute_humanoid_render_stats(
        config, obs_curr, current_view.position, current_view.yaw,
        skeleton, humanoid_semantic_ids,
        semantic_assignment_ok, semantic_assignment_count,
    )

    # 8.3 各策略选中位姿 + Oracle 候选：渲染并缓存 obs
    for policy in policies:
        selected = selected_by_policy[policy.name]
        if selected is current_view:
            continue
        _ensure_obs(selected)
    if oracle_enabled and oracle_policy is not None:
        for cand in valid_candidates:
            _ensure_obs(cand)

    # 8.4 每个策略选中位姿的渲染统计（selected_*）
    #     （策略选择已完成，安全使用 selected RGB/depth）
    selected_render_stats = {}
    for policy in policies:
        sel = selected_by_policy[policy.name]
        obs_sel = render_cache.get(sel.candidate_id)
        rs_sel = compute_humanoid_render_stats(
            config, obs_sel, sel.position, sel.yaw, skeleton,
            humanoid_semantic_ids,
            semantic_assignment_ok, semantic_assignment_count,
        )
        selected_render_stats[policy.name] = rs_sel

    # ---- 9. Oracle 离线上界（depth 口径 + depth_coverage 门槛 + 同口径 gap）----
    oracle_q_true, oracle_gap = None, None
    oracle_valid, oracle_sel_id = 0, None
    oracle_depth_eligible, oracle_excluded = 0, 0
    oracle_available = 0
    oracle_gap_valid = 0
    oracle_gap_reason = None
    min_dc = config.get("oracle", {}).get("min_depth_coverage", 0.8)
    oracle_eval = None
    if oracle_enabled and oracle_policy is not None:
        oracle_view, odetail = oracle_policy.select(current_view, candidates)
        oracle_valid = odetail["valid_true_count"]
        oracle_depth_eligible = odetail["depth_eligible_count"]
        oracle_excluded = odetail["excluded_low_depth_coverage_count"]
        if oracle_view is not None:
            oracle_available = 1
            oracle_q_true = float(oracle_view.true_score.get("Q_true", 0.0))
            oracle_sel_id = oracle_view.candidate_id
            ours_view = selected_by_policy["Ours"]
            gap_info = compute_oracle_gap(
                oracle_view, ours_view, current_view, min_dc)
            oracle_gap = gap_info["oracle_gap"]
            oracle_gap_valid = int(gap_info["oracle_gap_valid"])
            oracle_gap_reason = gap_info["oracle_gap_reason"]
            oracle_eval = {
                "oracle_selected_candidate_id": oracle_view.candidate_id,
                "oracle_available": oracle_available,
                "oracle_Q_true": oracle_q_true,
                "oracle_valid_true_candidate_count": oracle_valid,
                "oracle_depth_eligible_candidate_count": oracle_depth_eligible,
                "oracle_excluded_low_depth_coverage_count": oracle_excluded,
                "oracle_gap": oracle_gap,
                "oracle_gap_valid": oracle_gap_valid,
                "oracle_gap_reason": oracle_gap_reason,
                "note": "offline evaluation (upper bound, not deployable)",
            }
        else:
            oracle_eval = {
                "oracle_available": 0,
                "oracle_valid_true_candidate_count": oracle_valid,
                "oracle_depth_eligible_candidate_count": 0,
                "oracle_excluded_low_depth_coverage_count": oracle_excluded,
                "oracle_aborted": "no_depth_coverage_eligible_candidates",
                "oracle_Q_true": None,
                "oracle_gap": None,
                "oracle_gap_valid": 0,
                "oracle_gap_reason": "oracle_unavailable",
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
            "invalid_occlusion_keypoint_count_pred": pred_score.get(
                "invalid_occlusion_keypoint_count_pred", 0),
            "invalid_occlusion_keypoint_rate_pred": pred_score.get(
                "invalid_occlusion_keypoint_rate_pred", 0.0),
            "target_surface_keypoint_count_pred": pred_score.get(
                "target_surface_keypoint_count_pred", 0),
            "environment_occluded_keypoint_count_pred": pred_score.get(
                "environment_occluded_keypoint_count_pred", 0),
            "self_occluded_keypoint_count_pred": pred_score.get(
                "self_occluded_keypoint_count_pred", 0),
            "unknown_occlusion_keypoint_count_pred": pred_score.get(
                "unknown_occlusion_keypoint_count_pred", 0),
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
            "environment_occluded_keypoint_count_true": true_score.get(
                "environment_occluded_keypoint_count_true", 0),
            "self_occluded_keypoint_count_true": true_score.get(
                "self_occluded_keypoint_count_true", 0),
            "target_surface_keypoint_count_true": true_score.get(
                "target_surface_keypoint_count_true", 0),
            "unknown_occlusion_keypoint_count_true": true_score.get(
                "unknown_occlusion_keypoint_count_true", 0),
            # ---- v5.0 第三轮真实分析指标 ----
            "geometry_target_surface_count_true": true_score.get(
                "geometry_target_surface_count_true", 0),
            "geometry_environment_occluded_count_true": true_score.get(
                "geometry_environment_occluded_count_true", 0),
            "geometry_self_occluded_count_true": true_score.get(
                "geometry_self_occluded_count_true", 0),
            "geometry_unknown_count_true": true_score.get(
                "geometry_unknown_count_true", 0),
            "geometry_none_count_true": true_score.get(
                "geometry_none_count_true", 0),
            "depth_occluded_keypoint_count_true": true_score.get(
                "depth_occluded_keypoint_count_true", 0),
            "depth_geometry_occlusion_agreement_count": true_score.get(
                "depth_geometry_occlusion_agreement_count", 0),
            "depth_geometry_occlusion_disagreement_count": true_score.get(
                "depth_geometry_occlusion_disagreement_count", 0),
            "depth_geometry_occlusion_agreement_rate": true_score.get(
                "depth_geometry_occlusion_agreement_rate"),
            "fov_visible_keypoint_count_true": true_score.get(
                "fov_visible_keypoint_count_true", 0),
            "depth_valid_in_fov_count_true": true_score.get(
                "depth_valid_in_fov_count_true", 0),
            "depth_coverage_true": true_score.get("depth_coverage_true", 0.0),
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
            "oracle_available": oracle_available,
            "oracle_gap": oracle_gap,
            "oracle_gap_valid": oracle_gap_valid,
            "oracle_gap_reason": oracle_gap_reason,
            "oracle_valid_true_candidate_count": oracle_valid,
            "oracle_selected_candidate_id": oracle_sel_id,
            "oracle_depth_eligible_candidate_count": oracle_depth_eligible,
            "oracle_excluded_low_depth_coverage_count": oracle_excluded,
            # ---- 在线选择有效性（Ours 过滤 invalid-occlusion）----
            "selected_occlusion_valid_pred": int(
                bool(pred_score.get("is_occlusion_valid_pred", True))),
            "ours_invalid_occ_candidate_excluded_count": ours_excluded_invalid,
            "ours_fallback_to_current_due_to_no_valid_occ_candidate": int(
                ours_fell_back),
            "ours_stay_by_score": int(
                policy.name == "Ours" and is_current and not ours_fell_back),
            "ours_stay_by_fallback": int(
                policy.name == "Ours" and ours_fell_back),
            # v5.0 Humanoid 状态
            "semantic_sensor_available": int(rs_curr.get(
                "semantic_sensor_available", False)),
            "semantic_assignment_ok": int(rs_curr.get(
                "semantic_assignment_ok", False)),
            "semantic_assignment_count": int(rs_curr.get(
                "semantic_assignment_count", 0)),
            "humanoid_enabled": int(config["humanoid"].get("enabled", True)),
            "humanoid_avatar_name": config["humanoid"]["avatar_name"],
            "humanoid_pose_name": hstate.pose_name,
            "humanoid_motion_frame": hstate.motion_frame,
            "humanoid_base_x": hstate.base_position[0],
            "humanoid_base_y": hstate.base_position[1],
            "humanoid_base_z": hstate.base_position[2],
            "humanoid_yaw": hstate.base_yaw,
            "humanoid_gt_skeleton_source": gt_skeleton_source,
            "humanoid_gt_keypoint_count": gt_validation["keypoint_count"],
            "humanoid_gt_direct_link_count": gt_validation["direct_link_count"],
            "humanoid_gt_link_derived_count": gt_validation["link_derived_count"],
            "humanoid_gt_fallback_count": gt_validation["fallback_count"],
            # ---- current 视角渲染验证 ----
            "current_humanoid_render_success": int(
                rs_curr["humanoid_render_success"]),
            "current_humanoid_validation_source": rs_curr.get(
                "humanoid_validation_source", ""),
            "current_humanoid_visible": int(_rs_visible(rs_curr)),
            "current_humanoid_pixel_count": _rs_pixels(rs_curr),
            "current_humanoid_pixel_ratio": _rs_pixel_ratio(rs_curr, config),
            "current_humanoid_match_ratio": _rs_match_ratio(rs_curr),
            "current_humanoid_bbox_x1": _rs_bbox(rs_curr, 0),
            "current_humanoid_bbox_y1": _rs_bbox(rs_curr, 1),
            "current_humanoid_bbox_x2": _rs_bbox(rs_curr, 2),
            "current_humanoid_bbox_y2": _rs_bbox(rs_curr, 3),
            "current_humanoid_depth_valid_ratio": rs_curr.get(
                "humanoid_depth_valid_ratio", 0.0),
            "current_humanoid_proxy_match_ratio": rs_curr.get(
                "humanoid_proxy_match_ratio"),
            # ---- selected 视角渲染验证 ----
            "selected_humanoid_render_success": int(
                selected_render_stats[policy.name]["humanoid_render_success"]),
            "selected_humanoid_validation_source": selected_render_stats[
                policy.name].get("humanoid_validation_source", ""),
            "selected_humanoid_visible": int(_rs_visible(selected_render_stats[policy.name])),
            "selected_humanoid_pixel_count": _rs_pixels(selected_render_stats[policy.name]),
            "selected_humanoid_pixel_ratio": _rs_pixel_ratio(selected_render_stats[policy.name], config),
            "selected_humanoid_match_ratio": _rs_match_ratio(selected_render_stats[policy.name]),
            "selected_humanoid_bbox_x1": _rs_bbox(selected_render_stats[policy.name], 0),
            "selected_humanoid_bbox_y1": _rs_bbox(selected_render_stats[policy.name], 1),
            "selected_humanoid_bbox_x2": _rs_bbox(selected_render_stats[policy.name], 2),
            "selected_humanoid_bbox_y2": _rs_bbox(selected_render_stats[policy.name], 3),
            "selected_humanoid_depth_valid_ratio": selected_render_stats[
                policy.name].get("humanoid_depth_valid_ratio", 0.0),
            "selected_humanoid_proxy_match_ratio": selected_render_stats[
                policy.name].get("humanoid_proxy_match_ratio"),
            # ---- 渲染/可见性增益 ----
            "humanoid_pixel_gain": _rs_pixels(selected_render_stats[policy.name])
            - _rs_pixels(rs_curr),
            # proxy_match_gain 仅在 selected 与 current 都为 gt_depth_proxy 时有效
            "humanoid_proxy_match_gain": (
                _rs_match_ratio(selected_render_stats[policy.name])
                - _rs_match_ratio(rs_curr)
                if (selected_render_stats[policy.name].get("humanoid_validation_source")
                    == "gt_depth_proxy"
                    and rs_curr.get("humanoid_validation_source") == "gt_depth_proxy")
                else None),
            "requested_human_yaw": requested_yaw,
            "actual_humanoid_yaw": actual_yaw,
            "humanoid_self_occlusion_status_pred": config["humanoid"].get(
                "raycast_self_occlusion_status", "unknown"),
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
        "requested_human_yaw": requested_yaw,
        "actual_humanoid_yaw": actual_yaw,
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
            "gt_skeleton_counts": {
                "keypoint_count": gt_validation["keypoint_count"],
                "direct_link_count": gt_validation["direct_link_count"],
                "link_derived_count": gt_validation["link_derived_count"],
                "fallback_count": gt_validation["fallback_count"],
                "missing_keypoints": gt_validation["missing_keypoints"],
            },
            "current_render_stats": rs_curr,
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
        "current_humanoid_render_success": bool(rs_curr["humanoid_render_success"]),
        "current_humanoid_validation_source": rs_curr.get(
            "humanoid_validation_source", ""),
        "oracle_Q_true": oracle_q_true,
        "oracle_gap": oracle_gap,
        "oracle_depth_eligible_candidate_count": oracle_depth_eligible,
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
    oracle_policy = OraclePolicy(
        min_depth_coverage=config.get("oracle", {}).get(
            "min_depth_coverage", 0.8)) if oracle_enabled else None

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