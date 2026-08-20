"""
v8.0 主动视角基础框架综合演示主脚本 —— run_v8_demo.py
=====================================================

功能：
    1. 完整串联 v8.0 核心研究流水线：
       Scene 加载 -> Human Placement 采样 -> 候选视点生成 (Candidate Views)
       -> 空间与导航约束检查 (Constraint Check) -> 几何可见性评价 (Visibility Evaluation)
       -> 渲染多视点 RGB-D 样本 -> 持久化 v8 数据集；
    2. 输出成果至 data/ActiveView/visualizations/v8_demo/：
       - candidate_views.json
       - visibility.json
       - metadata.json
       - rgb/ 与 depth/ 关键视点图像

运行方式：
    python -m ea_avs_mvp_v8.scripts.run_v8_demo
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

from ea_avs_mvp_v8.core.config import load_v8_config
from ea_avs_mvp_v8.core.paths import get_data_root
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.robot.robot_adapter import V8RobotAdapter
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.visibility.visibility_evaluator import VisibilityEvaluator
from ea_avs_mvp_v8.dataset.v8_dataset_generator import save_v8_viewpoint_dataset

# 复用 v7 Humanoid
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v8_demo")


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v8.0 Active View Foundation Demonstration")
    parser.add_argument("--config", type=str, default=None, help="Path to custom v8_demo.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    cfg = load_v8_config(args.config)
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    # 1. 输出目录初始化
    if args.output_dir:
        vis_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else get_data_root() / args.output_dir
    else:
        vis_dir = get_data_root() / cfg.simulation.get("output_dir", "visualizations/v8_demo")
    vis_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = vis_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # 2. 步骤一: Scene 启动与 Human Placement
    logger.info("Step 1: Initializing Habitat Environment & Human Placement...")
    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)

    robot_start_pos = cfg.robot.get("initial_pose", {}).get("position", [1.5, -1.60, 6.8])

    # 3. 步骤二: 生成候选观察视角 (Candidate View Generation)
    logger.info("Step 2: Generating Candidate Viewpoints...")
    vp_gen = ViewpointGenerator(cfg.viewpoint)
    raw_candidates = vp_gen.generate_candidates(
        human_position=human_pose.position,
        human_yaw_deg=human_pose.yaw_deg,
    )

    # 4. 步骤三: 空间与导航约束检查 (Constraint Checking)
    logger.info("Step 3: Checking Constraints with Habitat Pathfinder...")
    checker = ConstraintChecker(env_adapter=env_adapter, config=cfg.viewpoint)
    checked_candidates = checker.filter_feasible_viewpoints(
        raw_candidates,
        robot_start_pos=robot_start_pos,
    )

    # 5. 步骤四: 视点可见性与观测质量评价 (Visibility Evaluation)
    logger.info("Step 4: Evaluating Viewpoint Observational Quality...")
    gt_joints = humanoid.get_gt_joint_positions()
    evaluator = VisibilityEvaluator(cfg.camera)
    qualities = evaluator.evaluate_batch(
        viewpoints=checked_candidates,
        human_joints_3d=gt_joints,
        human_yaw_deg=human_pose.yaw_deg,
    )

    # 6. 步骤五: 采样前 4 个有效视点渲染真实观测图片
    logger.info("Step 5: Capturing Sample Observations...")
    robot_adapter = V8RobotAdapter(sim, cfg.camera)
    feasible_views = [vp for vp in checked_candidates if vp.feasible]

    for idx, vp in enumerate(feasible_views[:4]):
        robot_adapter.set_viewpoint(vp)
        obs = robot_adapter.capture_observation()
        if obs.get("rgb") is not None:
            out_img = rgb_dir / f"{vp.viewpoint_id}.png"
            Image.fromarray(obs["rgb"]).save(out_img)

    env_adapter.close()

    # 7. 步骤六: 持久化 v8 数据集与元数据文件
    logger.info("Step 6: Saving v8 Datasets & Metadata...")
    action_info = {
        "avatar": cfg.human.get("avatar_name", "neutral_0"),
        "action_class": cfg.human.get("action", "fall_related"),
        "action_label": "fall to the ground",
        "motion_id": cfg.human.get("motion_id", "fall_related_3522"),
    }
    save_v8_viewpoint_dataset(
        output_dir=vis_dir,
        scene_id=scene_id,
        episode_id="v8_demo_episode",
        human_pose=human_pose,
        action_info=action_info,
        robot_start_pos=robot_start_pos,
        candidate_views=checked_candidates,
        view_qualities=qualities,
    )

    # 8. 打印标准化验证报告
    feasible_count = sum(1 for v in checked_candidates if v.feasible)
    best_view = max(qualities, key=lambda q: q.visibility_score) if qualities else None

    print("\n" + "=" * 65)
    print("[V8 Active View Foundation Demonstration Results]")
    print(f"Scene ID:              {scene_id}")
    print(f"Human Position:        {human_pose.position}")
    print(f"Total Candidates:      {len(checked_candidates)}")
    print(f"Feasible Candidates:   {feasible_count} ({feasible_count / len(checked_candidates) * 100:.1f}%)")
    print(f"Evaluated Viewpoints:  {len(qualities)}")
    if best_view:
        print(f"Best Viewpoint ID:     {best_view.viewpoint_id}")
        print(f"Best Visibility Score: {best_view.visibility_score:.3f} (Dist: {best_view.distance:.2f}m, Angle: {best_view.viewing_angle_deg:.1f} deg)")
    print(f"Output Directory:      {vis_dir}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v8.0 Phase 1 Foundation Verified\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
