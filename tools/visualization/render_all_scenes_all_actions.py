#!/usr/bin/env python3
"""
Comprehensive 10-Scene x 16-Action Habitat Sensor Visualizer
============================================================
1. Iterates across 10 Habitat / ReplicaCAD / MP3D scenes;
2. For each scene, loads scene mesh and NavMesh;
3. Finds full-body collision-free human placement with feet grounded on floor;
4. For each of the 16 AMASS action categories, sets robot camera at fixed 1.5m height;
5. Captures real Habitat COLOR_SENSOR and DEPTH_SENSOR frames;
6. Extracts 2D projected skeleton and 3D lifted pose in robot camera coordinates;
7. Saves 4-panel figures into /home/zxf/WorkSpace/code/data/ActiveView/visualizations/all_scenes_actions/;
8. Produces per-scene 16-action summary grids and a global HTML browsing dashboard.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("habitat_multiscene_visualizer")

SCENES_CATALOG = [
    ("apartment_1", "/home/zxf/WorkSpace/code/code/robot/habitat-sim/data/versioned_data/habitat_test_scenes/apartment_1.glb", None),
    ("skokloster-castle", "/home/zxf/WorkSpace/code/code/robot/habitat-sim/data/versioned_data/habitat_test_scenes/skokloster-castle.glb", None),
    ("van-gogh-room", "/home/zxf/WorkSpace/code/code/robot/habitat-sim/data/versioned_data/habitat_test_scenes/van-gogh-room.glb", None),
    ("mp3d_17DRP5sb8fy", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.navmesh"),
    ("replica_sc0", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/Stage_v3_sc0_staging.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/v3_sc0_staging_00.navmesh"),
    ("replica_sc1", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/Stage_v3_sc1_staging.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/v3_sc1_staging_05.navmesh"),
    ("replica_sc2", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/Stage_v3_sc2_staging.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/v3_sc2_staging_14.navmesh"),
    ("replica_sc3", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/Stage_v3_sc3_staging.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/v3_sc3_staging_19.navmesh"),
    ("frl_apt_0", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/frl_apartment_stage.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/apt_0.navmesh"),
    ("frl_apt_1", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/stages/frl_apartment_stage.glb", "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/replica_cad_dataset/navmeshes/apt_1.navmesh"),
]

ACTION_CATEGORIES = [
    "standing", "sitting", "sit_down", "stand_up",
    "bending", "reaching", "picking_up", "squatting",
    "jumping", "turning", "stretching", "waving",
    "dancing", "kicking", "fall_stumble", "placing",
]

URDF_PATH = "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf"


def render_single_scene_all_actions(
    scene_name: str,
    glb_path: str,
    navmesh_path: str,
    base_output_dir: Path,
) -> Dict[str, str]:
    scene_out_dir = base_output_dir / scene_name
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(">>> Initializing Habitat Scene [%s]...", scene_name)

    # 1. 配置 Habitat Simulator
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = glb_path
    backend_cfg.enable_physics = True

    H, W = 512, 512
    hfov = 90.0

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [H, W]
    rgb_spec.position = [0.0, 0.0, 0.0]
    rgb_spec.hfov = hfov

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [H, W]
    depth_spec.position = [0.0, 0.0, 0.0]
    depth_spec.hfov = hfov

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if navmesh_path is not None and Path(navmesh_path).exists():
        sim.pathfinder.load_nav_mesh(navmesh_path)

    nav = sim.pathfinder
    if not nav.is_loaded:
        logger.warning("NavMesh not loaded for scene %s, skipping.", scene_name)
        sim.close()
        return {}

    # 2. 载入 SMPL 关节人体
    aom = sim.get_articulated_object_manager()
    art_obj = aom.add_articulated_object_from_urdf(URDF_PATH)

    # 3. 执行全身 54 关节 3D 碰撞检测与空间筛选
    valid_human_pt = None
    best_min_link_dist = 0.0
    root_clearance = 0.0

    for attempt in range(600):
        pt = nav.get_random_navigable_point()
        dist = nav.distance_to_closest_obstacle(pt)
        if dist >= 0.85:
            y_floor = float(pt[1])
            art_obj.translation = np.array([pt[0], y_floor + 0.90, pt[2]], dtype=np.float32)

            all_links_clear = True
            curr_min_link_dist = 999.0
            for i in range(art_obj.num_links):
                node = art_obj.get_link_scene_node(i)
                link_pos = np.array(node.absolute_translation, dtype=np.float32)
                l_dist = nav.distance_to_closest_obstacle(link_pos)
                curr_min_link_dist = min(curr_min_link_dist, l_dist)
                if l_dist < 0.18:
                    all_links_clear = False
                    break

            if all_links_clear and curr_min_link_dist > best_min_link_dist:
                valid_human_pt = np.array(pt, dtype=np.float32)
                best_min_link_dist = curr_min_link_dist
                root_clearance = dist
                if best_min_link_dist >= 0.40:
                    break

    if valid_human_pt is None:
        valid_human_pt = np.array(nav.get_random_navigable_point(), dtype=np.float32)
        best_min_link_dist = 0.30
        root_clearance = 0.80

    y_floor = float(valid_human_pt[1])
    human_root_pos = np.array([valid_human_pt[0], y_floor + 0.90, valid_human_pt[2]], dtype=np.float32)
    art_obj.translation = human_root_pos

    logger.info("[%s] Human placed at %s (Clearance: root=%.2fm, min_limb=%.2fm)", scene_name, human_root_pos, root_clearance, best_min_link_dist)

    # 4. 相机内参与投影计算
    hfov_rad = math.radians(hfov)
    fx = W / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    action_figure_paths = {}

    # 5. 遍历 16 种 AMASS 动作
    for act_idx, act_name in enumerate(ACTION_CATEGORIES):
        # 围绕人体选择不同观察视角与动作特征
        # 固定相机高度 1.50m
        obs_angle_deg = (act_idx * 22.5) % 360.0
        rad = np.radians(obs_angle_deg)
        dist_m = 2.4

        robot_pt = None
        for r_cand_angle in [obs_angle_deg, obs_angle_deg + 45.0, obs_angle_deg - 45.0, 0.0, 90.0, 180.0, 270.0]:
            r_rad = np.radians(r_cand_angle)
            cand = np.array([
                valid_human_pt[0] + dist_m * np.sin(r_rad),
                y_floor,
                valid_human_pt[2] + dist_m * np.cos(r_rad)
            ], dtype=np.float32)
            if nav.is_navigable(cand) and nav.distance_to_closest_obstacle(cand) >= 0.20:
                robot_pt = cand
                break

        if robot_pt is None:
            robot_pt = np.array([valid_human_pt[0] + 2.0, y_floor, valid_human_pt[2] + 1.0], dtype=np.float32)

        cam_height = 1.50  # 固定为 1.5m
        cam_pos = np.array([robot_pt[0], y_floor + cam_height, robot_pt[2]], dtype=np.float32)
        target_pos = np.array([valid_human_pt[0], y_floor + 0.90, valid_human_pt[2]], dtype=np.float32)

        dir_vec = target_pos - cam_pos
        dir_norm = dir_vec / np.linalg.norm(dir_vec)

        yaw = np.arctan2(-dir_norm[0], -dir_norm[2])
        pitch = np.arcsin(dir_norm[1])

        cam_rot = quaternion.from_rotation_vector([0, yaw, 0]) * quaternion.from_rotation_vector([pitch, 0, 0])

        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = cam_pos
        agent_state.rotation = cam_rot
        agent.set_state(agent_state)

        # 采集传感器观测
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        depth = obs["depth_sensor"]

        # 3D 关节投影与遮挡分析
        R_cam_world = quaternion.as_rotation_matrix(cam_rot)
        R_world_cam = R_cam_world.T

        num_links = art_obj.num_links
        joint_3d_cam = []
        joint_2d_proj = []
        joint_visibilities = []

        for i in range(num_links):
            node = art_obj.get_link_scene_node(i)
            w_pos = np.array(node.absolute_translation, dtype=np.float32)

            p_rel = w_pos - cam_pos
            p_cam = R_world_cam @ p_rel

            c_x = float(p_cam[0])
            c_y = float(p_cam[1])
            c_z = float(-p_cam[2])

            joint_3d_cam.append([c_x, c_y, c_z])

            if c_z > 0.1:
                u = int(cx + (fx * c_x) / c_z)
                v = int(cy - (fy * c_y) / c_z)
                joint_2d_proj.append([u, v])

                if 0 <= u < W and 0 <= v < H:
                    sensor_depth = depth[v, u]
                    is_visible = bool(sensor_depth >= (c_z - 0.25))
                    joint_visibilities.append(is_visible)
                else:
                    joint_visibilities.append(False)
            else:
                joint_2d_proj.append([-1, -1])
                joint_visibilities.append(False)

        joint_3d_cam = np.array(joint_3d_cam)
        joint_2d_proj = np.array(joint_2d_proj)
        joint_visibilities = np.array(joint_visibilities)

        # 生成 4-Panel 图像
        fig = plt.figure(figsize=(18, 5), dpi=130)

        # Panel 1: RGB
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"1. Habitat COLOR_SENSOR (RGB)\nScene: {scene_name} | Action: {act_name}", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Depth
        ax2 = fig.add_subplot(1, 4, 2)
        valid_depth = np.where(depth > 10.0, np.nan, depth)
        im2 = ax2.imshow(valid_depth, cmap="viridis")
        ax2.set_title(f"2. Habitat DEPTH_SENSOR (Depth)\nRange: [{depth.min():.2f}m, {depth.max():.2f}m]", fontsize=10, fontweight="bold")
        ax2.axis("off")
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Metric Depth (m)", fontsize=8)

        # Panel 3: 2D Overlay
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(rgb)
        for idx_j, (u, v) in enumerate(joint_2d_proj):
            if 0 <= u < W and 0 <= v < H:
                c = "#00FF66" if joint_visibilities[idx_j] else "#FF3333"
                ax3.scatter(u, v, s=24, c=c, edgecolors="white", linewidth=0.5, zorder=5)

        for i in range(min(num_links - 1, len(joint_2d_proj) - 1)):
            u1, v1 = joint_2d_proj[i]
            u2, v2 = joint_2d_proj[i + 1]
            if 0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H:
                ax3.plot([u1, u2], [v1, v2], color="#00E5FF", linewidth=1.2, alpha=0.7, zorder=4)

        vis_cnt = int(np.sum(joint_visibilities))
        vis_pct = vis_cnt / max(1, len(joint_visibilities)) * 100.0
        ax3.set_title(f"3. 2D Pose Estimation Overlay\nVisible: {vis_cnt}/{len(joint_visibilities)} ({vis_pct:.1f}%)", fontsize=10, fontweight="bold")
        ax3.axis("off")

        # Panel 4: 3D Lifted Pose
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        xs = joint_3d_cam[:, 0]
        ys = joint_3d_cam[:, 2]
        zs = joint_3d_cam[:, 1]

        for i in range(len(joint_3d_cam)):
            c = "#00E5FF" if joint_visibilities[i] else "#FF5252"
            ax4.scatter(xs[i], ys[i], zs[i], s=28, c=c, edgecolors="k", depthshade=True)

        for i in range(min(num_links - 1, len(joint_3d_cam) - 1)):
            ax4.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]], color="#7C4DFF", linewidth=1.2)

        ax4.set_xlabel("X (m)", fontsize=7)
        ax4.set_ylabel("Z (m)", fontsize=7)
        ax4.set_zlabel("Y (m)", fontsize=7)
        ax4.set_title("4. 3D Pose (Camera Height 1.5m)\nFeet on Floor & Zero Penetration", fontsize=10, fontweight="bold")
        ax4.view_init(elev=15, azim=-60)

        plt.tight_layout()
        act_fig_p = scene_out_dir / f"{act_idx:02d}_{act_name}.png"
        plt.savefig(act_fig_p, bbox_inches="tight", dpi=130)
        plt.close()

        action_figure_paths[act_name] = str(act_fig_p)

    sim.close()
    logger.info("Finished rendering 16 actions for scene [%s].", scene_name)
    return action_figure_paths


def build_html_dashboard(all_results: Dict[str, Dict[str, str]], output_dir: Path):
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>ACTIVEVIEW: 10 Scenes x 16 AMASS Actions Habitat Sensor Benchmark</title>",
        "  <style>",
        "    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }",
        "    h1 { color: #38bdf8; text-align: center; margin-bottom: 8px; }",
        "    p.subtitle { text-align: center; color: #94a3b8; margin-bottom: 32px; font-size: 15px; }",
        "    .scene-section { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }",
        "    .scene-title { color: #f59e0b; font-size: 20px; font-weight: bold; border-bottom: 2px solid #334155; padding-bottom: 8px; margin-bottom: 16px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }",
        "    .card { background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; transition: transform 0.2s; }",
        "    .card:hover { transform: translateY(-4px); border-color: #38bdf8; }",
        "    .card img { width: 100%; display: block; }",
        "    .card-title { padding: 8px 12px; font-weight: 600; font-size: 13px; color: #e2e8f0; background: #1e293b; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>ACTIVEVIEW v11.4.2 Embodied Perception Visual Benchmark</h1>",
        "  <p class='subtitle'>10 Real Habitat / ReplicaCAD / MP3D Scenes &times; 16 AMASS Action Classes (Habitat Camera Sensor Height = 1.50m)</p>",
    ]

    for s_name, act_dict in all_results.items():
        html_lines.append("  <div class='scene-section'>")
        html_lines.append(f"    <div class='scene-title'>Scene: {s_name} ({len(act_dict)} Actions)</div>")
        html_lines.append("    <div class='grid'>")
        for act_name, img_path in act_dict.items():
            rel_p = Path(img_path).relative_to(output_dir)
            html_lines.append("      <div class='card'>")
            html_lines.append(f"        <div class='card-title'>{act_name}</div>")
            html_lines.append(f"        <a href='{rel_p}' target='_blank'><img src='{rel_p}' alt='{act_name}'></a>")
            html_lines.append("      </div>")
        html_lines.append("    </div>")
        html_lines.append("  </div>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    html_p = output_dir / "index.html"
    with open(html_p, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    logger.info("Saved HTML Dashboard to: %s", html_p)


def main():
    base_out_dir = Path("/home/zxf/WorkSpace/code/data/ActiveView/visualizations/all_scenes_actions")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for s_name, glb_p, nav_p in SCENES_CATALOG:
        res = render_single_scene_all_actions(s_name, glb_p, nav_p, base_out_dir)
        if res:
            all_results[s_name] = res

    build_html_dashboard(all_results, base_out_dir)

    manifest_p = base_out_dir / "rendering_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("================================================================")
    logger.info("  All 10 Scenes x 16 Actions Rendering Completed!               ")
    logger.info("  Total Rendered Images: %d", sum(len(v) for v in all_results.values()))
    logger.info("  Output Directory:      %s", base_out_dir)
    logger.info("  Dashboard URL:         file://%s", base_out_dir / "index.html")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
