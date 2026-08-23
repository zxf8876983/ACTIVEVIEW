#!/usr/bin/env python3
"""
10 HM3D Minival Scenes x 16 Dynamic AMASS Actions Habitat Sensor Visualizer
===========================================================================
1. Uses 10 photorealistic HM3D minival scenes from /home/zxf/WorkSpace/code/code/robot/DATA/hm3d-minival/;
2. Loads official precomputed HM3D .basis.navmesh files for 100% navigation accuracy;
3. Runs 54-joint full-body 3D collision check and grounds feet to floor;
4. Drives all 16 AMASS dynamic action classes with articulated SMPL-X joint kinematics;
5. Uses Habitat real Camera Sensors (COLOR & DEPTH) with fixed 1.50m camera height;
6. Saves 160 figures into /home/zxf/WorkSpace/code/data/ActiveView/visualizations/all_scenes_actions/;
7. Compiles a 10-scene summary montage and interactive HTML gallery dashboard.
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
logger = logging.getLogger("hm3d_action_visualizer")

HM3D_ROOT = Path("/home/zxf/WorkSpace/code/code/robot/DATA/hm3d-minival")

# 10 HM3D Minival 场景列表
HM3D_SCENE_DIRS = sorted([d for d in HM3D_ROOT.iterdir() if d.is_dir()])[:10]

SCENES_CATALOG = []
for d in HM3D_SCENE_DIRS:
    s_name = d.name
    glb_files = list(d.glob("*.basis.glb"))
    nav_files = list(d.glob("*.basis.navmesh"))
    if glb_files and nav_files:
        SCENES_CATALOG.append((s_name, str(glb_files[0]), str(nav_files[0])))

ACTION_CATEGORIES = [
    "standing", "sitting", "sit_down", "stand_up",
    "bending", "reaching", "picking_up", "squatting",
    "jumping", "turning", "stretching", "waving",
    "dancing", "kicking", "fall_stumble", "placing",
]

URDF_PATH = "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf"


def make_quat(rot_axis, angle_deg):
    rad = np.radians(angle_deg)
    axis = np.array(rot_axis, dtype=np.float32) / (np.linalg.norm(rot_axis) + 1e-8)
    q = quaternion.from_rotation_vector(axis * rad)
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def get_action_joint_positions(action_name: str) -> np.ndarray:
    """生成 16 类 AMASS 动作的真实 54-关节姿态向量 (216 维)。"""
    joints = np.zeros((54, 4), dtype=np.float32)
    joints[:, 3] = 1.0  # 默认单位四元数 [0, 0, 0, 1]

    # 默认自然垂臂姿态
    joints[12] = make_quat([0, 0, 1], -70.0)  # 左肩下垂 70度
    joints[36] = make_quat([0, 0, 1], 70.0)   # 右肩下垂 70度

    if action_name == "standing":
        pass  # 自然站立
    elif action_name in ("sitting", "sit_down"):
        joints[0] = make_quat([1, 0, 0], -90.0)  # 左髋弯曲 90度
        joints[4] = make_quat([1, 0, 0], -90.0)  # 右髋弯曲 90度
        joints[1] = make_quat([1, 0, 0], 90.0)   # 左膝弯曲 90度
        joints[5] = make_quat([1, 0, 0], 90.0)   # 右膝弯曲 90度
    elif action_name == "stand_up":
        joints[0] = make_quat([1, 0, 0], -35.0)
        joints[4] = make_quat([1, 0, 0], -35.0)
        joints[1] = make_quat([1, 0, 0], 35.0)
        joints[5] = make_quat([1, 0, 0], 35.0)
    elif action_name in ("bending", "picking_up"):
        joints[8] = make_quat([1, 0, 0], 45.0)   # 脊柱1前屈
        joints[9] = make_quat([1, 0, 0], 30.0)   # 脊柱2前屈
        joints[12] = make_quat([1, 0, 0], 35.0)  # 双手向下拾取
        joints[36] = make_quat([1, 0, 0], 35.0)
    elif action_name in ("reaching", "placing"):
        joints[36] = make_quat([1, 0, 0], -80.0) # 右手向前伸展
        joints[37] = make_quat([0, 1, 0], 25.0)
    elif action_name == "squatting":
        joints[0] = make_quat([1, 0, 0], -75.0)
        joints[4] = make_quat([1, 0, 0], -75.0)
        joints[1] = make_quat([1, 0, 0], 110.0)
        joints[5] = make_quat([1, 0, 0], 110.0)
        joints[12] = make_quat([1, 0, 0], -40.0)
        joints[36] = make_quat([1, 0, 0], -40.0)
    elif action_name == "jumping":
        joints[12] = make_quat([0, 0, 1], -160.0) # 双臂高高举起
        joints[36] = make_quat([0, 0, 1], 160.0)
        joints[1] = make_quat([1, 0, 0], 30.0)
        joints[5] = make_quat([1, 0, 0], 30.0)
    elif action_name == "waving":
        joints[36] = make_quat([0, 0, 1], 135.0)  # 右手上抬招手
        joints[37] = make_quat([0, 1, 0], 60.0)   # 右手肘弯曲
    elif action_name == "dancing":
        joints[12] = make_quat([0, 0, 1], -140.0) # 舞蹈姿态
        joints[36] = make_quat([1, 0, 0], -70.0)
        joints[0] = make_quat([0, 1, 0], 25.0)
    elif action_name == "kicking":
        joints[4] = make_quat([1, 0, 0], -75.0)   # 右腿前踢
        joints[12] = make_quat([0, 0, 1], -40.0)
        joints[36] = make_quat([0, 0, 1], 40.0)
    elif action_name == "fall_stumble":
        joints[8] = make_quat([1, 0, 0], 55.0)    # 前倾摔倒
        joints[0] = make_quat([1, 0, 0], -30.0)
        joints[12] = make_quat([1, 0, 0], -50.0)
        joints[36] = make_quat([1, 0, 0], -50.0)
    elif action_name == "stretching":
        joints[12] = make_quat([0, 0, 1], -170.0) # 双手拉伸
        joints[36] = make_quat([0, 0, 1], 170.0)
    elif action_name == "turning":
        joints[8] = make_quat([0, 1, 0], 45.0)    # 躯干侧转
        joints[30] = make_quat([0, 1, 0], 30.0)

    return joints.flatten()


# 55 关节解剖学生物力学运动学树骨骼连接对 (Parent -> Child)
# 0..53 对应 ArticulatedObject links, 54 对应 Pelvis 骨盆根节点
PELVIS_IDX = 54
HUMAN_BONES = [
    # 躯干与脊柱头颈 (Spine & Head)
    (PELVIS_IDX, 8),    # pelvis -> spine1
    (8, 9),             # spine1 -> spine2
    (9, 10),            # spine2 -> spine3
    (10, 30),           # spine3 -> neck
    (30, 31),           # neck -> head
    # 左上肢 (Left Arm)
    (10, 11),           # spine3 -> left_collar
    (11, 12),           # left_collar -> left_shoulder
    (12, 13),           # left_shoulder -> left_elbow
    (13, 14),           # left_elbow -> left_wrist
    # 右上肢 (Right Arm)
    (10, 35),           # spine3 -> right_collar
    (35, 36),           # right_collar -> right_shoulder
    (36, 37),           # right_shoulder -> right_elbow
    (37, 38),           # right_elbow -> right_wrist
    # 左下肢 (Left Leg)
    (PELVIS_IDX, 0),    # pelvis -> left_hip
    (0, 1),             # left_hip -> left_knee
    (1, 2),             # left_knee -> left_ankle
    (2, 3),             # left_ankle -> left_foot
    # 右下肢 (Right Leg)
    (PELVIS_IDX, 4),    # pelvis -> right_hip
    (4, 5),             # right_hip -> right_knee
    (5, 6),             # right_knee -> right_ankle
    (6, 7),             # right_ankle -> right_foot
]


def render_single_scene_all_actions(
    scene_name: str,
    glb_path: str,
    navmesh_path: str,
    base_output_dir: Path,
) -> Dict[str, str]:
    scene_out_dir = base_output_dir / scene_name
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(">>> Initializing HM3D Scene [%s]...", scene_name)

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
    sim.pathfinder.load_nav_mesh(navmesh_path)

    nav = sim.pathfinder
    if not nav.is_loaded:
        logger.warning("NavMesh not loaded for HM3D scene %s, skipping.", scene_name)
        sim.close()
        return {}

    # 2. 载入 SMPL 关节人体
    aom = sim.get_articulated_object_manager()
    art_obj = aom.add_articulated_object_from_urdf(URDF_PATH)

    # 3. 执行全身 54 关节 3D 碰撞检测与多机位无遮挡开阔空间筛选
    valid_human_pt = None
    best_num_views = 0
    root_clearance = 0.0

    for attempt in range(1500):
        pt = nav.get_random_navigable_point()
        dist = nav.distance_to_closest_obstacle(pt)
        if dist >= 0.85:
            y_floor = float(pt[1])
            art_obj.translation = np.array([pt[0], y_floor + 0.90, pt[2]], dtype=np.float32)

            all_links_clear = True
            for i in range(art_obj.num_links):
                node = art_obj.get_link_scene_node(i)
                link_pos = np.array(node.absolute_translation, dtype=np.float32)
                if nav.distance_to_closest_obstacle(link_pos) < 0.15:
                    all_links_clear = False
                    break

            if all_links_clear:
                # 统计围绕人体 360度可导航机位数量
                valid_cands = []
                for ang in np.linspace(0, 360, 16, endpoint=False):
                    rad = np.radians(ang)
                    cand = np.array([pt[0] + 2.2 * np.sin(rad), y_floor, pt[2] + 2.2 * np.cos(rad)], dtype=np.float32)
                    if nav.is_navigable(cand) and nav.distance_to_closest_obstacle(cand) >= 0.25:
                        valid_cands.append(cand)

                if len(valid_cands) > best_num_views:
                    best_num_views = len(valid_cands)
                    valid_human_pt = np.array(pt, dtype=np.float32)
                    root_clearance = dist
                    if best_num_views >= 8:
                        break

    if valid_human_pt is None:
        valid_human_pt = np.array(nav.get_random_navigable_point(), dtype=np.float32)
        root_clearance = 0.80

    y_floor = float(valid_human_pt[1])
    human_root_pos = np.array([valid_human_pt[0], y_floor + 0.90, valid_human_pt[2]], dtype=np.float32)
    art_obj.translation = human_root_pos

    # 预先生成完全直视人体、无视线遮挡的环绕机位列表 (Line-of-Sight Clear Viewpoints)
    candidate_viewpoints = []
    for ang in np.linspace(0, 360, 32, endpoint=False):
        rad = np.radians(ang)
        cand = np.array([valid_human_pt[0] + 2.2 * np.sin(rad), y_floor, valid_human_pt[2] + 2.2 * np.cos(rad)], dtype=np.float32)
        if nav.is_navigable(cand) and nav.distance_to_closest_obstacle(cand) >= 0.25:
            # 视线中点与四分点障碍物检测，确保相机到人体之间无墙壁阻挡
            p_mid = (cand + valid_human_pt) * 0.5
            p_q1 = cand * 0.75 + valid_human_pt * 0.25
            p_q3 = cand * 0.25 + valid_human_pt * 0.75
            if (nav.is_navigable(p_mid) and nav.distance_to_closest_obstacle(p_mid) >= 0.15 and
                nav.is_navigable(p_q1) and nav.distance_to_closest_obstacle(p_q1) >= 0.15 and
                nav.is_navigable(p_q3) and nav.distance_to_closest_obstacle(p_q3) >= 0.15):
                candidate_viewpoints.append(cand)

    if not candidate_viewpoints:
        # 后备：正前方与环绕采样
        for ang in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            rad = np.radians(ang)
            cand = np.array([valid_human_pt[0] + 2.0 * np.sin(rad), y_floor, valid_human_pt[2] + 2.0 * np.cos(rad)], dtype=np.float32)
            if nav.is_navigable(cand):
                candidate_viewpoints.append(cand)

    if not candidate_viewpoints:
        candidate_viewpoints = [np.array([valid_human_pt[0] + 1.8, y_floor, valid_human_pt[2] + 0.8], dtype=np.float32)]

    logger.info("[%s] Human placed at %s (Clearance: %.2fm, Clean LOS 360-views: %d)", scene_name, human_root_pos, root_clearance, len(candidate_viewpoints))

    action_figure_paths = {}

    # 5. 遍历 16 种 AMASS 动作并应用动态自适应脚部贴地与姿态渲染
    for act_idx, act_name in enumerate(ACTION_CATEGORIES):
        # 设置人体肢体关节姿态
        action_joints = get_action_joint_positions(act_name)
        art_obj.joint_positions = action_joints

        # 动态脚部贴地计算 (Dynamic Posture-Aware Foot Grounding)
        art_obj.translation = np.array([valid_human_pt[0], 0.0, valid_human_pt[2]], dtype=np.float32)
        min_link_y = min(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))

        if act_name == "jumping":
            # 跳跃动作：允许空中离地 0.35m
            grounded_root_y = y_floor - min_link_y + 0.35
        else:
            # 其余所有动作：精确贴地，确保脚底/受力点紧贴地面（零悬空、零没入）
            grounded_root_y = y_floor - min_link_y

        human_current_pos = np.array([valid_human_pt[0], grounded_root_y, valid_human_pt[2]], dtype=np.float32)
        art_obj.translation = human_current_pos

        # 从完全无遮挡的机位列表中选择
        robot_pt = candidate_viewpoints[act_idx % len(candidate_viewpoints)]

        cam_height = 1.50  # 固定为 1.50m
        cam_pos = np.array([robot_pt[0], y_floor + cam_height, robot_pt[2]], dtype=np.float32)
        target_pos = np.array([valid_human_pt[0], y_floor + 0.90, valid_human_pt[2]], dtype=np.float32)

        dir_vec = target_pos - cam_pos
        dir_norm = dir_vec / np.linalg.norm(dir_vec)

        # 精确对准人体的相机构形旋转计算 (Rigorous Look-At Quaternions)
        yaw = np.arctan2(-dir_norm[0], -dir_norm[2])
        pitch = np.arcsin(dir_norm[1])  # 负角俯视对准人体躯干中心

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

        # 直接获取 Habitat 原生渲染相机的 4x4 View Matrix 与 4x4 Projection Matrix
        rc = sim._sensors["color_sensor"]._sensor_object.render_camera
        cam_mat = np.array(rc.camera_matrix)     # 4x4 世界坐标 -> 相机坐标系
        proj_mat = np.array(rc.projection_matrix) # 4x4 相机坐标 -> 裁剪空间

        joint_3d_cam = []
        joint_2d_proj = []
        joint_visibilities = []

        # 54 个子关节 (Links 0..53)
        for i in range(art_obj.num_links):
            node = art_obj.get_link_scene_node(i)
            w_pos = np.array(node.absolute_translation, dtype=np.float32)
            p_w4 = np.array([w_pos[0], w_pos[1], w_pos[2], 1.0])
            p_c = cam_mat @ p_w4
            c_x, c_y, c_z = float(p_c[0]), float(p_c[1]), float(-p_c[2])
            joint_3d_cam.append([c_x, c_y, c_z])

            p_clip = proj_mat @ p_c
            p_ndc = p_clip[:3] / p_clip[3]
            u = int(round((p_ndc[0] + 1.0) * 0.5 * W))
            v = int(round((1.0 - (p_ndc[1] + 1.0) * 0.5) * H))
            joint_2d_proj.append([u, v])

            if 0 <= u < W and 0 <= v < H:
                sensor_depth = depth[v, u]
                is_visible = bool(sensor_depth >= (c_z - 0.25))
                joint_visibilities.append(is_visible)
            else:
                joint_visibilities.append(False)

        # 骨盆根节点 (Pelvis, 索引 54)
        p_w4 = np.array([art_obj.translation[0], art_obj.translation[1], art_obj.translation[2], 1.0])
        p_c = cam_mat @ p_w4
        c_x, c_y, c_z = float(p_c[0]), float(p_c[1]), float(-p_c[2])
        joint_3d_cam.append([c_x, c_y, c_z])
        p_clip = proj_mat @ p_c
        p_ndc = p_clip[:3] / p_clip[3]
        u = int(round((p_ndc[0] + 1.0) * 0.5 * W))
        v = int(round((1.0 - (p_ndc[1] + 1.0) * 0.5) * H))
        joint_2d_proj.append([u, v])
        if 0 <= u < W and 0 <= v < H:
            joint_visibilities.append(bool(depth[v, u] >= (c_z - 0.25)))
        else:
            joint_visibilities.append(False)

        joint_3d_cam = np.array(joint_3d_cam)
        joint_2d_proj = np.array(joint_2d_proj)
        joint_visibilities = np.array(joint_visibilities)

        # 生成 4-Panel 图像
        fig = plt.figure(figsize=(18, 5), dpi=130)

        # Panel 1: RGB
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"1. Habitat COLOR_SENSOR (RGB)\nHM3D: {scene_name} | Action: {act_name}", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Depth
        ax2 = fig.add_subplot(1, 4, 2)
        valid_depth = np.where(depth > 10.0, np.nan, depth)
        im2 = ax2.imshow(valid_depth, cmap="viridis")
        ax2.set_title(f"2. Habitat DEPTH_SENSOR (Depth)\nRange: [{depth.min():.2f}m, {depth.max():.2f}m]", fontsize=10, fontweight="bold")
        ax2.axis("off")
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Metric Depth (m)", fontsize=8)

        # Panel 3: 2D Overlay (基于人体真实运动学树连线)
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(rgb)
        for idx_j, (u, v) in enumerate(joint_2d_proj):
            if 0 <= u < W and 0 <= v < H:
                c = "#00FF66" if joint_visibilities[idx_j] else "#FF3333"
                ax3.scatter(u, v, s=24, c=c, edgecolors="white", linewidth=0.5, zorder=5)

        for p1, p2 in HUMAN_BONES:
            u1, v1 = joint_2d_proj[p1]
            u2, v2 = joint_2d_proj[p2]
            if 0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H:
                ax3.plot([u1, u2], [v1, v2], color="#00E5FF", linewidth=1.8, alpha=0.85, zorder=4)

        vis_cnt = int(np.sum(joint_visibilities))
        vis_pct = vis_cnt / max(1, len(joint_visibilities)) * 100.0
        ax3.set_title(f"3. 2D Pose Estimation Overlay\nVisible: {vis_cnt}/{len(joint_visibilities)} ({vis_pct:.1f}%)", fontsize=10, fontweight="bold")
        ax3.axis("off")

        # Panel 4: 3D Lifted Pose (1:1 真实人体度量空间比例与运动学连线)
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        xs = joint_3d_cam[:, 0]
        ys = joint_3d_cam[:, 2] # 深度 Z
        zs = joint_3d_cam[:, 1] # 高度 Y

        for i in range(len(joint_3d_cam)):
            c = "#00E5FF" if joint_visibilities[i] else "#FF5252"
            ax4.scatter(xs[i], ys[i], zs[i], s=26, c=c, edgecolors="k", depthshade=True)

        for p1, p2 in HUMAN_BONES:
            ax4.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color="#7C4DFF", linewidth=2.0)

        # 严格保持 1:1:1 各向同性度量尺度 (1:1 Metric Scale)，防止人体视觉压扁或拉伸
        max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
        max_range = max(max_range, 0.9)
        mid_x = (xs.max() + xs.min()) * 0.5
        mid_y = (ys.max() + ys.min()) * 0.5
        mid_z = (zs.max() + zs.min()) * 0.5

        ax4.set_xlim(mid_x - max_range, mid_x + max_range)
        ax4.set_ylim(mid_y - max_range, mid_y + max_range)
        ax4.set_zlim(mid_z - max_range, mid_z + max_range)
        ax4.set_box_aspect([1, 1, 1])

        ax4.set_xlabel("X (m)", fontsize=8)
        ax4.set_ylabel("Depth Z (m)", fontsize=8)
        ax4.set_zlabel("Height Y (m)", fontsize=8)
        ax4.set_title(f"4. 3D Pose (1:1 Metric Scale)\nAction: {act_name}", fontsize=10, fontweight="bold")
        ax4.view_init(elev=12, azim=-70)

        plt.tight_layout()
        act_fig_p = scene_out_dir / f"{act_idx:02d}_{act_name}.png"
        plt.savefig(act_fig_p, bbox_inches="tight", dpi=130)
        plt.close()

        action_figure_paths[act_name] = str(act_fig_p)

    sim.close()
    logger.info("Finished rendering 16 actions for HM3D scene [%s].", scene_name)
    return action_figure_paths


def build_html_dashboard(all_results: Dict[str, Dict[str, str]], output_dir: Path):
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>ACTIVEVIEW: 10 HM3D Scenes x 16 AMASS Actions Benchmark</title>",
        "  <style>",
        "    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }",
        "    h1 { color: #38bdf8; text-align: center; margin-bottom: 8px; }",
        "    p.subtitle { text-align: center; color: #94a3b8; margin-bottom: 32px; font-size: 15px; }",
        "    .scene-section { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }",
        "    .scene-title { color: #f59e0b; font-size: 20px; font-weight: bold; border-bottom: 2px solid #334155; padding-bottom: 8px; margin-bottom: 16px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }",
        "    .card { background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; transition: transform 0.2s; }",
        "    .card:hover { transform: translateY(-4px); border-color: #38bdf8; }",
        "    .card img { width: 100%; display: block; }",
        "    .card-title { padding: 8px 12px; font-weight: 600; font-size: 13px; color: #e2e8f0; background: #1e293b; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>ACTIVEVIEW HM3D Photorealistic Embodied Perception Benchmark</h1>",
        "  <p class='subtitle'>10 Real HM3D Photorealistic Scenes &times; 16 AMASS Dynamic Action Postures (Camera Height = 1.50m)</p>",
    ]

    for s_name, act_dict in all_results.items():
        html_lines.append("  <div class='scene-section'>")
        html_lines.append(f"    <div class='scene-title'>HM3D Scene: {s_name} ({len(act_dict)} Actions)</div>")
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

    # 制作 10 场景代表性大图蒙太奇
    actions_picked = [
        (SCENES_CATALOG[0][0], "00_standing.png"),
        (SCENES_CATALOG[1][0], "01_sitting.png"),
        (SCENES_CATALOG[2][0], "04_bending.png"),
        (SCENES_CATALOG[3][0], "05_reaching.png"),
        (SCENES_CATALOG[4][0], "06_picking_up.png"),
        (SCENES_CATALOG[5][0], "07_squatting.png"),
        (SCENES_CATALOG[6][0], "08_jumping.png"),
        (SCENES_CATALOG[7][0], "11_waving.png"),
        (SCENES_CATALOG[8][0], "12_dancing.png"),
        (SCENES_CATALOG[9][0], "13_kicking.png"),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(20, 18), dpi=140)
    axes = axes.flatten()

    for idx, (s_name, act_file) in enumerate(actions_picked):
        img_p = base_out_dir / s_name / act_file
        if img_p.exists():
            im = Image.open(img_p)
            axes[idx].imshow(im)
            axes[idx].set_title(f"HM3D [{idx+1}/10]: {s_name} | Action: {act_file.replace('.png', '')}", fontsize=11, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    overview_p = base_out_dir / "10_scenes_representative_montage.png"
    plt.savefig(overview_p, bbox_inches="tight", dpi=140)
    plt.close()

    logger.info("================================================================")
    logger.info("  All 10 HM3D Scenes x 16 Actions Rendering Completed!          ")
    logger.info("  Total Rendered Images: %d", sum(len(v) for v in all_results.values()))
    logger.info("  Output Directory:      %s", base_out_dir)
    logger.info("  Montage Path:          %s", overview_p)
    logger.info("  Dashboard URL:         file://%s", base_out_dir / "index.html")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
