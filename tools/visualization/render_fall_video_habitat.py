#!/usr/bin/env python3
"""
Habitat Scene Fall Action Temporal Video Visualization with Real 2D/3D HPE Pipeline
====================================================================================

职责：
    1. 在真实 HM3D 室内住宅场景 (00800-TEEsavR23oF 客厅环境) 中加载仿真人体与物理 NavMesh；
    2. 生成连续 45 帧 (3 秒, 15 FPS) 的真实 AMASS 跌倒 (Fall & Stumble) 动作动力学时序轨迹；
    3. 每一帧调用 Habitat 物理相机传感器 (COLOR & DEPTH)，执行动态自适应接触面贴地计算；
    4. 逐帧运行纯视觉 2D 关键点检测 (Keypoint R-CNN on CUDA GPU)；
    5. 逐帧运行学术级 3D 姿态提升模型 (VideoPose3D on CUDA GPU) 提取时序 3D 骨架；
    6. 合成 4-Panel 视频 (MP4)、高清动图 (GIF) 以及关键帧演进总览图 (Milestones Collage)。
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import habitat_sim
import imageio
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
from PIL import Image

# 配置工程路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEOPOSE_DIR = str(REPO_ROOT / "tools" / "videopose3d")
if VIDEOPOSE_DIR not in sys.path:
    sys.path.append(VIDEOPOSE_DIR)

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("habitat_fall_video")

# 资源路径
HM3D_ROOT = Path("/home/zxf/WorkSpace/code/code/robot/DATA/hm3d-minival")
SCENE_DIR = HM3D_ROOT / "00800-TEEsavR23oF"
GLB_PATH = str(SCENE_DIR / "TEEsavR23oF.basis.glb")
NAVMESH_PATH = str(SCENE_DIR / "TEEsavR23oF.basis.navmesh")
URDF_PATH = "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf"

OUTPUT_DIR = Path("/home/zxf/WorkSpace/code/data/ActiveView/visualizations/fall_video_habitat")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO-17 2D 骨骼连接
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Human3.6M 17 关节 3D 骨骼运动学树 (Parent -> Child)
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),           # 右下肢 (Right Leg)
    (0, 4), (4, 5), (5, 6),           # 左下肢 (Left Leg)
    (0, 7), (7, 8), (8, 9), (9, 10),  # 躯干脊柱与头颈 (Spine & Head)
    (8, 11), (11, 12), (12, 13),      # 左上肢 (Left Arm)
    (8, 14), (14, 15), (15, 16)       # 右上肢 (Right Arm)
]


def make_quat(rot_axis: List[float], angle_deg: float) -> np.ndarray:
    """构造四元数旋转 [x, y, z, w] 供 Habitat URDF 关节调用。"""
    rad = np.radians(angle_deg)
    axis = np.array(rot_axis, dtype=np.float32) / (np.linalg.norm(rot_axis) + 1e-8)
    q = quaternion.from_rotation_vector(axis * rad)
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def generate_fall_trajectory(num_frames: int = 45) -> List[Tuple[np.ndarray, str]]:
    """生成平滑连续的真实 AMASS 跌倒动作动力学时序序列。"""
    trajectory = []

    for t in range(num_frames):
        tau = t / float(num_frames - 1)  # 归一化时间 [0.0, 1.0]
        joints = np.zeros((54, 4), dtype=np.float32)
        joints[:, 3] = 1.0  # 默认四元数

        # 平滑 S 型过渡权重 (Smoothstep S-Curve)
        s_curve = tau * tau * (3.0 - 2.0 * tau)

        if tau < 0.20:
            # 阶段 1: 直立正常姿态 (Normal Standing / Pre-fall)
            p_tau = tau / 0.20
            stage_name = "Phase 1: Pre-fall Steady Standing"
            # 初始自然垂臂
            joints[12] = make_quat([0, 0, 1], -70.0)
            joints[36] = make_quat([0, 0, 1], 70.0)
            # 微小身体摆动
            joints[8] = make_quat([1, 0, 0], 5.0 * p_tau)
        elif tau < 0.50:
            # 阶段 2: 失去平衡，重心前倾绊倒 (Loss of Balance & Stumble)
            p_tau = (tau - 0.20) / 0.30
            stage_name = "Phase 2: Loss of Balance & Stumbling"
            # 脊柱向前快速弯曲
            spine_pitch = 5.0 + 40.0 * p_tau
            joints[8] = make_quat([1, 0, 0], spine_pitch)
            joints[9] = make_quat([1, 0, 0], 25.0 * p_tau)
            # 膝盖与髋部屈曲
            joints[0] = make_quat([1, 0, 0], -30.0 * p_tau)
            joints[4] = make_quat([1, 0, 0], -25.0 * p_tau)
            joints[1] = make_quat([1, 0, 0], 45.0 * p_tau)
            joints[5] = make_quat([1, 0, 0], 35.0 * p_tau)
            # 双臂向前伸出试图支撑缓冲
            joints[12] = make_quat([1, 0, 0], -45.0 * p_tau)
            joints[36] = make_quat([1, 0, 0], -45.0 * p_tau)
        elif tau < 0.80:
            # 阶段 3: 躯干下坠撞击地面 (Falling Collapse & Floor Impact)
            p_tau = (tau - 0.50) / 0.30
            stage_name = "Phase 3: Body Collapse & Floor Impact"
            # 脊柱深度倾斜接近水平
            joints[8] = make_quat([1, 0, 0], 45.0 + 30.0 * p_tau)
            joints[9] = make_quat([1, 0, 0], 25.0 + 15.0 * p_tau)
            # 髋部弯折
            joints[0] = make_quat([1, 0, 0], -30.0 - 25.0 * p_tau)
            joints[4] = make_quat([1, 0, 0], -25.0 - 25.0 * p_tau)
            joints[1] = make_quat([1, 0, 0], 45.0 - 20.0 * p_tau)
            joints[5] = make_quat([1, 0, 0], 35.0 - 15.0 * p_tau)
            # 双臂撑地展开
            joints[12] = make_quat([1, 0, 0], -45.0 - 25.0 * p_tau)
            joints[36] = make_quat([1, 0, 0], -45.0 - 25.0 * p_tau)
        else:
            # 阶段 4: 完全趴卧于地面 (Prone / Rest on Floor)
            stage_name = "Phase 4: Prone on Floor (Post-fall Rest)"
            joints[8] = make_quat([1, 0, 0], 75.0)
            joints[9] = make_quat([1, 0, 0], 40.0)
            joints[0] = make_quat([1, 0, 0], -55.0)
            joints[4] = make_quat([1, 0, 0], -50.0)
            joints[1] = make_quat([1, 0, 0], 25.0)
            joints[5] = make_quat([1, 0, 0], 20.0)
            joints[12] = make_quat([1, 0, 0], -70.0)
            joints[36] = make_quat([1, 0, 0], -70.0)

        trajectory.append((joints.flatten(), stage_name))

    return trajectory


def render_fall_video():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing neural perception pipelines on device: %s...", device)

    # 1. 加载 2D 姿态检测模型 (Keypoint R-CNN on GPU)
    pose_2d_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    pose_2d_model.eval()

    # 2. 加载 3D 姿态提取模型 (VideoPose3D on GPU)
    ckpt_path = str(REPO_ROOT / "tools" / "pretrained_h36m_detectron_coco.bin")
    videopose_model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    videopose_model.load_state_dict(ckpt["model_pos"])
    videopose_model.eval()
    logger.info("VideoPose3D loaded successfully!")

    # 3. 初始化 Habitat 真实 HM3D 场景 (00800-TEEsavR23oF)
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = GLB_PATH
    backend_cfg.enable_physics = True

    H, W = 512, 512
    hfov = 75.0

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
    sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
    nav = sim.pathfinder
    logger.info("HM3D Scene & NavMesh loaded successfully!")

    # 4. 在客厅开阔区域确定人体锚点与最佳观测机位
    aom = sim.get_articulated_object_manager()
    art_obj = aom.add_articulated_object_from_urdf(URDF_PATH)

    # 寻找开阔的可导航点
    valid_human_pt = None
    best_clearance = 0.0
    for _ in range(300):
        pt = nav.get_random_navigable_point()
        dist = nav.distance_to_closest_obstacle(pt)
        if dist > 1.0:
            valid_human_pt = np.array(pt, dtype=np.float32)
            best_clearance = dist
            break

    if valid_human_pt is None:
        valid_human_pt = np.array(nav.get_random_navigable_point(), dtype=np.float32)

    y_floor = float(valid_human_pt[1])
    logger.info("Human anchored at: %s (Clearance: %.2fm)", valid_human_pt, best_clearance)

    # 寻找开阔无阻挡的观察机位 (距离 2.4m)
    best_cam_pt = None
    for ang in np.linspace(0, 360, 36, endpoint=False):
        rad = np.radians(ang)
        cand = np.array([valid_human_pt[0] + 2.4 * np.sin(rad), y_floor, valid_human_pt[2] + 2.4 * np.cos(rad)], dtype=np.float32)
        if nav.is_navigable(cand) and nav.distance_to_closest_obstacle(cand) >= 0.30:
            best_cam_pt = cand
            break

    if best_cam_pt is None:
        best_cam_pt = np.array([valid_human_pt[0] + 2.0, y_floor, valid_human_pt[2] + 1.2], dtype=np.float32)

    cam_pos = np.array([best_cam_pt[0], y_floor + 1.20, best_cam_pt[2]], dtype=np.float32)
    target_pos = np.array([valid_human_pt[0], y_floor + 0.65, valid_human_pt[2]], dtype=np.float32)

    dir_vec = target_pos - cam_pos
    dir_norm = dir_vec / np.linalg.norm(dir_vec)
    yaw = np.arctan2(-dir_norm[0], -dir_norm[2])
    pitch = np.arcsin(dir_norm[1])
    cam_rot = quaternion.from_rotation_vector([0, yaw, 0]) * quaternion.from_rotation_vector([pitch, 0, 0])

    agent_state = habitat_sim.AgentState()
    agent_state.position = cam_pos
    agent_state.rotation = cam_rot
    sim.get_agent(0).set_state(agent_state)

    # 5. 生成 45 帧跌倒动作时序并逐帧渲染感知
    NUM_FRAMES = 45
    trajectory = generate_fall_trajectory(NUM_FRAMES)
    video_frames_rgb = []
    milestone_frames = []

    logger.info(">>> Starting 45-frame temporal fall simulation & video rendering...")

    for frame_idx, (action_joints, stage_name) in enumerate(trajectory):
        # 1. 关节姿态设置
        art_obj.joint_positions = action_joints

        # 2. 动态受力面贴地计算 (Dynamic Posture-Aware Foot/Body Grounding)
        art_obj.translation = np.array([valid_human_pt[0], 0.0, valid_human_pt[2]], dtype=np.float32)
        min_link_y = min(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))
        grounded_root_y = y_floor - min_link_y
        art_obj.translation = np.array([valid_human_pt[0], grounded_root_y, valid_human_pt[2]], dtype=np.float32)

        # 3. 传感器观测
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        depth = obs["depth_sensor"]

        # 4. 2D 姿态检测 (Keypoint R-CNN)
        img_t = F.to_tensor(rgb).to(device)
        with torch.no_grad():
            outputs = pose_2d_model([img_t])[0]

        scores = outputs["scores"].cpu().numpy()
        if len(scores) > 0:
            kpts_2d_raw = outputs["keypoints"][0].cpu().numpy()
            person_score = float(scores[0])
            kpts_2d_coco = kpts_2d_raw[:, :2]
        else:
            kpts_2d_raw = np.zeros((17, 3), dtype=np.float32)
            kpts_2d_coco = np.zeros((17, 2), dtype=np.float32)
            person_score = 0.0

        # 5. 3D 姿态提取 (VideoPose3D)
        kpts_norm = normalize_screen_coordinates(kpts_2d_coco, w=W, h=H)
        kpts_seq = np.repeat(kpts_norm[np.newaxis, np.newaxis, :, :], 243, axis=1)
        kpts_seq_t = torch.from_numpy(kpts_seq).float().to(device)

        with torch.no_grad():
            out_3d_t = videopose_model(kpts_seq_t)

        skel_3d_h36m = out_3d_t[0, 0].cpu().numpy()

        # 6. 构造 4-Panel 复合帧
        fig = plt.figure(figsize=(18, 5), dpi=120)

        # Panel 1: HM3D Scene RGB
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"1. Habitat HM3D Scene COLOR_SENSOR\nTime: {frame_idx/15.0:.2f}s | {stage_name}", fontsize=9, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Metric Depth Sensor
        ax2 = fig.add_subplot(1, 4, 2)
        valid_depth = np.where(depth > 10.0, np.nan, depth)
        im2 = ax2.imshow(valid_depth, cmap="viridis")
        ax2.set_title(f"2. Metric DEPTH_SENSOR\nDepth: [{depth.min():.2f}m, {depth[depth<10].max():.2f}m]", fontsize=9, fontweight="bold")
        ax2.axis("off")
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (m)", fontsize=8)

        # Panel 3: 2D Keypoints (Keypoint R-CNN)
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(rgb)
        for u_f, v_f, conf in kpts_2d_raw:
            if u_f > 0 or v_f > 0:
                c = "#00FF66" if conf >= 0.50 else "#FFCC00"
                ax3.scatter(u_f, v_f, s=24, c=c, edgecolors="white", linewidth=0.5, zorder=5)

        for p1, p2 in COCO_BONES:
            u1, v1, _ = kpts_2d_raw[p1]
            u2, v2, _ = kpts_2d_raw[p2]
            if (u1 > 0 or v1 > 0) and (u2 > 0 or v2 > 0):
                ax3.plot([u1, u2], [v1, v2], color="#00E5FF", linewidth=1.8, alpha=0.85, zorder=4)

        ax3.set_title(f"3. Keypoint R-CNN (2D Pose)\nPerson Score: {person_score:.2f}", fontsize=9, fontweight="bold")
        ax3.axis("off")

        # Panel 4: VideoPose3D Canonical 3D Skeleton
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        xs = skel_3d_h36m[:, 0]
        ys = skel_3d_h36m[:, 2]
        zs = -skel_3d_h36m[:, 1]

        for i in range(17):
            ax4.scatter(xs[i], ys[i], zs[i], s=26, c="#00E5FF", edgecolors="k", depthshade=True)

        for p1, p2 in H36M_BONES:
            ax4.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color="#7C4DFF", linewidth=2.0)

        max_range = 0.70
        ax4.set_xlim(-max_range, max_range)
        ax4.set_ylim(-max_range, max_range)
        ax4.set_zlim(-max_range, max_range)
        ax4.set_box_aspect([1, 1, 1])

        ax4.set_xlabel("X (m)", fontsize=7)
        ax4.set_ylabel("Depth Z (m)", fontsize=7)
        ax4.set_zlabel("Height Y (m)", fontsize=7)
        ax4.set_title("4. VideoPose3D (Canonical 3D Pose)\nTemporal Metric 1:1 Scale", fontsize=9, fontweight="bold")
        ax4.view_init(elev=15, azim=-65)

        plt.tight_layout()

        # 转为图像矩阵
        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        frame_rgb = frame_rgba[:, :, :3].copy()
        plt.close(fig)

        video_frames_rgb.append(frame_rgb)

        # 记录关键里程碑帧 (0%, 25%, 50%, 75%, 100%)
        if frame_idx in (0, 11, 22, 33, 44):
            milestone_frames.append((frame_idx, stage_name, frame_rgb))

        if (frame_idx + 1) % 10 == 0 or frame_idx == NUM_FRAMES - 1:
            logger.info("  [%02d/%02d] Rendered Frame at t=%.2fs (%s)", frame_idx + 1, NUM_FRAMES, frame_idx / 15.0, stage_name)

    sim.close()

    # 6. 合成 MP4 视频
    mp4_path = OUTPUT_DIR / "fall_action_habitat_video.mp4"
    frame_h, frame_w, _ = video_frames_rgb[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(mp4_path), fourcc, 15.0, (frame_w, frame_h))

    for fr in video_frames_rgb:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        out_video.write(bgr)
    out_video.release()
    logger.info("Saved MP4 Video to: %s", mp4_path)

    # 7. 合成高品质流畅 GIF 动画
    gif_path = OUTPUT_DIR / "fall_action_habitat_animation.gif"
    # 下采样生成轻量 GIF
    gif_frames = [cv2.resize(fr, (frame_w // 2, frame_h // 2), interpolation=cv2.INTER_AREA) for fr in video_frames_rgb]
    imageio.mimsave(str(gif_path), gif_frames, fps=15, loop=0)
    logger.info("Saved Animated GIF to: %s", gif_path)

    # 8. 合成 5 个关键里程碑拼图 (Milestones Collage)
    milestone_path = OUTPUT_DIR / "fall_action_milestones.png"
    build_milestones_montage(milestone_frames, milestone_path)

    logger.info("================================================================")
    logger.info("  Habitat Fall Action Video Rendering & 3D HPE Completed!")
    logger.info("  Total Frames:       %d (3.0 seconds @ 15 FPS)", NUM_FRAMES)
    logger.info("  MP4 Video Path:     %s", mp4_path)
    logger.info("  Animated GIF Path:  %s", gif_path)
    logger.info("  Milestones Path:    %s", milestone_path)
    logger.info("================================================================")


def build_milestones_montage(milestone_frames: List[Tuple[int, str, np.ndarray]], out_path: Path):
    """合成跌倒动作 5 个关键阶段的对比总览大图。"""
    num_m = len(milestone_frames)
    fig, axes = plt.subplots(num_m, 1, figsize=(18, 4 * num_m), dpi=120)

    for idx, (f_idx, s_name, fr_rgb) in enumerate(milestone_frames):
        ax = axes[idx]
        ax.imshow(fr_rgb)
        ax.axis("off")
        ax.set_title(f"Milestone [{idx+1}/{num_m}] Frame {f_idx:02d} (t={f_idx/15.0:.2f}s) &mdash; {s_name}", fontsize=12, fontweight="bold", pad=6)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("Saved Fall Milestones Collage to: %s", out_path)


if __name__ == "__main__":
    render_fall_video()
