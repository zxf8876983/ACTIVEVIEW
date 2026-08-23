#!/usr/bin/env python3
"""
Pure Solid Background Humanoid Action 3D Pose Extraction with VideoPose3D —— render_scenefree_actions_videopose3d.py
===================================================================================================================

职责：
    1. 在 100% 纯色背景（Pure Solid Black Background, scene_id="NONE"，零场景网格）下渲染 SMPL 仿真人体；
    2. 正面正对人体（Frontal View），精准居中人体完整视野（从头顶到脚底全覆盖）；
    3. 遍历 16 种非位移（Non-Locomotion）AMASS 日常与应急动作；
    4. 运行纯视觉 2D 姿态检测模型 (Keypoint R-CNN on CUDA GPU)；
    5. 使用学术界成熟的 3D 姿态模型 (VideoPose3D on CUDA GPU) 端到端提取规范 3D 骨架 (Human3.6M 17 关节)；
    6. 生成 16 种动作的高清 4-Panel 可视化图片、16 动作全景拼图与交互式 Web Gallery。
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F

# 配置路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEOPOSE_DIR = str(REPO_ROOT / "tools" / "videopose3d")
if VIDEOPOSE_DIR not in sys.path:
    sys.path.append(VIDEOPOSE_DIR)

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scenefree_videopose3d")

# 资源路径
URDF_PATH = "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf"
OUTPUT_DIR = Path("/home/zxf/WorkSpace/code/data/ActiveView/visualizations/scenefree_videopose3d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 16 种标准 AMASS 动作分类
ACTION_CATEGORIES = [
    "standing", "sitting", "sit_down", "stand_up",
    "bending", "reaching", "picking_up", "squatting",
    "jumping", "turning", "stretching", "waving",
    "dancing", "kicking", "fall_stumble", "placing"
]

# COCO-17 2D 骨骼连接
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Human3.6M 17 关节 3D 骨骼运动学树 (Parent -> Child)
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),           # 右下肢 (Right Leg: Hip -> Knee -> Ankle)
    (0, 4), (4, 5), (5, 6),           # 左下肢 (Left Leg: Hip -> Knee -> Ankle)
    (0, 7), (7, 8), (8, 9), (9, 10),  # 躯干脊柱与头颈 (Spine -> Thorax -> Neck -> Head)
    (8, 11), (11, 12), (12, 13),      # 左上肢 (Left Arm: Shoulder -> Elbow -> Wrist)
    (8, 14), (14, 15), (15, 16)       # 右上肢 (Right Arm: Shoulder -> Elbow -> Wrist)
]


def make_quat(rot_axis: List[float], angle_deg: float) -> np.ndarray:
    """构造四元数旋转 [x, y, z, w] 供 Habitat URDF 关节调用。"""
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


def run_scenefree_videopose3d_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing models on device: %s...", device)

    # 1. 2D 姿态检测模型 (Keypoint R-CNN on GPU)
    pose_2d_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    pose_2d_model.eval()

    # 2. 3D 姿态提取模型 (VideoPose3D on GPU)
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
    logger.info("Keypoint R-CNN and VideoPose3D loaded successfully!")

    # 3. 初始化 100% 纯色单色背景 (scene_id = 'NONE', 零网格加载)
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True

    H, W = 512, 512
    hfov = 50.0

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
    aom = sim.get_articulated_object_manager()
    art_obj = aom.add_articulated_object_from_urdf(URDF_PATH)

    action_figure_paths = {}

    logger.info(">>> Starting Pure Solid Background Rendering & 3D Pose Extraction across 16 Actions...")

    for act_idx, act_name in enumerate(ACTION_CATEGORIES):
        # 1. 设置动作关节姿态
        action_joints = get_action_joint_positions(act_name)
        art_obj.joint_positions = action_joints

        # 2. 动态贴地与中心高度计算
        art_obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        min_link_y = min(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))
        max_link_y = max(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))
        grounded_y = -min_link_y
        art_obj.translation = np.array([0.0, grounded_y, 0.0], dtype=np.float32)

        human_center_y = grounded_y + (max_link_y - min_link_y) * 0.50

        # 3. 正面正视机位 (Frontal View, 精准居中对齐)
        cam_pos = np.array([0.0, human_center_y - 0.75, 2.6], dtype=np.float32)
        target_pos = np.array([0.0, human_center_y, 0.0], dtype=np.float32)
        dir_vec = target_pos - (cam_pos + np.array([0.0, 0.75, 0.0]))
        dir_norm = dir_vec / np.linalg.norm(dir_vec)
        yaw = np.arctan2(-dir_norm[0], -dir_norm[2])
        pitch = np.arcsin(dir_norm[1])
        cam_rot = quaternion.from_rotation_vector([0, yaw, 0]) * quaternion.from_rotation_vector([pitch, 0, 0])

        agent_state = habitat_sim.AgentState()
        agent_state.position = cam_pos
        agent_state.rotation = cam_rot
        sim.get_agent(0).set_state(agent_state)

        # 4. 采集传感器观测 (100% 纯色黑底)
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        depth = obs["depth_sensor"]

        # 5. 纯视觉 2D 关键点检测 (Keypoint R-CNN on GPU)
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

        # 6. 成熟 3D 姿态模型端到端提取 (VideoPose3D on GPU)
        kpts_norm = normalize_screen_coordinates(kpts_2d_coco, w=W, h=H)
        kpts_seq = np.repeat(kpts_norm[np.newaxis, np.newaxis, :, :], 243, axis=1)  # (1, 243, 17, 2)
        kpts_seq_t = torch.from_numpy(kpts_seq).float().to(device)

        with torch.no_grad():
            out_3d_t = videopose_model(kpts_seq_t)  # (1, 1, 17, 3)

        skel_3d_h36m = out_3d_t[0, 0].cpu().numpy()  # (17, 3) [x, y, z]

        # 7. 渲染 4-Panel 图像
        fig = plt.figure(figsize=(18, 5), dpi=130)

        # Panel 1: Pure Solid Background RGB (Frontal View)
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"1. Pure Solid Background (RGB)\nFrontal View | Action: {act_name}", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Depth Sensor
        ax2 = fig.add_subplot(1, 4, 2)
        valid_depth = np.where(depth > 10.0, np.nan, depth)
        im2 = ax2.imshow(valid_depth, cmap="viridis")
        ax2.set_title(f"2. Pure Human DEPTH_SENSOR\nMetric Depth [{depth[depth>0].min() if np.any(depth>0) else 0:.2f}m, {depth.max():.2f}m]", fontsize=10, fontweight="bold")
        ax2.axis("off")
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (m)", fontsize=8)

        # Panel 3: 2D Keypoint Prediction (Keypoint R-CNN)
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(rgb)
        for u_f, v_f, conf in kpts_2d_raw:
            if u_f > 0 or v_f > 0:
                c = "#00FF66" if conf >= 0.50 else "#FFCC00"
                ax3.scatter(u_f, v_f, s=26, c=c, edgecolors="white", linewidth=0.6, zorder=5)

        for p1, p2 in COCO_BONES:
            u1, v1, _ = kpts_2d_raw[p1]
            u2, v2, _ = kpts_2d_raw[p2]
            if (u1 > 0 or v1 > 0) and (u2 > 0 or v2 > 0):
                ax3.plot([u1, u2], [v1, v2], color="#00E5FF", linewidth=1.8, alpha=0.85, zorder=4)

        ax3.set_title(f"3. Keypoint R-CNN (2D Pose)\nPerson Score: {person_score:.2f}", fontsize=10, fontweight="bold")
        ax3.axis("off")

        # Panel 4: VideoPose3D Canonical 3D Skeleton
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        xs = skel_3d_h36m[:, 0]
        ys = skel_3d_h36m[:, 2]   # 深度 Z
        zs = -skel_3d_h36m[:, 1]  # 高度 Y (H36M 坐标系 Y 轴向下，取负向上)

        for i in range(17):
            ax4.scatter(xs[i], ys[i], zs[i], s=26, c="#00E5FF", edgecolors="k", depthshade=True)

        for p1, p2 in H36M_BONES:
            ax4.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], color="#7C4DFF", linewidth=2.0)

        # 严格保持 1:1:1 各向同性度量尺度 (1:1 Metric Scale)
        max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
        max_range = max(max_range, 0.6)
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
        ax4.set_title(f"4. VideoPose3D (Canonical 3D Pose)\nAction: {act_name}", fontsize=10, fontweight="bold")
        ax4.view_init(elev=12, azim=-70)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / f"{act_idx:02d}_{act_name}.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=130)
        plt.close()

        action_figure_paths[act_name] = str(fig_path)
        logger.info("  [%02d/16] Processed %s -> %s", act_idx + 1, act_name, fig_path.name)

    sim.close()

    # 8. 生成 16 动作总览大拼图 (16-Action Overview Montage)
    build_16_actions_montage(action_figure_paths, OUTPUT_DIR / "16_actions_scenefree_montage.png")

    # 9. 生成 HTML Dashboard
    build_scenefree_html(action_figure_paths, OUTPUT_DIR / "index.html")

    logger.info("================================================================")
    logger.info("  Pure Solid Background 16-Action VideoPose3D Completed!")
    logger.info("  Total Rendered Images: %d", len(action_figure_paths))
    logger.info("  Output Directory:      %s", OUTPUT_DIR)
    logger.info("  Montage Path:          %s", OUTPUT_DIR / "16_actions_scenefree_montage.png")
    logger.info("  Dashboard URL:         file://%s", OUTPUT_DIR / "index.html")
    logger.info("================================================================")


def build_16_actions_montage(action_figure_paths: Dict[str, str], montage_out_path: Path):
    """合成 16 动作 8 行 2 列的高清总览大图。"""
    fig, axes = plt.subplots(8, 2, figsize=(26, 36), dpi=120)
    axes_flat = axes.flatten()

    for idx, act_name in enumerate(ACTION_CATEGORIES):
        ax = axes_flat[idx]
        img_p = action_figure_paths.get(act_name)
        if img_p and Path(img_p).exists():
            im = plt.imread(img_p)
            ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"Action [{idx:02d}/15]: {act_name}", fontsize=14, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(montage_out_path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("Saved 16-Action Overview Montage to: %s", montage_out_path)


def build_scenefree_html(action_figure_paths: Dict[str, str], html_path: Path):
    """构建交互式 Web 评测画廊。"""
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>ACTIVEVIEW: Pure Solid Background 16 Actions VideoPose3D Benchmark</title>",
        "  <style>",
        "    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }",
        "    h1 { color: #38bdf8; text-align: center; margin-bottom: 8px; }",
        "    p.subtitle { text-align: center; color: #94a3b8; margin-bottom: 32px; font-size: 15px; }",
        "    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(580px, 1fr)); gap: 24px; max-width: 1800px; margin: 0 auto; }",
        "    .card { background: #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); border: 1px solid #334155; transition: transform 0.2s; }",
        "    .card:hover { transform: translateY(-4px); border-color: #38bdf8; }",
        "    .card-header { padding: 12px 16px; background: #0f172a; border-bottom: 1px solid #334155; font-weight: bold; font-size: 15px; color: #38bdf8; }",
        "    .card img { width: 100%; display: block; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>ACTIVEVIEW: Pure Solid Background 16 Actions VideoPose3D Benchmark</h1>",
        "  <p class='subtitle'>100% Solid Black Background (scene_id='NONE') &bull; Frontal View &bull; Keypoint R-CNN (2D) &bull; VideoPose3D (3D HPE)</p>",
        "  <div class='gallery'>"
    ]

    for idx, act_name in enumerate(ACTION_CATEGORIES):
        rel_p = f"{idx:02d}_{act_name}.png"
        html_lines.append(f"    <div class='card'>")
        html_lines.append(f"      <div class='card-header'>Action {idx:02d}: {act_name}</div>")
        html_lines.append(f"      <img src='{rel_p}' alt='{act_name}'>")
        html_lines.append(f"    </div>")

    html_lines.extend([
        "  </div>",
        "</body>",
        "</html>"
    ])

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    logger.info("Saved Scene-Free HTML Dashboard to: %s", html_path)


if __name__ == "__main__":
    run_scenefree_videopose3d_benchmark()
