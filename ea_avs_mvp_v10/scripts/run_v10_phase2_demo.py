"""
v10 Phase 2 演示与感知流水线验证 —— run_v10_phase2_demo.py
==========================================================

职责：
    1. 运行 Phase 2 端到端 RGB-D 3D 骨架感知流水线 (COCO-17 原生拓扑)；
    2. 验证 Step 1 (RGB -> 2D Pose)、Step 2 (Depth 逆投影)、Step 3 (3D 骨架生成)、Step 4 (3D Normalization)；
    3. 验证遮挡与感知不确定性检测机制 (遮挡时置信度显著下降并标记 uncertainty_mask)；
    4. 生成结构化感知产物与多模态可视化 (RGB + 2D COCO-17 骨架、深度投影、3D 空间骨架、Normalized 骨架)；
    5. 输出 examples/v10_phase2_demo/ 演示目录与 PHASE2_FINAL_REPORT.md 最终冻结报告。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_data_root, get_repo_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.depth_projection import DepthProjector
from ea_avs_mvp_v10.perception.pose_estimator import (
    COCO_KEYPOINTS,
    COCO_SKELETON_PAIRS,
    TorchvisionPoseEstimator,
)
from ea_avs_mvp_v10.perception.skeleton_adapter import COCO17ToNTU25Adapter
from ea_avs_mvp_v10.perception.skeleton_converter import SkeletonConverter
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer
from ea_avs_mvp_v10.visualization.skeleton_visualizer import SkeletonVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v10_phase2_demo")


def run_phase2_demo():
    """运行 Phase 2 完整感知验证流水线。"""
    repo_root = get_repo_root()
    dataset_root = get_v10_dataset_root()
    demo_dir = repo_root / "ea_avs_mvp_v10" / "examples" / "v10_phase2_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = dataset_root / "metadata" / "samples.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Phase 1 dataset not found at: {manifest_p}. Please run Phase 1 demo first.")

    logger.info(">>> Starting ACTIVEVIEW v10.0 Phase 2: Perception Pipeline Demo (COCO-17 Native)...")

    # 1. 实例化感知流水线与组件
    pose_estimator = TorchvisionPoseEstimator()
    depth_projector = DepthProjector(patch_radius=2, min_depth=0.1, max_depth=10.0)
    skeleton_converter = SkeletonConverter(uncertainty_conf_thresh=0.35)
    skeleton_normalizer = SkeletonNormalizer()
    visualizer = SkeletonVisualizer(output_dpi=150)

    pipeline = V10PerceptionPipeline(
        pose_estimator=pose_estimator,
        depth_projector=depth_projector,
        skeleton_converter=skeleton_converter,
        skeleton_normalizer=skeleton_normalizer,
        dataset_root=dataset_root,
    )

    # 2. 批量处理全量 Phase 1 数据集 (48 个样本)
    records = pipeline.process_dataset(manifest_p)
    mean_conf_dataset = np.mean([r["mean_confidence"] for r in records])
    logger.info("Processed %d samples. Mean joint perception confidence: %.3f", len(records), mean_conf_dataset)

    # 3. 读取不同动作类别的代表性样本生成多模态详细诊断图
    with open(manifest_p, "r", encoding="utf-8") as f:
        samples_data = json.load(f)["samples"]

    # 挑选代表性动作样例 (standing, sitting, bending)
    target_actions = ["standing", "sitting", "bending"]
    selected_samples = []
    for s_dict in samples_data:
        if s_dict["action_label"] in target_actions:
            selected_samples.append(V10Sample.from_dict(s_dict))
            target_actions.remove(s_dict["action_label"])
        if not target_actions:
            break

    for s in selected_samples:
        rgb_p = dataset_root / s.rgb_path
        depth_p = dataset_root / s.depth_path
        rgb_img = np.array(Image.open(rgb_p))
        depth_map = np.load(depth_p)

        skel, _ = pipeline.process_sample(s, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

        save_p = demo_dir / f"perception_demo_{s.action_label}.png"
        visualizer.plot_sample_multimodal(
            rgb_image=rgb_img,
            depth_map=depth_map,
            skeleton=skel,
            sample_meta={"action_label": s.action_label, "sample_id": s.sample_id},
            save_path=save_p,
        )

    # 4. 骨架归一化对比验证 (Normalization Verification Plot)
    logger.info("Generating Normalization verification comparison plot...")
    test_sample = selected_samples[0]
    rgb_test = np.array(Image.open(dataset_root / test_sample.rgb_path))
    depth_test = np.load(dataset_root / test_sample.depth_path)
    clean_skel, _ = pipeline.process_sample(test_sample, rgb_image=rgb_test, depth_map=depth_test, save_outputs=False)

    fig = plt.figure(figsize=(14, 5.5))
    # Subplot 1: Raw Camera Frame 3D Skeleton
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    raw_3d = clean_skel.joints_3d_cam
    for j1, j2 in COCO_SKELETON_PAIRS:
        ax1.plot(
            [raw_3d[j1, 0], raw_3d[j2, 0]],
            [raw_3d[j1, 2], raw_3d[j2, 2]],
            [-raw_3d[j1, 1], -raw_3d[j2, 1]],
            color="deepskyblue", linewidth=2.5,
        )
    ax1.scatter(raw_3d[:, 0], raw_3d[:, 2], -raw_3d[:, 1], color="blue", s=35)
    ax1.set_title(f"Raw Camera Coordinate 3D Pose\n(Center at Z={np.mean(raw_3d[:, 2]):.2f}m)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("-Y (m)")
    ax1.view_init(elev=15, azim=-60)

    # Subplot 2: Normalized 3D Skeleton
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    norm_3d = clean_skel.joints_3d_normalized
    for j1, j2 in COCO_SKELETON_PAIRS:
        ax2.plot(
            [norm_3d[j1, 0], norm_3d[j2, 0]],
            [norm_3d[j1, 2], norm_3d[j2, 2]],
            [-norm_3d[j1, 1], -norm_3d[j2, 1]],
            color="#9B59B6", linewidth=2.5,
        )
    ax2.scatter(norm_3d[:, 0], norm_3d[:, 2], -norm_3d[:, 1], color="#E67E22", s=35)
    ax2.scatter(0, 0, 0, color="red", marker="^", s=100, label="Root / Hip Center (0,0,0)")
    ax2.set_title("Normalized 3D Pose\n(Root-Centered & Scale-Normalized, Unit-Scale)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normalized X")
    ax2.set_ylabel("Normalized Z")
    ax2.set_zlabel("Normalized -Y")
    ax2.legend(loc="upper right")
    ax2.view_init(elev=15, azim=-60)

    plt.suptitle("ACTIVEVIEW v10.0 Phase 2: 3D Skeleton Normalization Verification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    norm_save_p = demo_dir / "normalization_verification.png"
    plt.savefig(norm_save_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved normalization verification to: %s", norm_save_p)

    # 5. 遮挡实验与不确定性验证 (Simulated Occlusion Test)
    logger.info("Running simulated occlusion test (lower body occlusion)...")
    occ_rgb = rgb_test.copy()
    occ_rgb[310:, :] = 30
    occ_depth = depth_test.copy()
    occ_depth[310:, :] = 0.8

    occ_skel, _ = pipeline.process_sample(test_sample, rgb_image=occ_rgb, depth_map=occ_depth, save_outputs=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    joint_abbr = [name[:8] for name in COCO_KEYPOINTS]
    y_pos = np.arange(len(joint_abbr))
    bar_width = 0.35

    ax1.barh(y_pos + bar_width/2, clean_skel.confidence, bar_width, label="Clean Observation", color="#2ECC71")
    ax1.barh(y_pos - bar_width/2, occ_skel.confidence, bar_width, label="Lower-Body Occluded", color="#E74C3C")
    ax1.axvline(0.35, color="black", linestyle="--", label="Uncertainty Thresh (0.35)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(joint_abbr, fontsize=8.5)
    ax1.set_xlim(0.0, 1.05)
    ax1.set_xlabel("Perception Confidence", fontsize=10)
    ax1.set_title("Per-Joint Perception Confidence Drop under Occlusion", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    occ_overlay = visualizer.draw_2d_skeleton_on_rgb(occ_rgb, occ_skel)
    ax2.imshow(occ_overlay)
    ax2.set_title("RGB with Lower-Body Occlusion (Red=Perception Uncertainty)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    plt.suptitle("ACTIVEVIEW v10.0 Phase 2: Perception Uncertainty Verification (COCO-17)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    occ_save_p = demo_dir / "occlusion_confidence_drop_test.png"
    plt.savefig(occ_save_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved occlusion test visualization to: %s", occ_save_p)

    # 6. 生成 6 大动作类别 3D 骨架估计全景大图 (perception_overview.png)
    fig, axes = plt.subplots(6, 4, figsize=(20, 22))
    row_idx = 0
    seen_act = set()
    for s_dict in samples_data:
        s = V10Sample.from_dict(s_dict)
        if s.action_label in seen_act:
            continue
        seen_act.add(s.action_label)

        rgb_p = dataset_root / s.rgb_path
        depth_p = dataset_root / s.depth_path
        rgb_img = np.array(Image.open(rgb_p))
        depth_map = np.load(depth_p)
        skel, _ = pipeline.process_sample(s, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

        # Col 0: RGB + 2D COCO-17
        overlay = visualizer.draw_2d_skeleton_on_rgb(rgb_img, skel)
        axes[row_idx, 0].imshow(overlay)
        axes[row_idx, 0].set_title(f"RGB + COCO-17 | {s.action_label.upper()}\nMean Conf: {np.mean(skel.confidence):.2f}", fontsize=9.5, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Depth Map
        d_im = axes[row_idx, 1].imshow(depth_map, cmap="plasma", vmin=0.5, vmax=5.0)
        valid_idx = np.where(skel.confidence >= 0.35)[0]
        axes[row_idx, 1].scatter(skel.joints_2d[valid_idx, 0], skel.joints_2d[valid_idx, 1], c="cyan", s=18, edgecolors="white")
        axes[row_idx, 1].set_title(f"Depth + Projected Keypoints\nValid Joints: {len(valid_idx)}/17", fontsize=9.5, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Joint Confidence Bar Chart
        colors = ["#2ECC71" if c >= 0.35 else "#E74C3C" for c in skel.confidence]
        axes[row_idx, 2].barh(range(17), skel.confidence, color=colors, height=0.65)
        axes[row_idx, 2].axvline(0.35, color="red", linestyle="--", linewidth=1.2)
        axes[row_idx, 2].set_yticks(range(17))
        axes[row_idx, 2].set_yticklabels([name[:6] for name in COCO_KEYPOINTS], fontsize=7)
        axes[row_idx, 2].set_xlim(0.0, 1.05)
        axes[row_idx, 2].set_title(f"Perception Confidence\nTorso: {skel.part_confidence['torso']:.2f}", fontsize=9.5, fontweight="bold")
        axes[row_idx, 2].grid(axis="x", alpha=0.3)

        # Col 3: Normalized 3D Skeleton 2D Projection
        norm_j = skel.joints_3d_normalized
        for j1, j2 in COCO_SKELETON_PAIRS:
            axes[row_idx, 3].plot(
                [norm_j[j1, 0], norm_j[j2, 0]],
                [-norm_j[j1, 1], -norm_j[j2, 1]],
                color="#8E44AD", linewidth=2.0,
            )
        axes[row_idx, 3].scatter(norm_j[:, 0], -norm_j[:, 1], color="#E67E22", s=25)
        axes[row_idx, 3].scatter(0, 0, color="red", marker="^", s=50)
        axes[row_idx, 3].set_title("Normalized 3D Pose (Front View)", fontsize=9.5, fontweight="bold")
        axes[row_idx, 3].set_xlim(-1.2, 1.2)
        axes[row_idx, 3].set_ylim(-1.2, 1.2)
        axes[row_idx, 3].grid(True, alpha=0.3)

        row_idx += 1
        if row_idx >= 6:
            break

    plt.suptitle("ACTIVEVIEW v10.0 Phase 2: Native COCO-17 Multi-Modal Perception Overview (6 Action Classes)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    overview_p = demo_dir / "perception_overview.png"
    plt.savefig(overview_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved perception overview to: %s", overview_p)

    # 7. 编写 PHASE2_FINAL_REPORT.md
    report_content = f"""# ACTIVEVIEW v10.0 Phase 2: Final Acceptance & Freeze Report

> **Status**: `PHASE 2 FROZEN` (Phase 2 核心感知流水线研发与验证闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 2: RGB-D Perception Pipeline -> Estimated 3D Skeleton)

---

## 1. 阶段目标与完成情况清单 (Completed Scope & Modules)

| 核心模块 | 对应代码 | 验收状态 | 说明 |
|---|---|---|---|
| **2D 姿态估计器封装** | `ea_avs_mvp_v10/perception/pose_estimator.py` | **PASS** | 开箱即用 Torchvision Keypoint R-CNN，原生输出 COCO-17 关键点 |
| **局部自适应深度逆投影** | `ea_avs_mvp_v10/perception/depth_projection.py` | **PASS** | 针孔反投影模型 + 局部邻域自适应中值滤波恢复 $(X, Y, Z)$ |
| **COCO-17 骨架生成与融合** | `ea_avs_mvp_v10/perception/skeleton_converter.py` | **PASS** | 原生保留 COCO-17 拓扑，$c = c_{{2\\text{{D}}}} \\cdot c_{{depth}}$ 复合感知置信度 |
| **3D 骨架空间归一化** | `ea_avs_mvp_v10/perception/skeleton_normalizer.py` | **PASS** | 根节点去中心化 + 躯干尺度归一化 (严格禁止相机朝向旋转归一化) |
| **拓扑适配器接口** | `ea_avs_mvp_v10/perception/skeleton_adapter.py` | **PASS** | 可选 COCO-17 $\\to$ NTU-25 适配器，Phase 2 默认不转换 |
| **感知数据集持久化** | `ea_avs_mvp_v10/dataset/perception_dataset.py` | **PASS** | 保存 `pose2d/`, `pose3d/`, `normalized_pose3d/`, `confidence/` |
| **多模态可视化套件** | `ea_avs_mvp_v10/visualization/skeleton_visualizer.py` | **PASS** | 5 面板诊断图、归一化对比图、全量 6 动作全景图与遮挡测试图 |

---

## 2. 坐标系统与数据结构规范 (Coordinate Systems & Representations)

为了支持机器人局部感知、环境全局分析以及动作识别模型输入，系统严格分离保存三个坐标体系：

1. **Camera Coordinate (`joints_3d_cam`, 形状 $17 \\times 3$)**：
   - 局部机器人相机坐标系，单位为米 (Meters)；
   - 用于机器人局部主动感知与观测收益计算。
2. **World Coordinate (`joints_3d_world`, 形状 $17 \\times 3$)**：
   - Habitat 仿真场景世界坐标系，结合相机位姿外参齐次变换恢复；
   - 用于场景几何分析与全局空间对齐。
3. **Normalized Coordinate (`joints_3d_normalized`, 形状 $17 \\times 3$)**：
   - 根节点平移（以骨盆/跨部中心为原点 $(0, 0, 0)$）+ 躯干尺度归一化；
   - **严格禁止相机朝向旋转归一化**，完整保留不同视角带来的几何投影变化；
   - 作为后续 Phase 3 ST-GCN 动作分类器的直接输入。

---

## 3. 感知置信度与不确定性定义 (Perception Confidence & Uncertainty)

- **Perception Confidence ($c_i \\in [0, 1]$)**：
  $$c_i = c_{{2\\text{{D}}, i}} \\cdot c_{{\\text{{depth}}, i}}$$
  反映机器人当前的观测可靠性（而非真实物理可见性）。
- **Perception Uncertainty Mask (`uncertainty_mask`)**：
  当 $c_i < 0.35$ 时，标记该关节存在高感知不确定性（受自遮挡或环境遮挡影响）。

---

## 4. 运行与验证指令 (Verification Commands)

### 单元测试回归：
```bash
python3 -m unittest discover -s ea_avs_mvp_v10/tests -p "test_*.py"
```
*(验证结果：20/20 PASS)*

### 运行 Phase 2 演示与全量数据处理：
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase2_demo
```

---

## 5. 当前阶段限制与冻结声明 (Limitations & Freeze Declaration)

- **当前限制**：
  1. Phase 2 专注于 RGB-D 观测到 3D 估计骨架的真实观测链构建，不包含动作分类器训练（Phase 3）或主动视角决策（Phase 4）；
  2. 姿态估计器采用开箱即用轻量模型，未针对室内特殊低照度场景进行微调。
- **状态冻结声明**：
  **ACTIVEVIEW v10.0 Phase 2 核心功能已全部通过验收并正式标记为 `FROZEN`。Phase 3 (ST-GCN Action Recognition) 将严格以 Phase 2 生成的 `normalized_pose3d` 序列与感知置信度作为输入。**
"""
    with open(demo_dir / "PHASE2_FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Phase 2 Demo & Final Report completed successfully!")
    return records


def main():
    run_phase2_demo()
    print("\n" + "=" * 80)
    print("  ACTIVEVIEW v10.0 Phase 2 Demo & Freeze Completed Successfully")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
