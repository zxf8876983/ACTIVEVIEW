"""
v10 Phase 2 演示与感知流水线验证 —— run_v10_phase2_demo.py
==========================================================

职责：
    1. 运行 Phase 2 端到端 RGB-D 3D 骨架感知流水线；
    2. 验证 Step 1 (RGB -> 2D Pose)、Step 2 (Depth 逆投影)、Step 3 (3D 骨架拓扑融合)；
    3. 验证遮挡检测机制 (遮挡时置信度显著下降并标记 occluded_mask)；
    4. 生成结构化感知产物与多模态可视化 (RGB + 2D 骨架、深度投影、3D 空间骨架)；
    5. 输出 examples/v10_phase2_demo/ 演示目录与文档。
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
from ea_avs_mvp_v10.perception.pose_estimator import TorchvisionPoseEstimator
from ea_avs_mvp_v10.perception.skeleton_converter import SkeletonConverter, UNIFIED_JOINT_NAMES, UNIFIED_SKELETON_PAIRS
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

    logger.info(">>> Starting ACTIVEVIEW v10.0 Phase 2: Perception Pipeline Demo...")

    # 1. 实例化感知流水线
    pose_estimator = TorchvisionPoseEstimator()
    depth_projector = DepthProjector(patch_radius=2, min_depth=0.1, max_depth=10.0)
    skeleton_converter = SkeletonConverter(occlusion_conf_thresh=0.35)
    visualizer = SkeletonVisualizer(output_dpi=150)

    pipeline = V10PerceptionPipeline(
        pose_estimator=pose_estimator,
        depth_projector=depth_projector,
        skeleton_converter=skeleton_converter,
        dataset_root=dataset_root,
    )

    # 2. 批量处理全量 Phase 1 数据集 (48 个样本)
    records = pipeline.process_dataset(manifest_p)
    logger.info("Processed %d samples. Mean joint confidence across dataset: %.3f",
                len(records), np.mean([r["mean_confidence"] for r in records]))

    # 3. 读取不同动作类别的代表性样本生成多模态详细诊断图
    with open(manifest_p, "r", encoding="utf-8") as f:
        samples_data = json.load(f)["samples"]

    # 挑选 3 个代表性动作样例 (standing, sitting, bending)
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

    # 4. 遮挡实验与不确定性验证 (Simulated Occlusion Test)
    logger.info("Running simulated occlusion test (lower body occlusion)...")
    test_sample = selected_samples[0]  # standing sample
    rgb_test = np.array(Image.open(dataset_root / test_sample.rgb_path)).copy()
    depth_test = np.load(dataset_root / test_sample.depth_path).copy()

    # 正常状态骨架
    clean_skel, _ = pipeline.process_sample(test_sample, rgb_image=rgb_test, depth_map=depth_test, save_outputs=False)

    # 注入遮挡：将下半身 (v >= 320 像素) 进行黑色遮挡物覆盖，模拟桌子/沙发遮挡
    h, w = rgb_test.shape[:2]
    occ_rgb = rgb_test.copy()
    occ_rgb[310:, :] = 30  # 黑色实体遮挡物
    occ_depth = depth_test.copy()
    occ_depth[310:, :] = 0.8  # 遮挡物距离 0.8m (近距离遮挡)

    occ_skel, _ = pipeline.process_sample(test_sample, rgb_image=occ_rgb, depth_map=occ_depth, save_outputs=False)

    # 绘制遮挡对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 正常 vs 遮挡置信度对比条形图
    joint_abbr = [name[:8] for name in UNIFIED_JOINT_NAMES]
    y_pos = np.arange(len(joint_abbr))
    bar_width = 0.35

    ax1.barh(y_pos + bar_width/2, clean_skel.confidence, bar_width, label="Clean Observation", color="#2ECC71")
    ax1.barh(y_pos - bar_width/2, occ_skel.confidence, bar_width, label="Lower-Body Occluded", color="#E74C3C")
    ax1.axvline(0.35, color="black", linestyle="--", label="Occlusion Thresh (0.35)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(joint_abbr, fontsize=8.5)
    ax1.set_xlim(0.0, 1.05)
    ax1.set_xlabel("Joint Confidence", fontsize=10)
    ax1.set_title("Per-Joint Confidence Drop under Occlusion", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    # 遮挡下的 RGB + 2D 骨架渲染
    occ_overlay = visualizer.draw_2d_skeleton_on_rgb(occ_rgb, occ_skel)
    ax2.imshow(occ_overlay)
    ax2.set_title("RGB with Lower-Body Occlusion (Red=Detected Occluded Joints)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    plt.suptitle("ACTIVEVIEW v10.0 Phase 2: Occlusion & Perception Uncertainty Verification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    occ_save_p = demo_dir / "occlusion_confidence_drop_test.png"
    plt.savefig(occ_save_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved occlusion test visualization to: %s", occ_save_p)

    # 5. 生成 6 大动作类别 3D 骨架估计全景大图 (perception_overview.png)
    fig, axes = plt.subplots(6, 3, figsize=(16, 22))
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

        # Col 0: RGB + 2D Skeleton
        overlay = visualizer.draw_2d_skeleton_on_rgb(rgb_img, skel)
        axes[row_idx, 0].imshow(overlay)
        axes[row_idx, 0].set_title(f"RGB + 2D Skeleton | Action: {s.action_label.upper()}\nMean Conf: {np.mean(skel.confidence):.2f}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Depth Map
        d_im = axes[row_idx, 1].imshow(depth_map, cmap="plasma", vmin=0.5, vmax=5.0)
        valid_idx = np.where(skel.confidence >= 0.35)[0]
        axes[row_idx, 1].scatter(skel.joints_2d[valid_idx, 0], skel.joints_2d[valid_idx, 1], c="cyan", s=20, edgecolors="white")
        axes[row_idx, 1].set_title(f"Depth + Projected Keypoints\nValid Joints: {len(valid_idx)}/16", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Joint Confidence Bar Chart
        colors = ["#2ECC71" if c >= 0.35 else "#E74C3C" for c in skel.confidence]
        axes[row_idx, 2].barh(range(16), skel.confidence, color=colors, height=0.65)
        axes[row_idx, 2].axvline(0.35, color="red", linestyle="--", linewidth=1.2)
        axes[row_idx, 2].set_yticks(range(16))
        axes[row_idx, 2].set_yticklabels([name[:6] for name in UNIFIED_JOINT_NAMES], fontsize=7)
        axes[row_idx, 2].set_xlim(0.0, 1.05)
        axes[row_idx, 2].set_title(f"Per-Joint Confidence Distribution\nHead: {skel.part_confidence['head']:.2f}, Torso: {skel.part_confidence['torso']:.2f}", fontsize=10, fontweight="bold")
        axes[row_idx, 2].grid(axis="x", alpha=0.3)

        row_idx += 1
        if row_idx >= 6:
            break

    plt.suptitle("ACTIVEVIEW v10.0 Phase 2: Multi-Modal 3D Skeleton Perception Overview (6 Action Classes)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    overview_p = demo_dir / "perception_overview.png"
    plt.savefig(overview_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved perception overview to: %s", overview_p)

    # 6. 编写 README.md
    readme_content = f"""# ACTIVEVIEW v10.0 Phase 2: Perception Pipeline Demo

> **Phase 2 Overview & Verification**  
> 模拟真实机器人视觉观测链路：RGB-D 观测 $\\to$ 2D 关键点估计 $\\to$ 深度逆投影 $\\to$ 估计 3D 骨架与置信度融合。

---

## 1. 感知流水线数学原理与三步流程

1. **Step 1 (RGB $\\to$ 2D Keypoints)**:
   采用开箱即用轻量级检测器提取 17 关键点 2D 像素坐标 $(u, v)$ 与检测置信度 $c_{{2\\text{{D}}}}$；
2. **Step 2 (Depth 逆投影)**:
   结合相机几何内参与局部窗口自适应深度滤波：
   $$X_{{\\text{{cam}}}} = \\frac{{(u - c_x) \\cdot Z}}{{f_x}}, \\quad Y_{{\\text{{cam}}}} = \\frac{{(v - c_y) \\cdot Z}}{{f_y}}, \\quad Z_{{\\text{{cam}}}} = Z$$
3. **Step 3 (3D 拓扑融合与置信度计算)**:
   融合复合置信度 $c_i = c_{{2\\text{{D}}, i}} \\cdot c_{{\\text{{depth}}, i}}$，对低于阈值 (0.35) 的关节标记遮挡/不确定。

---

## 2. 产物目录结构 (Perception Artifacts)

物理保存在：
`{dataset_root / "perception"}`

```text
perception/
├── pose2d/             # 2D 关键点与检测框 (JSON)
├── pose3d/             # 相机/世界系 16 关节 3D 坐标 (JSON)
├── confidence/         # 逐关节与部位置信度 (JSON)
├── visualization/      # 样本可视化诊断图
└── metadata/
    └── perception_manifest.json # 全量感知样本元数据清单
```

---

## 3. 可视化图表

- **全量动作多模态感知概览**：`perception_overview.png`
- **遮挡置信度下降测试**：`occlusion_confidence_drop_test.png`
- **单动作多模态诊断图**：`perception_demo_standing.png`, `perception_demo_sitting.png`, `perception_demo_bending.png`

---

## 4. 运行验证命令

```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase2_demo
```
"""
    with open(demo_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Phase 2 Demo completed successfully!")
    return records


def main():
    run_phase2_demo()
    print("\n" + "=" * 80)
    print("  ACTIVEVIEW v10.0 Phase 2 Demo Executed Successfully")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
