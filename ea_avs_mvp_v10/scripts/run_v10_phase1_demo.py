"""
v10 Phase 1 演示与数据生成流水线 —— run_v10_phase1_demo.py
==========================================================

职责：
    1. 运行 Phase 1 RGB-D 数据采集流水线；
    2. 针对 6 大动作类别 (standing, walking, sitting, bending, reaching, falling) 各采集多视角样本；
    3. 输出完整的 RGB 图像、深度图、相机位姿、动作标签与 GT 骨骼 (Oracle用)；
    4. 生成论文级多模态可视化对比图 (sample_visualization.png)；
    5. 输出 examples/v10_phase1_demo/ 演示目录与说明文档。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.config import load_v10_config
from ea_avs_mvp_v10.core.paths import get_data_root, get_repo_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import ActionClassV10
from ea_avs_mvp_v10.dataset.v10_dataset_generator import V10DatasetGenerator
from ea_avs_mvp_v10.motion.motion_manager import MotionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v10_phase1_demo")


def run_phase1_demo():
    """运行 Phase 1 完整演示流水线。"""
    repo_root = get_repo_root()
    demo_dir = repo_root / "ea_avs_mvp_v10" / "examples" / "v10_phase1_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(">>> Starting ACTIVEVIEW v10.0 Phase 1 Demo...")
    cfg = load_v10_config()
    generator = V10DatasetGenerator(config=cfg)

    # 1. 选取 6 大类别动作资产各 1 个进行标准采样
    motion_mgr = MotionManager()
    selected_motions = []
    for act_class in ActionClassV10:
        m_list = motion_mgr.get_motions_by_class(act_class)
        if m_list:
            selected_motions.append(m_list[0])
        else:
            logger.warning("No motion found for class %s", act_class.value)

    logger.info("Selected 6 motion assets for Phase 1 Demo: %s", selected_motions)

    # 2. 执行多视角 RGB-D 采集
    samples = generator.generate_motion_dataset(
        motion_ids=selected_motions,
        human_position=[1.5, -1.60, 4.0],
        human_yaw_deg=0.0,
        frame_step=15,
        max_frames_per_motion=2,
        max_viewpoints=4,
    )

    logger.info("Generated %d dataset samples in total.", len(samples))

    # 3. 选取代表性样本生成多模态组合可视化图
    dataset_root = get_v10_dataset_root()

    # 按动作类别挑选 4 个展示样例
    display_samples = []
    seen_classes = set()
    for s in samples:
        if s.action_label not in seen_classes:
            display_samples.append(s)
            seen_classes.add(s.action_label)
        if len(display_samples) >= 4:
            break

    if not display_samples and samples:
        display_samples = samples[:4]

    fig, axes = plt.subplots(len(display_samples), 3, figsize=(15, 3.8 * len(display_samples)))
    if len(display_samples) == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, s in enumerate(display_samples):
        rgb_full_p = dataset_root / s.rgb_path
        depth_full_p = dataset_root / s.depth_path

        # 加载 RGB
        if rgb_full_p.exists():
            rgb_img = Image.open(rgb_full_p)
        else:
            rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # 加载 Depth
        if depth_full_p.exists():
            depth_arr = np.load(depth_full_p)
        else:
            depth_arr = np.ones((480, 640), dtype=np.float32)

        # 绘制 RGB
        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].set_title(f"RGB | Action: {s.action_label.upper()} (Frame {s.frame_idx})\nView: {s.view_id}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 绘制 Depth
        d_plot = axes[row_idx, 1].imshow(depth_arr, cmap="plasma", vmin=0.5, vmax=5.0)
        axes[row_idx, 1].set_title(f"Depth Map (Meters)\nMin: {depth_arr.min():.2f}m, Max: {depth_arr.max():.2f}m", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")
        fig.colorbar(d_plot, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 绘制 Camera & Target Metadata 卡片
        axes[row_idx, 2].axis("off")
        meta_text = (
            f"Sample ID: {s.sample_id}\n"
            f"Scene ID: {s.scene_id}\n"
            f"Motion ID: {s.motion_id}\n"
            f"Action Label: {s.action_label}\n"
            f"View ID: {s.view_id}\n"
            f"----------------------------------------\n"
            f"Camera Pos: [{s.camera_pose.position[0]:.2f}, {s.camera_pose.position[1]:.2f}, {s.camera_pose.position[2]:.2f}]\n"
            f"Camera Yaw: {s.camera_pose.yaw_deg:.1f}°\n"
            f"Intrinsics: fx={s.camera_pose.intrinsics.fx:.1f}, cx={s.camera_pose.intrinsics.cx:.1f}\n"
            f"GT Skeleton Available: {'YES (Oracle Only)' if s.gt_skeleton_path else 'NO'}\n"
            f"Resolution: 640x480 (HFOV 90°)\n"
        )
        axes[row_idx, 2].text(
            0.05, 0.5, meta_text,
            fontsize=9.5, family="monospace", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECEFF1", edgecolor="#90A4AE", alpha=0.9),
        )

    plt.suptitle("ACTIVEVIEW v10.0 Phase 1: Habitat Multi-View RGB-D Dataset Sample Overview", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    vis_p = demo_dir / "sample_visualization.png"
    plt.savefig(vis_p, dpi=160)
    plt.close(fig)
    logger.info("Saved sample visualization to: %s", vis_p)

    # 4. 生成 demo README.md 说明文档
    demo_readme = f"""# ACTIVEVIEW v10.0 Phase 1: RGB-D Dataset Generation Demo

> **Phase 1 Acceptance & Verification Overview**  
> 本目录展示并记录 ACTIVEVIEW v10.0 Phase 1 阶段生成的标准仿真多视角 RGB-D 观测样本。

---

## 1. Phase 1 数据集标准结构 (Dataset Structure)

数据集物理保存在：
`{dataset_root}`

```text
datasets/v10/
├── raw/
│   ├── rgb/              # 原始 RGB 观察图像 (PNG, uint8, 640x480)
│   ├── depth/            # 原始深度数据 (NPY float32 距离矩阵 + 可视化 PNG)
│   ├── camera_pose/      # 相机外参、世界位置、四元数与内参 (JSON)
│   └── scene_meta/       # 场景、人体位置、朝向等元数据 (JSON)
├── ground_truth/
│   ├── skeleton/         # 16 关键点 3D 坐标真值 (仅用于 Oracle/评估，严禁模型推理)
│   └── action/           # 真实动作类别与子标签 (JSON)
└── metadata/
    └── samples.json      # 全量样本索引清单 (JSON)
```

---

## 2. 样本模态与元数据示例 (Sample Metadata Example)

```json
{{
  "sample_id": "{samples[0].sample_id}",
  "scene_id": "{samples[0].scene_id}",
  "motion_id": "{samples[0].motion_id}",
  "action_label": "{samples[0].action_label}",
  "frame_idx": {samples[0].frame_idx},
  "view_id": "{samples[0].view_id}",
  "camera_pose": {{
    "position": {samples[0].camera_pose.position},
    "rotation_quat": {samples[0].camera_pose.rotation_quat},
    "yaw_deg": {samples[0].camera_pose.yaw_deg},
    "intrinsics": {{
      "width": 640,
      "height": 480,
      "fx": 320.0,
      "fy": 320.0,
      "cx": 320.0,
      "cy": 240.0,
      "hfov_deg": 90.0
    }}
  }},
  "rgb_path": "{samples[0].rgb_path}",
  "depth_path": "{samples[0].depth_path}",
  "gt_skeleton_path": "{samples[0].gt_skeleton_path}"
}}
```

---

## 3. 多模态样本可视化概览 (Multi-Modal Sample Visualization)

![Phase 1 Sample Overview](sample_visualization.png)

---

## 4. 复现与运行命令 (Execution Commands)

### 运行 Phase 1 完整测试与生成流水线：
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase1_demo
```
"""
    with open(demo_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(demo_readme)

    logger.info("Phase 1 Demo completed successfully!")
    return samples


def main():
    run_phase1_demo()
    print("\n" + "=" * 80)
    print("  ACTIVEVIEW v10.0 Phase 1 Demo Executed Successfully")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
