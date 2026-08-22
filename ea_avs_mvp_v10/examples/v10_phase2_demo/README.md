# ACTIVEVIEW v10.0 Phase 2: Perception Pipeline Demo

> **Phase 2 Overview & Verification**  
> 模拟真实机器人视觉观测链路：RGB-D 观测 $\to$ 2D 关键点估计 $\to$ 深度逆投影 $\to$ 估计 3D 骨架与置信度融合。

---

## 1. 感知流水线数学原理与三步流程

1. **Step 1 (RGB $\to$ 2D Keypoints)**:
   采用开箱即用轻量级检测器提取 17 关键点 2D 像素坐标 $(u, v)$ 与检测置信度 $c_{2\text{D}}$；
2. **Step 2 (Depth 逆投影)**:
   结合相机几何内参与局部窗口自适应深度滤波：
   $$X_{\text{cam}} = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y_{\text{cam}} = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z_{\text{cam}} = Z$$
3. **Step 3 (3D 拓扑融合与置信度计算)**:
   融合复合置信度 $c_i = c_{2\text{D}, i} \cdot c_{\text{depth}, i}$，对低于阈值 (0.35) 的关节标记遮挡/不确定。

---

## 2. 产物目录结构 (Perception Artifacts)

物理保存在：
`/home/zxf/WorkSpace/code/data/ActiveView/datasets/v10/perception`

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
