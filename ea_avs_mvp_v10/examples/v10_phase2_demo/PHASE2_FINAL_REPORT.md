# ACTIVEVIEW v10.0 Phase 2: Final Acceptance & Freeze Report

> **Status**: `PHASE 2 FROZEN` (Phase 2 核心感知流水线研发与验证闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 2: RGB-D Perception Pipeline -> Estimated 3D Skeleton)

---

## 1. 阶段目标与完成情况清单 (Completed Scope & Modules)

| 核心模块 | 对应代码 | 验收状态 | 说明 |
|---|---|---|---|
| **2D 姿态估计器封装** | `ea_avs_mvp_v10/perception/pose_estimator.py` | **PASS** | 开箱即用 Torchvision Keypoint R-CNN，原生输出 COCO-17 关键点 |
| **局部自适应深度逆投影** | `ea_avs_mvp_v10/perception/depth_projection.py` | **PASS** | 针孔反投影模型 + 局部邻域自适应中值滤波恢复 $(X, Y, Z)$ |
| **COCO-17 骨架生成与融合** | `ea_avs_mvp_v10/perception/skeleton_converter.py` | **PASS** | 原生保留 COCO-17 拓扑，$c = c_{2\text{D}} \cdot c_{depth}$ 复合感知置信度 |
| **3D 骨架空间归一化** | `ea_avs_mvp_v10/perception/skeleton_normalizer.py` | **PASS** | 根节点去中心化 + 躯干尺度归一化 (严格禁止相机朝向旋转归一化) |
| **拓扑适配器接口** | `ea_avs_mvp_v10/perception/skeleton_adapter.py` | **PASS** | 可选 COCO-17 $\to$ NTU-25 适配器，Phase 2 默认不转换 |
| **感知数据集持久化** | `ea_avs_mvp_v10/dataset/perception_dataset.py` | **PASS** | 保存 `pose2d/`, `pose3d/`, `normalized_pose3d/`, `confidence/` |
| **多模态可视化套件** | `ea_avs_mvp_v10/visualization/skeleton_visualizer.py` | **PASS** | 5 面板诊断图、归一化对比图、全量 6 动作全景图与遮挡测试图 |

---

## 2. 坐标系统与数据结构规范 (Coordinate Systems & Representations)

为了支持机器人局部感知、环境全局分析以及动作识别模型输入，系统严格分离保存三个坐标体系：

1. **Camera Coordinate (`joints_3d_cam`, 形状 $17 \times 3$)**：
   - 局部机器人相机坐标系，单位为米 (Meters)；
   - 用于机器人局部主动感知与观测收益计算。
2. **World Coordinate (`joints_3d_world`, 形状 $17 \times 3$)**：
   - Habitat 仿真场景世界坐标系，结合相机位姿外参齐次变换恢复；
   - 用于场景几何分析与全局空间对齐。
3. **Normalized Coordinate (`joints_3d_normalized`, 形状 $17 \times 3$)**：
   - 根节点平移（以骨盆/跨部中心为原点 $(0, 0, 0)$）+ 躯干尺度归一化；
   - **严格禁止相机朝向旋转归一化**，完整保留不同视角带来的几何投影变化；
   - 作为后续 Phase 3 ST-GCN 动作分类器的直接输入。

---

## 3. 感知置信度与不确定性定义 (Perception Confidence & Uncertainty)

- **Perception Confidence ($c_i \in [0, 1]$)**：
  $$c_i = c_{2\text{D}, i} \cdot c_{\text{depth}, i}$$
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
