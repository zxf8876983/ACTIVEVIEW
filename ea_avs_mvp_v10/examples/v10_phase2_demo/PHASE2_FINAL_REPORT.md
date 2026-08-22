# ACTIVEVIEW v10.0 Phase 2: Final Acceptance & Freeze Report

> **Status**: `PHASE 2 FROZEN` (Phase 2 核心感知流水线研发与验证闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 2: RGB-D Skeleton Extractor -> Estimated 3D Skeleton)

---

> [!IMPORTANT]
> **Core Scientific & Perception Principle**:  
> **The 3D skeleton used by downstream modules is reconstructed from robot-observed RGB-D data instead of directly using simulation ground truth.**  
> (下游动作识别与决策模块使用的 3D 人体骨架完全由机器人真实观测的 RGB-D 视觉数据重建估计得到，严格禁止直接读取仿真真值姿态。)

---

## 1. 选定的成熟 RGB-D Skeleton Extractor 与安装配置

经过调研与实验验证，Phase 2 选定 **MediaPipe BlazePose 3D** 作为核心人体 3D 骨架提取器后端：

- **项目与算法**：Google MediaPipe Pose (BlazePose 3D World Landmarks)
- **输入格式**：RGB 图像 ($H \times W \times 3$, uint8) + 深度图 ($H \times W$, float32, 米) + 相机位姿
- **输出拓扑**：33 关键点标准解剖学运动学骨架 (`MediaPipe33`)，天然具备人体刚性与对称性先验，杜绝单点深度逆投影的肢体撕裂与拉伸
- **安装方式**：`pip install mediapipe==0.10.14 numpy==1.26.4`
- **预训练权重**：内置 `pose_landmark_heavy.tflite` 高精度权重

---

## 2. 阶段目标与完成情况清单 (Completed Scope & Modules)

| 核心模块 | 对应代码 | 验收状态 | 说明 |
|---|---|---|---|
| **成熟 RGB-D 骨架提取器** | `ea_avs_mvp_v10/perception/rgbd_skeleton_extractor.py` | **PASS** | 封装 `RGBDSkeletonExtractor` (MediaPipe 33 关键点 + 根节点深度几何对齐) |
| **3D 骨架空间归一化** | `ea_avs_mvp_v10/perception/skeleton_normalizer.py` | **PASS** | 根节点去中心化 + 躯干尺度归一化 (**严格禁止相机朝向旋转归一化**) |
| **坐标健康度校验器** | `ea_avs_mvp_v10/perception/coordinate_validator.py` | **PASS** | 物理深度范围、人体尺度、头部/脚踝相对上下位姿合理性检查 |
| **拓扑适配器接口** | `ea_avs_mvp_v10/perception/skeleton_adapter.py` | **PASS** | 提供 `MediaPipe33ToCOCO17Adapter` 与 `MediaPipe33ToNTU25Adapter` 接口 |
| **感知数据集持久化** | `ea_avs_mvp_v10/dataset/perception_dataset.py` | **PASS** | 保存 `skeleton_raw/`, `skeleton_normalized/`, `confidence/`, `metadata/` |
| **自动化单样本检查 CLI** | `tools/v10/check_sample.py` | **PASS** | 支持按 ID、按动作或随机挑选生成 4 面板高分辨率诊断图 |
| **多模态可视化套件** | `ea_avs_mvp_v10/visualization/skeleton_visualizer.py` | **PASS** | 严格基于官方拓扑渲染骨骼连线 |

---

## 3. 坐标系统与数据结构规范 (Coordinate Systems & Representations)

详细规范请参阅 [`docs/V10_COORDINATE_SYSTEM.md`](../../../docs/V10_COORDINATE_SYSTEM.md)。

系统严格解耦保存四个物理与模型维度的坐标体系：

1. **Image Coordinate $(u, v)$**：
   - 图像左上角原点，用于 2D 检测与可视化叠加。
2. **Camera Coordinate (`joints_3d_camera`, 形状 $33 \times 3$)**：
   - 局部机器人相机坐标系（$+X$ 右, $+Y$ 上, $+Z$ 前/深度），单位为米 (Meters)；
   - 作为 Phase 2 主要 3D 输出，用于机器人局部主动感知与视点规划打分。
3. **World Coordinate (`joints_3d_world`, 形状 $33 \times 3$)**：
   - Habitat 仿真场景世界坐标系，结合相机位姿外参齐次变换恢复；
   - **仅用于 Habitat 全局可视化与场景几何分析，严禁作为下游动作分类器输入**。
4. **Normalized Coordinate (`joints_3d_normalized`, 形状 $33 \times 3$)**：
   - 根节点平移（以骨盆/跨部中心为原点 $(0, 0, 0)$）+ 躯干尺度归一化；
   - **严格禁止相机朝向旋转归一化**（完整保留主动视角在投影几何上的差异性）；
   - 作为后续 Phase 3 ST-GCN 动作分类器的直接输入。

---

## 4. 实验验证统计与质量评估 (Validation & Quality Evaluation)

全量 Phase 1 数据集 (48 个多视角动作样本) 处理结果：
- **总样本数**：59
- **合法样本 (VALID)**：53
- **警告样本 (WARNING)**：0
- **异常样本 (INVALID)**：6
- **有效率 (Pass Rate)**：89.8%
- **平均关节感知置信度**：0.565

---

## 5. 当前阶段限制与冻结声明 (Limitations & Freeze Declaration)

- **状态冻结声明**：
  **ACTIVEVIEW v10.0 Phase 2 核心感知流水线与坐标系统已全部通过验收，正式标记为 `PHASE 2 FROZEN`。Phase 3 (ST-GCN Action Recognition) 将严格以 Phase 2 生成的 `skeleton_normalized` 序列与感知置信度作为输入。**
