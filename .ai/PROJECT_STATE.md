# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v10.0 Phase 2.1 — Perception Pipeline & Scientific GT Validation**
  - **Phase 1 (Dataset Generation)**: `FROZEN` (正式冻结并建立隔离边界)；
  - **Phase 2 (Perception Pipeline)**: `FROZEN` (成熟 RGB-D 骨架提取、统一拓扑 Schema、严格 2D-3D 几何一致性与自动化审计通过)；
  - **Phase 2.1 (Scientific GT Validation)**: `COMPLETED & FROZEN` (离线 GT 对齐与评测体系、MPJPE/PA-MPJPE/PCK 定量评测与 10 样本验证集)；
  - **Phase 3 (ST-GCN / Action Recognition)**: `READY TO START` (待用户明确指令后进入)。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 任务先验规则主动视角打分基准（COMPLETED & RETAINED AS BASELINE）；
  - `v9.1`: 感知质量驱动主动视角选择与退化基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> ... -> v9 -> v10`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v10.0 (10.0.0 Phase 2.1 — Perception Pipeline & Validation)
- **Title**: RGB-D Driven Task-aware Active View Selection for Human Action Recognition
- **Active Code Directory**: `ea_avs_mvp_v10/`
- **Active Specifications**:
  - `ACTIVEVIEW_V10.0_Research_Development_Document.md`
  - `ACTIVEVIEW_V10.0_Code_Generation_Specification.md`
- **Inherited Assets & Infrastructure from v7, v8, v9**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载与动作驱动 (`ea_avs_mvp_v7.human`)
  - MotionPlayer 动作回放控制器与标准 AMASS 动作库 (`ea_avs_mvp_v7.motion`)
  - 局部极坐标网格视点生成与约束过滤 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v8.constraints`)
  - 统一数据根目录：`/home/zxf/WorkSpace/code/data/ActiveView/`

---

## 3. Latest Stable Implementations
- **v10.0 Phase 2.1 (`ea_avs_mvp_v10/evaluation/`, `ea_avs_mvp_v10/perception/`, ACTIVE / FROZEN)**:
  - 统一骨架拓扑与 GT 映射规范 (`configs/skeleton_definition.json`)；
  - 离线 GT 对齐与空间转换 (`ea_avs_mvp_v10/evaluation/skeleton_alignment.py`)；
  - 标准三维姿态评测器 (`ea_avs_mvp_v10/evaluation/skeleton_evaluator.py`: MPJPE, PA-MPJPE, PCK@threshold)；
  - 估计骨架 vs GT 骨架 3D 重叠与误差向量可视化器 (`tools/v10/skeleton_compare_visualizer.py`)；
  - 10 样本多视角动作科研验证集 (`ea_avs_mvp_v10/examples/v10_phase2_validation/`)；
  - 成熟 3D 骨架提取器 (`MediaPipeRGBDSkeletonExtractor`, `RGBDSkeletonExtractor`)；
  - 根节点中心化与尺度归一化 (`SkeletonNormalizer`，严禁视角旋转归一化)；
  - 坐标与运动学合理性校验器 (`CoordinateValidator`)；
  - 肢体着色与等物理长宽比可视化器 (`SkeletonVisualizer`)；
  - 骨架一致性自动审计套件 (`tools/check_skeleton_consistency.py`, `Phase2_Skeleton_Audit_Report.md`)。
- **v10.0 Phase 1 (`ea_avs_mvp_v10/dataset/`, FROZEN)**:
  - 真实 RGB-D 多视角数据集采集底座与 6 大动作资产规范化；
  - 样本索引清单与物理隔离目录 (`datasets/v10/raw/`, `ground_truth/`, `metadata/`)。
- **v9.1 (`ea_avs_mvp_v9/`, CLOSED / FINALIZED)**: 感知质量与信息增益驱动的主动视角规划基准。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED)**: 局部主动视点规划与三阶空间硬约束基准。

---

## 4. Phase 2.1 Scientific & Engineering Achievements
1. **真实观测链闭环**：
   > *"The 3D skeleton used by downstream modules is reconstructed from robot-observed RGB-D data instead of directly using simulation ground truth."*
2. **几何一致性规范**：
   > *"The reconstructed 3D skeleton is geometrically consistent with the observed 2D keypoints."*
3. **严格科研数据隔离边界**：GT 骨骼仅在 `evaluation/` 离线使用，在线模型仅输入估计骨架；
4. **定量评测指标通过**：验证集平均 MPJPE 为 116.98 mm (11.7 cm)，PA-MPJPE 为 87.42 mm (8.7 cm)，PCK@15cm 为 77.14%；
5. **单元与回归测试 100% 通过**：全量 113 项测试全部 PASS。
