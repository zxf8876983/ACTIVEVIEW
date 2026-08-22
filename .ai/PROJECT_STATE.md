# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v10.0 Phase 2 — Perception Pipeline (RGB-D -> Estimated 3D Skeleton)**
  - **Phase 1 (Dataset Generation)**: `FROZEN` (正式冻结并建立隔离边界)；
  - **Phase 2 (Perception Pipeline)**: `COMPLETED & VERIFIED` (RGB -> 2D Pose -> Depth 逆投影 -> 3D 骨架与置信度融合闭环跑通)；
  - **Phase 3 (ST-GCN / Action Recognition)**: `READY TO START` (待进入)。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 任务先验规则主动视角打分基准（COMPLETED & RETAINED AS BASELINE）；
  - `v9.1`: 感知质量驱动主动视角选择与退化基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> ... -> v9 -> v10`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v10.0 (10.0.0 Phase 2 — Perception Pipeline)
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
- **v10.0 Phase 2 (`ea_avs_mvp_v10/perception/`, `ea_avs_mvp_v10/dataset/perception_dataset.py`, ACTIVE / COMPLETED & VERIFIED)**:
  - 2D 关键点检测器 (`TorchvisionPoseEstimator`, `MockPoseEstimator`)；
  - 深度邻域滤波与相机逆投影 (`DepthProjector`)；
  - 骨架格式转换与置信度融合器 (`SkeletonConverter`: $c = c_{\text{2D}} \cdot c_{\text{depth}}$)；
  - 2D/3D 多模态可视化诊断器 (`SkeletonVisualizer`)；
  - 遮挡检测与不确定性验证套件 (`examples/v10_phase2_demo/`)。
- **v10.0 Phase 1 (`ea_avs_mvp_v10/dataset/`, FROZEN)**:
  - 真实 RGB-D 多视角数据集采集底座与 6 大动作资产规范化；
  - 样本索引清单与物理隔离目录 (`datasets/v10/raw/`, `ground_truth/`, `metadata/`)。
- **v9.1 (`ea_avs_mvp_v9/`, CLOSED / FINALIZED)**: 感知质量与信息增益驱动的主动视角规划基准。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED)**: 局部主动视点规划与三阶空间硬约束基准。

---

## 4. Phase 2 Scientific & Engineering Achievements
1. **真实观测链闭环**：完全基于 RGB-D 观测输入（无需 GT 骨骼，无需 GT 可见性），实现 RGB 图像 $\to$ 2D 姿态 $\to$ 局部自适应深度逆投影 $\to$ 3D 关节坐标（相机系/世界系）的自动化估计；
2. **多模态置信度与遮挡检测**：复合置信度 $c_i = c_{\text{2D}, i} \cdot c_{\text{depth}, i}$ 能够真实反映视线遮挡（如下肢遮挡时置信度急剧下降至 $<0.05$ 并触发 `occluded_mask` 预警）；
3. **严格数据物理隔离**：估计产物规范存储于 `datasets/v10/perception/` (`pose2d/`, `pose3d/`, `confidence/`, `metadata/`)，为 Phase 3 ST-GCN 动作识别提供纯粹基于估计姿态的输入。
