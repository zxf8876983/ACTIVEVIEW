# ACTIVEVIEW Project State

Last Updated: 2026-08-20  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：v7.0 拟人化仿真环境已完成全链路闭环验证 (v7.0 Humanoid-driven Active Perception Simulation Environment — COMPLETED)。v6.0 保持正式封版定稿 (EA-AVS-MVP v6.0 — CLOSED / FINALIZED)。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> 未来版本`），各版本保留独立目录与设计文档。核心功能全链路跑通前暂不重构为单一 `src/`。

---

## 2. Active Development Version
- **Version**: v7.0 (7.0.0 — Completed)
- **Title**: Humanoid-driven Active Perception Simulation Environment
- **Active Code Directory**: `ea_avs_mvp_v7/`
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Previous Stable Specification**: `EA_AVS_MVP60_Code_Generation_Document.md` (v6.0 CLOSED)
- **Infrastructure & Tools**: `tools/motion_assets/` (BABEL / AMASS Elderly Motion Asset Pipeline)

---

## 3. Latest Stable Implementation
- **Version**: v6.0 (6.0.0 — CLOSED / FINALIZED)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Specification Document**: `EA_AVS_MVP60_Code_Generation_Document.md`
- **Previous Stable Baseline**: v5.0 (`ea_avs_mvp_v5/`)

---

## 4. Motion Asset Infrastructure (v7.0 Input & Assets)
- **数据根目录解析**：统一使用相对路径 `../../data/ActiveView`（支持 `ACTIVEVIEW_DATA_ROOT` 环境变量覆盖）。
- **BABEL 标注同步与解析**：已同步 `babel_v1.0_release`，建立动作筛选器解析出 10,370 条候选动作片段。
- **老人 5 类典型动作定义与 Feasibility Set**：
  1. `standing` (自然站立、静止站立) — 3 条
  2. `sitting` (坐下、坐姿端坐) — 3 条
  3. `bending` (弯腰、拾物) — 3 条
  4. `reaching` (触及、前伸伸手) — 3 条
  5. `fall_related` (高置信摔倒与倒地姿态) — 5 条
  - 精选 17 条高质量 frame-level 动作序列，覆盖 5 个核心 AMASS 子库（`BMLrub`, `CMU`, `EKUT`, `EyesJapanDataset`, `KIT`）。
- **AMASS 数据集落地**：已从 Hugging Face 完整下载并解压 5 个子库全部 7,862 个 `.npz` 文件（13.75 GB），`index_amass_files.py` 严格校验 17/17 PASS 并生成 `motion_asset_manifest.json`。

---

## 5. Version Evolution
- **v1.0 (`ea_avs_mvp/`)**: 几何版候选视角采样与可达性过滤，基于抽象骨架计算 FOV 可见性与移动代价，对比 Fixed/Random/Nearest/Ours。
- **v2.0 (`ea_avs_mvp_v2/`)**: 严格确立 `Pred`（决策前预测评分）与 `True`（到达后真实评估）信息边界，严禁选择阶段偷看未来图像，引入原地保留（stay）机制。
- **v3.0 (`ea_avs_mvp_v3/`)**: 引入多姿态骨架（standing, sitting, lying_fallen, bending）、人体朝向建模（`human_yaw`）与动作关键部位加权评分（`S_action_part`）。
- **v4.0 (`ea_avs_mvp_v4/`)**: 引入物理引擎 Ray Casting 实现环境障碍物遮挡检测，提出遮挡感知评分（`S_action_occ`）、消融策略体系与同口径 Oracle 离线上界。
- **v5.0 (`ea_avs_mvp_v5/`)**: 接入真实 Habitat KinematicHumanoid（`neutral_0`）与 RGB-D/语义渲染，由 URDF link 派生 15 关节 GT 骨架，实现 5 类遮挡判定与决策闭环。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 
  1. Static-Scene-Only 射线检测彻底消除对真实 Humanoid 铰接碰撞体的隐式 GT 泄漏；
  2. Estimated-State 移除 full-collision 降级，缺失 static API 时严格 fail-closed (`valid=False / unknown`)；
  3. `resolve_stage_id` 禁止猜测默认值，缺失 stage_id 时严格 fail-fast；
  4. AST Guard A2 静态严禁 `cast_ray_to_estimated_point` 出现 `runner.cast_ray` 调用；
  5. 修复 Shared-Pool GT 支路打分一致性，确保在同一 GT-centered 几何空间严谨对比 GT vs Est 评分；
  6. 建立 Oracle Candidate-Pool Identity Guard，跨池返回 `pool_mismatch`，上界破坏返回 `oracle_not_upper_bound`，杜绝 `max(0)` 掩盖错误；
  7. 恢复 `strict_gt_skeleton=True` 特权基线与评测约束；
  8. 建立 17 项纯 Python 严谨性单元测试套件与 4 项零 GT 防护测试。

---

## 6. Current Research Roadmap & Future Directions

### v7.0: Humanoid-driven Active Perception Simulation Environment
- **当前状态**：v7.0 Habitat Humanoid rendering verified (COMPLETED)：
  1. AMASS motion pipeline (标准化读取与四元数转换)
  2. Habitat humanoid playback (KinematicHumanoid 54 关节姿态驱动、场景物理地面高程对齐与相对位移动态补偿)
  3. Robot RGB-D observation (解耦 robot_pose 与 camera_pose)
  4. Episode generation (多动作多视角标准化 Episode 存储)
  5. Motion validity evaluation (多维动力学 ActionMotionMetrics)
  6. Visualization & video generation (统一 Demo 入口与 MP4 视频生成)
- **空间几何与动作位移核心规范 (Key Spatial Invariants)**：
  - **场景物理地面高程**：`apartment_1.glb` 物理地面标高为 $Y = -1.60\text{m}$（天花板 $Y = +1.14\text{m}$），严禁盲目假设 $Y = 0.0\text{m}$；
  - **相对位移与下沉补偿**：KinematicHumanoid 自带 $+0.90\text{m}$ 骨盆偏移，`apply_motion_frame` 统一采用相对位移公式 $\text{mat}[:3, 3] = [T_x(t)-T_x(0), T_y(t)-0.90, T_z(t)-T_z(0)]$，杜绝原点绝对叠加的同时完整保留坐下与摔倒触地真实下沉。
- **定位**：基础设施与实验平台构建版本。将 ACTIVEVIEW 从抽象/静态人体感知环境转变为由真实人体运动数据驱动的室内老人监护仿真环境。
- **核心目标与流水线**：
  ```text
  BABEL Action Annotation
          ↓
  AMASS Human Motion
          ↓
  Motion Normalization (NormalizedMotion)
          ↓
  Humanoid Motion Conversion (SMPL-X PKL)
          ↓
  Habitat Humanoid Playback (KinematicHumanoid)
          ↓
  Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
          ↓
  Episode Dataset
  ```
- **聚焦能力 (Focuses on)**：
  1. 逼真的人体 Humanoid 表达与 16 关键点 3D 世界坐标真值提取；
  2. AMASS 人体运动数据标准化加载与 SMPL-X 四元数转换回放；
  3. Habitat 室内场景集成与模拟器生命周期管理；
  4. 移动机器人搭载相机的 RGB-D 观测生成与内外参计算；
  5. 可复现的 Episode 数据集与标注文件 (metadata.json) 生成。
- **明确排除在 v7.0 之外的非目标 (Explicitly NOT in v7.0)**：
  - ❌ 不实现 NBV 视角选择算法
  - ❌ 不实现 Action-aware utility 优化
  - ❌ 不实现 Evidence Recovery 机制
  - ❌ 不实现 强化学习 (RL)
  - ❌ 不实现 多步路径规划 (Multi-step Planning)
  - ❌ 不训练 动作识别模型 (HAR)
  - ❌ 不修改 v6.0 代码及历史实现

### 未来研究方向（仅记录规划，当前版本不开发）：

#### v8.0: Action-aware Active View Selection
- Action hypothesis / 动作后验建模 $P(a \mid O_t)$
- Action-conditioned 视角效用函数设计
- 多策略与视角对比评估

#### v9.0: Uncertainty-aware Multi-step Active Perception
- 观测不确定性建模 (Observation Uncertainty)
- 判别性证据恢复机制 (Evidence Recovery)
- 序列化多步主动视角规划 (Sequential Viewpoint Planning)

---

## 7. Engineering Constraints & Operational Boundary
1. **数据边界保护**：严禁将 AMASS / BABEL 原始大型数据（npz, tar.bz2, images, pth）提交至 Git 仓库；
2. **统一路径解析**：统一使用 `../../data/ActiveView` 或 `ACTIVEVIEW_DATA_ROOT`，严禁机器路径硬编码；
3. **只读保护**：保持 `ea_avs_mvp_v6/` 及更早版本只读；
4. **小步验证**：每个模块均需具备明确的输入、输出、依赖与测试验证命令。

---

## 8. Historical Versions (Read-Only)
- `ea_avs_mvp/` (v1.0), `ea_avs_mvp_v2/` (v2.0), `ea_avs_mvp_v3/` (v3.0), `ea_avs_mvp_v4/` (v4.0), `ea_avs_mvp_v5/` (v5.0), `ea_avs_mvp_v6/` (v6.0) 为历史及稳定参考版本，保持只读，不参与当前开发修改。
