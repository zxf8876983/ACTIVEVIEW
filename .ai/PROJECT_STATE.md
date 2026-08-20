# ACTIVEVIEW Project State

Last Updated: 2026-08-20  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：进入 **v8.0 Phase 1 — Human-aware Active View Foundation** 研发阶段。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8`），各版本保留独立目录与设计文档。v8 模块化复用 v7 仿真底座，严禁重写/复制整个 v7 工程。

---

## 2. Active Development Version
- **Version**: v8.0 (8.0.0 — Phase 1: Foundation Architecture)
- **Title**: Human-aware Active View Foundation
- **Active Code Directory**: `ea_avs_mvp_v8/`
- **Active Specification**: `EA_AVS_MVP80_Code_Generation_Document.md`
- **Previous Specification**: `EA_AVS_MVP70_Code_Generation_Document.md` (v7.0 COMPLETED)
- **Inherited Assets & Infrastructure from v7**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`)
  - KinematicHumanoid 人形模型加载与 16 关节 3D GT 提取 (`ea_avs_mvp_v7.human`)
  - AMASS/BABEL 动作加载与相对位移补偿驱动 (`ea_avs_mvp_v7.motion`)
  - 移动机器人底盘与 RGB-D 传感器解耦观测 (`ea_avs_mvp_v7.robot`)
  - 结构化 Episode 数据集落盘规范 (`ea_avs_mvp_v7.dataset`, `ea_avs_mvp_v7.observation`)

---

## 3. Latest Stable Implementations
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 严格防泄漏 Estimated-State 物理光线追踪主动视角闭环。

---

## 4. v8.0 Research Objectives & Boundaries

### 核心科研目标 (Core Objectives):
将 ACTIVEVIEW 从 v7.0 的“固定人体位置 + 固定机器人视点观测”推进为“空间约束下的人体感知候选视点空间生成与质量评价”：
1. **Human Placement**: 人体在室内场景中的合法地面位置采样与几何校验；
2. **Candidate View Generation**: 围绕人体的多距离、多方位角机器人候选观察视角规则采样；
3. **Constraint Checking**: 基于 Habitat Pathfinder 与 NavMesh 的候选视点物理可行性与可达性过滤；
4. **Visibility & View Quality Evaluation**: 多维视点观测质量基础评价（距离、视角夹角、可见性判定、遮挡率）；
5. **Viewpoint Dataset Generation**: 生成包含候选视点集合与多视角观测质量的结构化科研数据集。

### 明确排除在 v8.0 之外的非目标 (Explicitly NOT in v8.0):
- ❌ 不实现 Active View Selection (NBV / 动作感知主动视角选择算法)
- ❌ 不实现 深度学习 / 强化学习 (RL / View utility learning)
- ❌ 不实现 机器人闭环自主导航与避障路径规划 (Closed-loop Navigation)
- ❌ 不实现 动作识别模型训练 (Action Recognition / HAR)
- ❌ 不修改 `ea_avs_mvp_v7/` 及历史版本代码

---

## 5. Engineering Constraints & Operational Boundary
1. **数据边界保护**：严禁将 AMASS / BABEL 原始大型数据（npz, tar.bz2, images, pth）提交至 Git 仓库；
2. **统一路径解析**：统一使用 `../../data/ActiveView` 或 `ACTIVEVIEW_DATA_ROOT`，严禁开发机路径硬编码；
3. **只读保护**：保持 `ea_avs_mvp_v7/` 及更早版本只读；
4. **模块化复用**：v8 通过导入接口调用 v7 仿真底座，杜绝大范围代码重复。
