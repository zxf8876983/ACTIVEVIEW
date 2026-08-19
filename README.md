# ACTIVEVIEW (EA-AVS)

> **Elderly Action Active View Selection for Mobile Assistive Robots**  
> 面向居家养老助老场景下老人动作感知的移动机器人主动视角/位姿选择科研项目。

---

## 1. 项目简介 (Project Overview)

`ACTIVEVIEW`（内部代号 `EA-AVS`）旨在研究移动机器人在室内仿真环境（Habitat-Sim）中，如何根据人体先验、室内地图几何与环境遮挡，主动选择下一最佳观察位姿（Next-Best-View, NBV），以最大化老人动作关键部位的可见性、最小化环境与自遮挡，为下游人体姿态估计与动作识别提供高质量视觉观测。

---

## 2. 版本演进与目录组织 (Version Progression)

项目当前处于科研原型 MVP 快速迭代阶段，采用独立版本目录与独立 Version Specification 规范文档并存的架构组织：

```text
v1.0 (MVP 0.1)   -> 几何主动视角采样与可见性闭环
      ↓
v2.0 (MVP 2.0)   -> 确立 Pred / True 信息边界，禁止未来图像泄漏
      ↓
v3.0 (MVP 3.0)   -> 多姿态人体骨架、人体朝向建模与动作部位加权
      ↓
v4.0 (MVP 4.0)   -> 物理射线遮挡检测、遮挡感知评分与同口径 Oracle 上界
      ↓
v5.0 (MVP 5.0)   -> 真实 Habitat Humanoid、RGB-D 渲染与 5 分类遮挡闭环
      ↓
v6.0 (MVP 6.0)   -> 当前 RGB-D 驱动的 Estimated-State 姿态估计与主动视角选择闭环 (Closed)
      ↓
未来版本 (v7+)   -> 动作假设不确定性建模与下游动作识别增益评测
```

### 目录与文档对照

| 版本 | 代码目录 | 规范文档 | 说明 |
|---|---|---|---|
| **v6.0** | **`ea_avs_mvp_v6/`** | **`EA_AVS_MVP60_Code_Generation_Document.md`** | **当前实现 (Active / Closed)** |
| v5.0 | `ea_avs_mvp_v5/` | `EA_AVS_MVP50_Code_Generation_Document.md` | 历史稳定快照（只读） |
| v4.0 | `ea_avs_mvp_v4/` | `EA_AVS_MVP40_Code_Generation_Document.md` | 历史快照（只读） |
| v3.0 | `ea_avs_mvp_v3/` | `EA_AVS_MVP30_Code_Generation_Document.md` | 历史快照（只读） |
| v2.0 | `ea_avs_mvp_v2/` | `EA_AVS_MVP20_Code_Generation_Document.md` | 历史快照（只读） |
| v1.0 | `ea_avs_mvp/` | `EA_AVS_MVP01_Project_Document.md` | 历史快照（只读） |

> **提示**：
> - 活跃版本与开发状态由 `.ai/PROJECT_STATE.md` 动态指定。
> - 历史版本目录保持只读。完整科学闭环跑通前暂不重构为单一 `src/`。

---

## 3. 本地运行时数据分离 (Local Runtime Data)

为保持 Git 源码仓库精简，项目实行源码与运行时数据物理分离：

- **Git 源码仓库 (Source Repository)**：
  `/home/zxf/WorkSpace/code/code/ActiveView/`
- **运行时数据根目录 (Runtime Data Root)**：
  `/home/zxf/WorkSpace/code/data/ActiveView/`
- **推荐环境变量**：
  ```bash
  export ACTIVEVIEW_DATA_ROOT=/home/zxf/WorkSpace/code/data/ActiveView
  ```

GitHub 仓库仅保存代码、配置、测试、科研文档与必要小型元数据，不保存：
- `datasets`（数据集）
- `cache`（缓存）
- `checkpoints`（模型权重）
- `runs`（大规模实验输出与评测结果）
- `visualizations`（批量渲染图与视频）
- `logs` / `tmp`（运行时日志与临时文件）

---

## 4. AI Coding Agent 指南 (For AI Coding Agents)

所有代码模型（Codex, DeepSeek, Claude Code, Gemini 等）接手本仓库时，**请按以下顺序读取上下文，严禁逐版本全量扫描历史目录**：

```text
1. Read AGENTS.md
2. Read .ai/PROJECT_STATE.md
3. Check .ai/CURRENT_TASK.md
4. Check .ai/HANDOFF.md
5. Read the active version specification (specified in .ai/PROJECT_STATE.md)
6. Inspect only task-relevant code
```

### 核心科研约束 (Scientific Invariants)
- **最小修改原则**：只改动任务范围内的代码，不擅自重构或合并历史版本。
- **信息边界防护**：在线选择阶段（`Q_pred`）严禁渲染或偷看候选点未来 RGB/Depth。
- **实验协议保护**：严禁私自放宽评测阈值或篡改 Oracle / Baseline 科学定义。

---

## 5. 上下文基础设施说明 (Context Infrastructure)

仓库根目录维护了一套轻量、模型无关的 AI 上下文基础设施：

- **`AGENTS.md`**：所有 AI Agent 的统一第一入口、阅读顺序、数据边界与科研行为守则。
- **`.ai/PROJECT_STATE.md`**：项目当前科研状态记忆（活跃版本、已实现能力、未决问题、科学约束）。
- **`.ai/CURRENT_TASK.md`**：可选的任务持久化槽位与标准任务模板。
- **`.ai/HANDOFF.md`**：跨模型切换、额度耗尽或会话中断时的标准交接记录。

---

## 6. 快速运行与验证 (Quick Start)

### 6.1 纯 Python 单元与有效性测试 (无需 Habitat 仿真)
```bash
python ea_avs_mvp_v6/scripts/test_v60_pure_python.py
python ea_avs_mvp_v6/scripts/test_no_gt_leakage.py
```

### 6.2 语法与编译检查
```bash
python -m compileall ea_avs_mvp_v6
```

### 6.3 运行 v6.0 主实验 (需 Habitat-Sim 及资产环境)
```bash
python ea_avs_mvp_v6/scripts/run_mvp60_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 20
```
