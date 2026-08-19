# AGENTS.md

> **ACTIVEVIEW AI Coding Agent Entrance & Operational Protocol**  
> 本文件是所有代码模型（Codex, DeepSeek, Claude Code, Gemini 等）进入 `ACTIVEVIEW` 仓库时的第一入口与操作规范。

---

## 1. 强制阅读顺序 (Mandatory Reading Order)

新代码模型接手仓库时，**严禁从 v1 开始逐目录全量扫描历史代码**，请严格按照以下顺序加载上下文：

```text
1. AGENTS.md (本文件: 总体入口与科研约束)
2. .ai/PROJECT_STATE.md (项目短期状态记忆: 当前阶段、活跃版本、已实现能力与未决问题)
3. .ai/CURRENT_TASK.md (当前任务定义: 检查是否有待执行或持久化的具体任务)
4. .ai/HANDOFF.md (跨模型交接记录: 检查是否有前任 Agent 留下的未完成交接)
5. 当前活跃版本对应的 Version Specification MD (如 EA_AVS_MVP50_Code_Generation_Document.md)
6. 与当前任务直接相关的局部代码文件
```

> **Token 节约原则**：仅在需要明确追溯某项算法或历史设计演进缘由时，才按需查阅旧版本文档与历史目录。

---

## 2. 当前活跃版本与目录结构 (Active Version & Repository Map)

| 目录 / 文件 | 状态 | 说明 |
|---|---|---|
| **`ea_avs_mvp_v5/`** | **当前活跃实现 (Active)** | 真实 Humanoid + RGB-D 渲染 + GT-State 支路 + 遮挡感知主动视角选择 |
| **`EA_AVS_MVP50_Code_Generation_Document.md`** | **当前规范文档 (Active Spec)** | v5.0 设计规范、接口定义与严谨性要求 |
| `ea_avs_mvp/` ~ `ea_avs_mvp_v4/` | 历史版本 (Historical) | **只读 (Read-Only)**。历史科研原型快照，默认严禁修改 |
| `EA_AVS_MVP01` ~ `MVP40_Document.md` | 历史规范文档 | 只读历史参考 |
| `.ai/` | 上下文基础设施 | `PROJECT_STATE.md`, `CURRENT_TASK.md`, `HANDOFF.md` |

> **注意**：活跃版本由 `.ai/PROJECT_STATE.md` 中的声明动态指定。若未来升级至 v6/v7，以 `PROJECT_STATE.md` 中指向的 active version 为准，请勿在代码或提示词中将“永远是 v5”写死。

---

## 3. 科研开发核心规则 (Scientific Development Rules)

### 3.1 最小修改原则 (Minimal Modification Principle)
1. **理解先行**：先阅读相关代码与规范，再进行最小范围修改。
2. **拒绝顺手重构**：不做未经明确要求的目录移动、架构重写或“代码整洁度”整理。
3. **过渡期架构尊重**：项目处于科研原型探索期，在核心闭环跑通前保持 `v1 -> v2 -> ... -> v5 -> 未来v6` 的独立版本目录模式，**严禁私自将多版本合并为单一 `src/`**。

### 3.2 科研边界与实验协议保护 (Research Boundary Protection)
未经明确科研授权，**严禁自行变更以下任何科学定义与实验协议**：
* 科研问题定义（单步/One-shot 主动观察位姿重选择）；
* 数据划分与场景配置；
* 评价指标计算公式（Visibility, Action Part Score, Orientation Score, Occlusion Rate 等）；
* **Pred / True 信息边界**；
* **Oracle 离线上界定义**（必须在同 aperture / 同 depth 口径下比较）；
* 候选点采样空间（radii, angles, candidate aperture）；
* Baseline 策略定义（Fixed, Random, Nearest, 消融策略）；
* 随机种子策略与实验评测环境参数；
* **严禁为了获得“更漂亮的实验数字”而静默放宽或篡改实验协议与判定阈值**。

### 3.3 严格防止未来观测信息泄漏 (No Future Observation Leakage)
ACTIVEVIEW 是主动视角选择研究：
* **决策时（Decision-time）**：策略选择（尤其是 `OursPolicy`）只能使用预测评分 `Q_pred`（基于当前已知地图几何、先验或当前状态估计）。**严禁在选择候选点阶段调用渲染器偷看候选点的未来 RGB、Depth 或真实遮挡状态**。
* **评估时（Evaluation-time）**：选中位姿渲染后的真实观测（`*_true`）仅供后验评估与指标记录，绝不能反向影响决策。
* 若任务需求存在破坏该边界的风险，必须立即终止并向用户报告。

---

## 4. 修改后的验证要求 (Validation Requirements)

Agent 每次必须根据以下优先级确定当前任务的验证命令：
1. **`.ai/CURRENT_TASK.md`** 中显式声明的 Validation Plan
2. **`.ai/PROJECT_STATE.md`** 指向的 active version
3. **当前 active version 的 Version Specification**（设计文档中的验证要求）
4. **当前版本已有的 tests / smoke / debug 脚本**

### Current v5 Default Validation Commands
当前 v5.0 活跃版本下的默认验证命令示例：
1. **轻量语法与导入检查**：
   ```bash
   python -m compileall ea_avs_mvp_v5
   ```
2. **纯 Python 逻辑与策略单元测试**（无需 Habitat 物理引擎）：
   ```bash
   python ea_avs_mvp_v5/scripts/test_v50_closure_policy.py
   ```
3. **功能脚本 / Smoke Test / Debug 脚本**（在具备完整仿真环境时按需运行）：
   ```bash
   cd ea_avs_mvp_v5
   python scripts/run_mvp50_humanoid.py --config configs/mvp50_humanoid.yaml --episodes 1
   ```

### 诚实报告原则 (Honest Validation Reporting)
* 严禁声称执行了未实际运行的测试。
* 若因缺少 Habitat-Sim / GPU / 资源资产 / 依赖包导致无法运行某些测试，必须明确记录为：
  `NOT RUN: <具体原因>`。

---

## 5. 跨 Agent 交接规范 (Handoff Rules)

当遇到以下情况时：
* 即将切换代码模型（如 Claude Code → Codex / DeepSeek / Gemini 等）；
* 当前模型会话长度或调用额度即将耗尽；
* 任务较为复杂，需要分多轮执行；
* 遇到未解决的阻塞问题或临时中断；
* 用户明确要求生成交接记录；

**必须在退出前更新 `.ai/HANDOFF.md`**。请根据实际执行情况填写当前分支、HEAD、未提交变更、已完成事项、未执行的测试及下一步建议。任务彻底闭环后，交接状态应重置为 `Status: CLEAN`。

---

## 6. 上下文维护原则 (Context Maintenance Principles)

- **代码变化** -> 由 Git commit / commit history 记录
- **科研阶段 / active version / 关键能力变化** -> 更新 `.ai/PROJECT_STATE.md`
- **复杂或跨会话任务** -> 写入 `.ai/CURRENT_TASK.md`
- **任务中途切换模型或中断** -> 写入 `.ai/HANDOFF.md`
- **长期工作规则变化** -> 更新 `AGENTS.md`
> 普通小 bug 修复或单轮局部修改不需要同时维护所有上下文 Markdown 文件，避免过度维护开销。
