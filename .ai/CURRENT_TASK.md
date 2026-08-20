# ACTIVEVIEW Current Task

Status: COMPLETED (v8.0 Phase 1: Foundation Architecture Initialized)

## Current Phase
v8.0 Phase 1 — Build Human-aware Active View Foundation Architecture (Completed)

## Current Objective
构建 ACTIVEVIEW v8.0 基础模块架构与标准接口：
1. **v8 模块架构建立**：
   - 创建 `ea_avs_mvp_v8/` 目录结构（`configs/`, `core/`, `environment/`, `human/`, `robot/`, `viewpoint/`, `constraints/`, `visibility/`, `dataset/`, `evaluation/`, `scripts/`, `tests/`）；
   - 严格通过接口复用 `ea_avs_mvp_v7` 仿真能力（Habitat, Humanoid, Motion, Robot, RGB-D）；
2. **Human Placement 接口实现**：
   - 新增 `human/human_placement.py`（类 `HumanPlacement`：`sample_position()`, `validate_position()`）；
3. **Viewpoint 生成器接口实现**：
   - 新增 `viewpoint/viewpoint_generator.py`（类 `ViewpointGenerator`：多半径、多水平角度采样候选视点）；
4. **Constraint 约束检查器接口实现**：
   - 新增 `constraints/constraint_checker.py`（类 `ConstraintChecker`：基于 NavMesh 校验可行性与可达性）；
5. **Visibility 视点质量评价接口实现**：
   - 新增 `visibility/visibility_evaluator.py`（类 `VisibilityEvaluator`：基础几何与可见性打分）；
6. **v8 演示入口与测试**：
   - 新增 `scripts/run_v8_demo.py`（输出 `candidate_views.json`, `visibility.json`, `metadata.json` 至 `visualizations/v8_demo/`）；
   - 增加纯 Python 单元测试套件并全量通过。

## Explicit Prohibitions & Non-Goals (严格遵守)
- ❌ 不实现 NBV 视角选择算法 / 视点效用学习
- ❌ 不实现 Action-aware utility
- ❌ 不实现 强化学习 (RL)
- ❌ 不实现 机器人闭环自主导航与避障路径规划
- ❌ 不训练 动作识别模型 (HAR)
- ❌ 不修改 `ea_avs_mvp_v7/` 及历史版本代码
- ❌ 不将 AMASS / BABEL 原始大型数据提交至 Git

## Current Version Status
- **Active Development Version**: v8.0 (8.0.0 — Phase 1 Completed)
- **Active Specification**: `EA_AVS_MVP80_Code_Generation_Document.md`
- **Previous Stable Baseline**: v7.0 (7.0.0 — CLOSED / FINALIZED & FROZEN, Read-Only)

## Validation Plan Execution Results
- [x] v8 Human Placement 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_human_placement.py` (PASS)
- [x] v8 Viewpoint Generator 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_viewpoint_generation.py` (PASS)
- [x] v8 Constraint Checker 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_constraint_checker.py` (PASS)
- [x] v8 Visibility Evaluator 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_visibility.py` (PASS)
- [x] v8 纯 Python 单元/冒烟测试集: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 9/9)
- [x] v8 端到端演示入口: `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo` (PASS, 18/24 feasible views)
- [x] v7 回归校验: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
