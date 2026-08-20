# ACTIVEVIEW Current Task

Status: COMPLETED (v8.0 Local Active View Planning Framework)

## Current Phase
v8.0 Phase 1 — Local Active View Planning (Architecture, Constraint Pipeline & Quality Evaluation Completed)

## Current Objective
重新调整 ACTIVEVIEW v8.0 设计以符合科研目标 —— **Local Active View Planning**：
1. **科研假设与边界明确**：
   - 目标老人/人体位置已知 (Known Human Location)；
   - 移动机器人已到达人体局部附近区域 (Robot in Human Vicinity)；
   - v8 仅负责在人体周围局部约束空间内寻找高质量观察视角，杜绝全局盲目探索与搜索。
2. **Local Candidate View Generation** (`viewpoint_generator.py`)：
   - 以人体为中心进行极坐标规则采样 (1.5m - 3.0m，0-360度，朝向正对人体)。
3. **View Constraint Pipeline** (`constraints/`)：
   - `NavigationConstraint`: NavMesh 可行域与路径可达性检查；
   - `LineOfSightConstraint`: 视线光线追踪通视性与硬阻挡检查；
   - `HumanVisibilityConstraint`: 人体是否进入相机视锥 FOV 检查。
4. **View Quality Evaluation & Ranking** (`evaluation/view_quality.py`)：
   - 科学打分公式：$w_1 \cdot \text{human\_visibility} + w_2 \cdot \text{pose\_coverage} - w_3 \cdot \text{distance\_penalty}$；
   - 全局视点降序排序与最优视角选择。
5. **Rendering & Demo Output** (`scripts/run_v8_demo.py`)：
   - 移动机器人至最优视角渲染高质量 `best_view_rgb.png`；
   - 输出 `candidate_views.json` (含可行性状态) 与 `best_view.json` (最优位姿与各项得分)。

## Explicit Prohibitions & Non-Goals (严格遵守)
- ❌ 不实现 人体全局搜索与环境探索 (No Human Search / No Exploration)
- ❌ 不实现 NBV 复杂策略选择 / 动作感知网络
- ❌ 不实现 深度强化学习 (RL)
- ❌ 不实现 全局闭环自主导航规划
- ❌ 不修改 `ea_avs_mvp_v7/` 及历史版本代码
- ❌ 不将 AMASS / BABEL 原始大型数据提交至 Git

## Validation Plan Execution Results
- [x] v8 View Quality 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_view_quality.py` (PASS)
- [x] v8 Constraint Checker 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_constraint_checker.py` (PASS)
- [x] v8 纯 Python 单元/冒烟测试集: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 12/12)
- [x] v8 Local Active View Planning 演示入口: `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo` (PASS, best score=0.796, best_view_rgb.png verified)
- [x] v7 回归校验: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
