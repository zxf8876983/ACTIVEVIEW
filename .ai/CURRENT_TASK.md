# ACTIVEVIEW Current Task

Status: COMPLETED (v8.0 Local Active View Planning Baseline Finalized)

## Current Phase
v8.0 — Local Active View Planning Baseline Finalization

## Accomplished Objectives
1. **View Quality 增加遮挡惩罚** (`evaluation/view_quality.py`)：
   - 评价公式：$Q(v) = w_1 \cdot \text{human\_visibility} + w_2 \cdot \text{pose\_coverage} - w_3 \cdot \text{distance\_penalty} - w_4 \cdot \text{occlusion\_penalty}$；
   - 权重配置：$w_1=0.40, w_2=0.30, w_3=0.15, w_4=0.15$。
2. **Human Visibility Constraint 增强** (`constraints/human_visibility_constraint.py`)：
   - 增加人体投影面积占比（`projected_area_ratio` $\ge 0.015$）判定；
   - 提供 16 关键点区域可见性判定接口 `pose_visibility_score()`。
3. **实验统计数据与输出规范** (`constraints/constraint_checker.py`, `scripts/run_v8_demo.py`)：
   - 输出 `candidate_statistics.json`（包含 `total_candidates`, `navmesh_valid`, `line_of_sight_valid`, `human_visible`, `feasible_candidates`, `selected_view`, `strategy`）；
   - 输出 `candidate_views.json`、`best_view.json` 与 `best_view_rgb.png`。
4. **Baseline 视角选择策略接口** (`evaluation/baseline_strategies.py`)：
   - 实现 `select_view(strategy)`，支持 `Random View`、`Nearest View`、`Geometry Best View` 三种标准基线。

## Explicit Prohibitions & Non-Goals (严格遵守)
- ❌ 不实现 人体全局搜索与环境探索 (No Human Search / No Exploration)
- ❌ 不实现 NBV 复杂策略选择 / 动作感知网络 (留待 v9)
- ❌ 不实现 深度强化学习 (RL)
- ❌ 不实现 全局闭环自主导航规划
- ❌ 不修改 `ea_avs_mvp_v7/` 及历史版本代码
- ❌ 不将 AMASS / BABEL 原始大型数据提交至 Git

## Validation Plan Execution Results
- [x] v8 Baseline Strategies 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_baseline_strategies.py` (PASS, 4/4)
- [x] v8 View Quality 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_view_quality.py` (PASS, 4/4)
- [x] v8 Constraint Checker 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_constraint_checker.py` (PASS, 2/2)
- [x] v8 全量单元测试集: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7 回归测试集: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] v8 Demo 端到端验证 (Geometry Best): `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy geometry_best` (PASS, score=0.697)
- [x] v8 Demo 端到端验证 (Nearest): `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy nearest` (PASS, score=0.643)
- [x] v8 Demo 端到端验证 (Random): `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy random` (PASS, score=0.585)
