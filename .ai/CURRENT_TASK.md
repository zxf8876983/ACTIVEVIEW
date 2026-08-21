# ACTIVEVIEW Current Task

Status: COMPLETED (v9.1 Perception-Aware Active View Selection Completed & Verified)

## Current Phase
v9.1 — Perception-Aware Active View Selection ($\hat{G}(v \mid O_{\text{curr}})$).

## Accomplished in Perception-Aware Redesign
1. **输入输出重塑为当前观测感知状态 ($O_{\text{curr}}$)**：
   - 彻底删除模型对 GT 姿态、SMPL-X 真值与 Action 标签的依赖；
   - 引入 71 维感知特征：16 关节估计坐标 (48d) + 16 关节置信度 (16d) + 7 大部位可见置信度 (7d)；
   - 新增 `ObservationEncoder` (`models/observation_encoder.py`)；
   - 学习目标重构为信息增益 $\text{Gain}(v) = \Delta \text{ObservationQuality}$。
2. **感知退化模拟与 4 大基准评测**：
   - 新增 `ObservationSimulator` (`features/observation_simulator.py`)，模拟遮挡衰减、定位噪声与关节点缺失；
   - 执行 4 大退化场景评测：无遮挡、人体自遮挡、家具遮挡、低置信度；
   - 产物输出至 `experiments/v9.1_validation/perception_degradation/degradation_report.json`。
3. **5 大基线综合横向对比与信息增益报表**：
   - 对比 Random, Nearest, Geometry-based (v8), Rule-based (v9.0), Perception-aware (v9.1)；
   - 产物输出至 `experiments/v9.1_validation/baseline/comparison_report.json` 与 `experiments/v9.1_validation/information_gain/gain_report.json`。
4. **4 项特征消融与可视化图表**：
   - 消融实验输出至 `experiments/v9.1_validation/ablation/ablation_report.json`；
   - 输出 `training_curve.png`, `viewpoint_ranking.png`, `best_view_examples.png`, `body_visibility_analysis.png`；
   - 生成详尽总结报告 `experiments/v9.1_validation/V91_EXPERIMENT_REPORT.md` 与 `README.md`。
5. **单元测试与代码质量**：
   - 29/29 v9.1 单元测试通过，17/17 v8 测试通过，30/30 v7 测试通过。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v9` (PASS)
- [x] v9.1 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 29/29)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] Full Validation Suite: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_validation_suite` (PASS, all reports and figures generated)
- [x] Demo Script: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo` (PASS)
