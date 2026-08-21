# ACTIVEVIEW Current Task

Status: COMPLETED (v9.1 Perception-Aware Active View Selection Finalized & Verified)

## Current Phase
v9.1 — Perception-Aware Active View Selection ($\hat{G}(v \mid O_t)$).

## Accomplished in Final Scientific Optimization
1. **严格信息边界与数据流重构**：
   - 彻底删除模型对 GT 姿态、SMPL-X 真值与 Action 标签的依赖；
   - 模型输入仅接收由 `ObservationSimulator` / 感知模块生成的估计观测状态 $O_t$（71 维：48d 估计坐标 + 16d 关节点置信度 + 7d 身体部位可见置信度）；
   - 抽象出 `BaseObservationProvider`，为 v10.0 RGB Pose Estimator 提供规范扩展接口。
2. **5 大感知退化场景基准实验 (Scenario A~E)**：
   - Scenario A: 无遮挡低噪声 (Clean view)
   - Scenario B: 人体自遮挡 (Self-occlusion)
   - Scenario C: 家具障碍物遮挡 (Furniture occlusion)
   - Scenario D: 严重姿态估计噪声 (Severe noise)
   - Scenario E: 关键部位缺失 (Missing keypoints)
   - 输出至 `experiments/v9.1_validation/perception_degradation_report.json`。
3. **6 大方法对比与 Oracle 理论上限**：
   - 实现 `OracleViewEvaluator` (`evaluation/oracle_evaluator.py`)；
   - 建立包含 Random, Nearest, Geometry v8, Rule v9.0, Perception-aware v9.1, Oracle Upper Bound 的横向对比；
   - 输出至 `experiments/v9.1_validation/comparison_report.json` 与 `oracle_report.json`。
4. **完整实验报告与图表产物**：
   - 输出 `V91_FINAL_REPORT.md`, `README.md`, `information_gain_report.json`, `ablation_report.json`；
   - 包含 4 张科研图表：`training_curve.png`, `viewpoint_ranking.png`, `best_view_examples.png`, `body_visibility_analysis.png`。
5. **单元测试与代码质量**：
   - 34/34 v9.1 单元测试全部通过，17/17 v8 测试通过，30/30 v7 测试通过。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v9` (PASS)
- [x] v9.1 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 34/34)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] Full Validation Suite: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_validation_suite` (PASS)
- [x] Demo Script: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo` (PASS)
