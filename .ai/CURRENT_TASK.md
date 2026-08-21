# ACTIVEVIEW Current Task

Status: COMPLETED (v9.0 Action-conditioned Active View Scoring Final Baseline & Experimental Closure)

## Current Phase
v9.0 — Action-conditioned Active View Scoring (Scientific baseline completed, experimentally verified, and ready for future learning-based extensions in v9.1+)

## Accomplished in v9.0 Finalization
1. **动作先验完全配置化解耦 (`configs/action_prior.yaml`, `action/action_encoder.py`)**：
   - 将 `region_weights`, `preferred_angle_range`, `optimal_distance`, `aspect_weight`, `distance_weight` 从 Python 迁移至 `configs/action_prior.yaml`；
   - `ActionEncoder` 彻底移除硬编码规则，纯配置驱动生成 `ActionEmbedding`。
2. **核心科学命题实验闭环 (`scripts/run_action_comparison.py`)**：
   - 证明 $Q(v \mid A) \neq Q(v)$：在同一场景和相同人体位置下，`sitting` 和 `bending` 动作最佳视角成功由纯正面（$1^\circ$）迁移至侧前方（$44^\circ$），视角增益提升；
   - 生成定量成果 `data/ActiveView/results/v9_action_comparison.json` 与 `action_comparison.png`。
3. **四大基线对比体系 (`evaluation/baseline_comparison.py`)**：
   - 对比 `Random View`, `Nearest View`, `Geometry Best (v8)`, `Action-conditioned (v9)`；
   - 统计 `pose_coverage`, `critical_region_coverage`, `visibility_loss`, `distance_error`, `total_score`。
4. **权重敏感性与消融实验 (`scripts/run_weight_ablation.py`)**：
   - 遍历测试 $w_{\text{geom}} / w_{\text{act}}$ 在 $0.8/0.2, 0.6/0.4, 0.4/0.6, 0.2/0.8$ 的表现；
   - 生成消融分析报表 `data/ActiveView/results/weight_ablation_report.json`。
5. **多维可视化与报表模块 (`visualization/action_comparison_plotter.py`)**：
   - 绘制视角迁移图、得分分解图与基线对比图。
6. **文档与代码规范完善 (`ea_avs_mvp_v9/README.md`)**：
   - 补充 v9.0 研究目标、方法流程、实验方法与复现指令。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v9` (PASS)
- [x] v9.0 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 11/11)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] v9.0 Multi-Action Comparison: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_action_comparison` (PASS, results & plots generated)
- [x] v9.0 Weight Ablation: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_weight_ablation` (PASS, ablation report generated)
- [x] v9.0 Single Demo: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v9_demo --action fall` (PASS, RGB image & JSONs verified)
