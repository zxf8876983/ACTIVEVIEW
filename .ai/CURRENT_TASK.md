# ACTIVEVIEW Current Task

Status: COMPLETED (v9.1 Human-state-aware Learnable Active View Selection Completed & Verified)

## Current Phase
v9.1 — Human-state-aware Learnable Active View Selection ($Q(v \mid H)$).

## Accomplished in v9.1 Redesign
1. **重塑网络输入与数学目标 ($Q(v \mid H)$)**：
   - 彻底移除 Action Label 作为模型输入；
   - `HumanPoseEncoder` (49d -> 32d): 16 骨骼关键点 3D 相对坐标 + 偏航角；
   - `ViewFeatureEncoder` (13d -> 32d): 包含距离、角度、覆盖率及 **7 大身体关键解剖部位可见性** (`head`, `torso`, `pelvis`, `left_hand`, `right_hand`, `left_leg`, `right_leg`)；
   - `LearnableViewScorer`: 纯姿态与视点特征融合，输出标量评分 $\hat{Q}(v \mid H) \in [0.0, 1.0]$。
2. **训练目标与排序损失 (`training/`)**：
   - 重构科学目标效用得分 $Q^*(v) = w_1 \cdot \text{global\_vis} + w_2 \cdot \text{pose\_cov} + w_3 \cdot \text{body\_part\_vis} - w_4 \cdot \text{dist\_pen}$；
   - 基于 `PairwiseRankingLoss` 训练，验证集 Top-1 选点准确率达 **95.0%**，目标效用比例达 **100.0%**；
   - 保存检查点 `checkpoints/model_checkpoint.pth` 与收敛曲线 `results/training_curve.png`。
3. **5 大基线综合评测 (`scripts/run_v91_demo.py`)**：
   - 对比 `Random`, `Nearest`, `Geometry Best (v8)`, `Rule-based (v9.0)`, `Learnable (v9.1)`；
   - 生成 `best_view.json`, `v91_best_view.json`, `comparison_report.json`, `body_part_visibility_report.json` 与渲染图像 `best_view_rgb.png`。
4. **空间布局与遮挡鲁棒性实验 (`scripts/run_v91_occlusion_experiment.py`)**：
   - 评测不同机器人初始站位与遮挡几何下的视点决策；
   - 输出 `results/v91_occlusion_experiment_report.json`。
5. **单元测试与代码质量**：
   - 23/23 v9.1 单元测试通过，17/17 v8 测试通过，30/30 v7 测试通过。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v9` (PASS)
- [x] v9.1 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 23/23)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] Model Training: `python -m ea_avs_mvp_v9.scripts.train_v91 --epochs 40` (PASS, 95.0% Acc)
- [x] 5-Baseline Demo: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo --action sitting` (PASS, all JSONs & RGB verified)
- [x] Occlusion Experiment: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_occlusion_experiment` (PASS, occlusion report generated)
