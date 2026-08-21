# ACTIVEVIEW Current Task

Status: COMPLETED (v9.1 Learnable Action-conditioned View Scoring Baseline Completed & Verified)

## Current Phase
v9.1 — Learnable Action-conditioned View Scoring (Neural scoring architecture, ranking loss trainer, inference engine, 5-baseline comparative evaluation, and feature ablations completed).

## Next Phase Target
v9.2 — Temporal Action-aware Active View Selection (Sequence/Trajectory-aware Viewpoint Planning).

## Accomplished in v9.1
1. **轻量神经网络打分架构 (`models/`)**：
   - `HumanPoseEncoder` (49d -> 32d): 16 骨骼关键点 3D 相对坐标 + 偏航角 MLP 编码；
   - `ViewFeatureEncoder` (11d -> 32d): 视点几何距离、角度、覆盖率、身体分区特征编码；
   - `LearnableViewScorer`: 姿态、动作 One-hot (16d) 与视角特征三向融合 MLP，输出标量连续打分 $\hat{Q}(v \mid H, A) \in [0.0, 1.0]$，支持特征消融开关。
2. **两两排序与回归损失训练管道 (`training/`, `scripts/train_v91.py`)**：
   - 实现了 `PairwiseRankingLoss` 与 `CombinedRankingRegressionLoss`；
   - 自动化生成多场景、多位姿、5 类动作训练集与验证集；
   - 40 轮训练后验证集 Top-1 选点准确率达 **97.5%**，目标效用比例达 **100.0%**；
   - 保存权重 `checkpoints/model_checkpoint.pth` 与收敛曲线 `results/training_curve.png`。
3. **前向推理与预测器 (`inference/predict_view.py`)**：
   - `ViewPredictor` 支持从检查点载入权重，对候选视点池进行批量前向预测与降序排序。
4. **5 大基线综合横向评测体系 (`scripts/run_v91_demo.py`)**：
   - 比较 `Random`, `Nearest`, `Geometry Best (v8)`, `Rule Action (v9.0)`, `Learnable Action (v9.1)`；
   - 移动机器人至神经网络选定视点并渲染 `best_view_rgb.png`；
   - 输出 `v91_best_view.json`, `comparison_report.json`, `comparison_with_v90.json`。
5. **三大特征消融实验 (`scripts/run_v91_ablation.py`)**：
   - 评测 Full Model, w/o Action, w/o Pose, 以及 Rule vs Learning；
   - 证实动作特征是驱动视角自适应偏转的关键因素；
   - 输出 `results/v91_ablation_report.json`。
6. **单元测试与代码质量**：
   - 编写 5 组针对 v9.1 的单元测试（23/23 PASS），v8（17/17 PASS），v7（30/30 PASS）。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v9` (PASS)
- [x] v9.1 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 23/23)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] v9.1 Model Training: `python -m ea_avs_mvp_v9.scripts.train_v91 --epochs 40` (PASS, 97.5% Acc)
- [x] v9.1 5-Baseline Demo: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo --action sitting` (PASS, RGB & JSONs verified)
- [x] v9.1 Feature Ablation: `conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_ablation` (PASS, ablation report generated)
