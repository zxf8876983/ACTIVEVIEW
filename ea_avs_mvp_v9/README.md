# ACTIVEVIEW v9.1: Human-State-Aware Learnable Active View Selection

> **ACTIVEVIEW v9.1 Scientific Specification & Experimental Closure Guide**  
> *Learning active observation utility directly from human physical state $H$ (SMPL-X 3D joints and body orientation) and candidate viewpoint geometry $v$: $\hat{Q}(v \mid H)$.*

---

## 1. 版本定位与核心演进 (Version Overview & Evolution)

- **v9.0 (Heuristic Rule Baseline)**：基于人工规则先验知识库 (`action_prior.yaml`) 的动作条件化打分基准；
- **v9.1 (Human-state-aware Learnable Active View Selection)**：将视角打分数学目标重构为：
  $$Q(v \mid H)$$
  其中 $H$ 为人体空间物理姿态状态（16 骨骼关键点 3D 相对坐标 + 偏航朝向角），$v$ 为候选视角。
  **严禁将 Action Label 作为神经网络模型输入**。Action Label 仅用于数据集分类、离线统计与跨动作泛化评测。

### 核心科研目标 (Core Research Objective):
在已知人体位置和候选视角集合情况下，机器人根据**人体当前解剖姿态与关键部位可见性**，学习选择最有信息价值的观察视角。

### 明确排除的非目标 (Explicit Non-Goals):
- ❌ **No Action Input to Model**: 动作标签不输入模型；
- ❌ **No Action Recognition / Classification**: 不训练动作分类网络；
- ❌ **No Pose Estimator**: 在仿真环境中使用真实 Ground-Truth 人体状态；
- ❌ **No SLAM / Global Navigation**: 聚焦于已知人体周边局部候选视角的高效选择；
- ❌ **No RL / Heavy Foundation Models**: 采用轻量 MLP 架构，验证学习型打分的有效性。

---

## 2. 神经网络架构 (Model Architecture)

```text
Human Pose State (16 Joints 3D + Yaw, 49d) ───► [HumanPoseEncoder]   (32d) ┐
                                                                              ├─► [Fusion MLP] ──► Scalar Score Q_hat(v|H) ∈ [0, 1]
Candidate View Features (13d, incl. 7 parts)───► [ViewFeatureEncoder] (32d) ┘   (64d -> 64d -> 32d -> 1d)
```

- **HumanPoseEncoder** (`models/pose_encoder.py`)：$49\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
- **ViewFeatureEncoder** (`models/view_encoder.py`)：$13\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
  - 13 维特征包含：`distance`, `sin(angle)`, `cos(angle)`, `pose_coverage`, `visibility_loss`, `projected_area`, 及 **7 大身体关键解剖部位可见性** (`head`, `torso`, `pelvis`, `left_hand`, `right_hand`, `left_leg`, `right_leg`)；
- **Fusion MLP** (`models/view_scorer.py`)：$(32 + 32 = 64)\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d} \rightarrow 1\text{d}$ (Sigmoid 激活)。

---

## 3. 训练目标与损失函数 (Training Objective & Loss)

### 科学目标效用函数 (Ground-Truth Utility):
$$Q^*(v) = w_1 \cdot \text{global\_visibility} + w_2 \cdot \text{pose\_coverage} + w_3 \cdot \text{body\_part\_visibility} - w_4 \cdot \text{distance\_penalty}$$
其中 $w_1 = 0.35, w_2 = 0.35, w_3 = 0.20, w_4 = 0.10$。

### 排序损失 (Pairwise Ranking Loss):
$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max\left(0, -\text{sign}(y_i - y_j)(\hat{s}_i - \hat{s}_j) + m\right)$$
辅以平滑 L1 回归损失，以确保候选视角的精准排序与数值校准。

---

## 4. 五大基线对比体系 (Five Baselines)

1. **Random View**: 随机选择可行视点；
2. **Nearest View**: 选择离人体最近的可行视点；
3. **Geometry Best (v8 Baseline)**: 基于纯几何可见性与距离选择最优视点；
4. **Rule-based (v9.0 Baseline)**: 基于可解释先验规则打分 $Q(v \mid a)$；
5. **Learnable (v9.1 Ours)**: 基于人体状态感知的神经网络预测得分 $\hat{Q}(v \mid H)$。

---

## 5. 快速运行指令 (Execution Guide)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 训练 v9.1 人体状态感知打分模型:
```bash
python -m ea_avs_mvp_v9.scripts.train_v91 --epochs 40 --lr 0.001
```
- 输出检查点：`data/ActiveView/checkpoints/model_checkpoint.pth`
- 输出收敛曲线：`data/ActiveView/results/training_curve.png`

### 3. 运行端到端 5 基线对比演示:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo --action sitting
```
- 输出产物：
  - `data/ActiveView/visualizations/v91_demo/best_view_rgb.png`
  - `data/ActiveView/visualizations/v91_demo/best_view.json`
  - `data/ActiveView/visualizations/v91_demo/comparison_report.json`
  - `data/ActiveView/results/body_part_visibility_report.json`

### 4. 运行空间布局与遮挡鲁棒性实验:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_occlusion_experiment
```
- 输出报表：`data/ActiveView/results/v91_occlusion_experiment_report.json`
