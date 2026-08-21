# ACTIVEVIEW v9.1: Learnable Action-conditioned Active View Scoring

> **ACTIVEVIEW v9.1 Scientific Specification & Experimental Closure Guide**  
> *Learning action-conditioned active observation utility from human posture, action embedding, and candidate viewpoint geometry.*

---

## 1. 版本定位与核心演进 (Version Overview & Evolution)

- **v9.0 (Heuristic Baseline)**：基于人工规则知识库 (`action_prior.yaml`) 的动作感知打分基准；
- **v9.1 (Learnable Active View Scorer)**：将人工先验打分规则升级为**可训练的轻量神经网络打分模型 (MLP-based)**，使机器人能够从人体姿态、动作嵌入与候选视角几何特征中端到端学习视角的观测效用 $\hat{Q}(v \mid H, A)$。

### 核心科研假设 (Core Hypothesis):
$$\hat{Q}(v \mid H, A) = \mathcal{F}_{\theta}(\text{Pose}(H), \text{Action}(A), \text{View}(v))$$
通过 Pairwise Ranking Loss 训练的神经网络能够精准排序候选视点，并在保持规则可解释性优点的同时，自适应捕捉动作对视角的非线性偏好。

### 严格研究边界 (Strict Research Boundaries):
- ❌ **No Human Detection / Localization**: 假设粗糙人体位置已知；
- ❌ **No Action Recognition Model**: 动作状态直接由外部系统/标注提供；
- ❌ **No SLAM / Global Navigation**: 聚焦于已知人体周边候选视点的高效局部选择；
- ❌ **No Heavy Foundation / RL Models**: 采用轻量 MLP 架构，验证学习型打分的有效性。

---

## 2. 神经网络架构 (Model Architecture)

```text
Human Pose (16 Joints 3D + Yaw, 49d) ───► [HumanPoseEncoder]   (32d) ┐
                                                                       │
Action Embedding (One-hot, 5d)        ───► [ActionEncoder]     (16d) ─┼─► [Fusion MLP] ──► Scalar Score Q_hat ∈ [0, 1]
                                                                       │   (80d -> 64d -> 32d -> 1d)
Candidate View Features (11d)         ───► [ViewFeatureEncoder](32d) ┘
```

- **HumanPoseEncoder** (`models/pose_encoder.py`)：$49\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
- **ViewFeatureEncoder** (`models/view_encoder.py`)：$11\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
- **Fusion MLP** (`models/view_scorer.py`)：$(32 + 16 + 32 = 80)\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d} \rightarrow 1\text{d}$ (Sigmoid 激活)。

---

## 3. 五大基线对比体系 (Five Baselines)

1. **Random View**: 随机选择可行视点；
2. **Nearest View**: 选择离人体最近的可行视点；
3. **Geometry Best (v8 Baseline)**: 基于纯几何可见性与距离选择最优视点；
4. **Rule Action (v9.0 Baseline)**: 基于可解释先验规则打分 $Q(v \mid a)$；
5. **Learnable Action-conditioned (v9.1 Ours)**: 基于神经网络预测得分 $\hat{Q}(v \mid H, A)$。

---

## 4. 模块组织 (Module Structure)

```text
ea_avs_mvp_v9/
├── configs/
│   ├── v9_demo.yaml            # 默认演示配置
│   ├── action_prior.yaml       # 动作解剖关键部位配置
│   └── v91_training.yaml       # 模型训练超参数配置
├── models/
│   ├── pose_encoder.py         # 人体姿态 MLP 编码器
│   ├── view_encoder.py         # 视角几何多维特征 MLP 编码器
│   └── view_scorer.py          # 学习型动作感知视角综合打分网络
├── training/
│   ├── dataset.py              # ActiveViewScoringDataset 与自动生成器
│   ├── losses.py               # PairwiseRankingLoss 与 CombinedLoss
│   └── trainer.py              # ViewScorerTrainer 与收敛曲线绘制
├── inference/
│   └── predict_view.py         # 视点推理与排序引擎
├── scripts/
│   ├── train_v91.py            # 训练入口 (输出 model_checkpoint.pth 与 training_curve.png)
│   ├── run_v91_demo.py         # 5 大基线综合演示脚本 (输出 best_view_rgb.png)
│   ├── run_v91_ablation.py     # 三大消融实验脚本 (输出 v91_ablation_report.json)
│   └── run_action_comparison.py# v9.0 多动作对比脚本
└── tests/
    └── unit/                   # 23 个单元测试用例
```

---

## 5. 快速运行指令 (Execution Guide)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 训练 v9.1 视点打分模型:
```bash
python -m ea_avs_mvp_v9.scripts.train_v91 --epochs 40 --lr 0.001
```
- 输出检查点：`data/ActiveView/checkpoints/model_checkpoint.pth`
- 输出收敛图：`data/ActiveView/results/training_curve.png`

### 3. 运行端到端 5 基线对比演示:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo --action sitting
```

### 4. 运行特征与模型消融实验:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_ablation
```
- 输出报表：`data/ActiveView/results/v91_ablation_report.json`
