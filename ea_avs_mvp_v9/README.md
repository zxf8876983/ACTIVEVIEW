# ACTIVEVIEW v9.1: Human-State-Aware Learnable Active View Selection

> **ACTIVEVIEW v9.1 Scientific Specification & Experimental Closure Guide**  
> *Under the assumptions that the human position is known, the robot is in the vicinity, and the human remains observable, the robot learns to select the most informative observation viewpoint directly from the human's physical state: $\hat{Q}(v \mid H)$.*

---

## 1. 科研定位与核心假设 (Scientific Scope & Assumptions)

ACTIVEVIEW v9.1 正式确立了**人体物理状态驱动的学习型主动视角选择基准（Human-state-aware Learnable Active View Selection Baseline）**。

### 核心科研假设 (Core Assumptions):
1. **Human location is known**: 人体粗略位置由外部系统/定位已知；
2. **Robot is in vicinity**: 移动机器人已经到达目标人体所在局部区域；
3. **Human remains observable**: 机器人可自适应调整相机朝向使人体位于观察视锥内，核心研究目标为**选择最具解剖信息观测价值的拍摄视角**，而非视野寻人。

### 数学目标 (Mathematical Objective):
$$Q(v \mid H)$$
其中：
- $H$: 人体物理姿态状态（16 骨骼关键点 3D 相对坐标 + 偏航朝向角，49 维向量）；
- $v$: 候选视点描述子（包含距离、相对朝向角、全局覆盖率与 **7 大身体解剖部位可见性**，13 维向量）。

### 严格研究边界 (Strict Research Boundaries):
- ❌ **No Action Input to Model**: 动作标签严禁输入模型，仅用于实验分组与数据统计；
- ❌ **No Action Recognition / Classification**: 不训练动作识别网络；
- ❌ **No Human Search / Detection / Navigation / SLAM / RL**: 聚焦已知人体周边的局部高质量视角选择；
- ❌ **No Pose Estimator**: 仿真中使用 Habitat + SMPL-X 物理 GT 状态。

---

## 2. 神经网络架构 (Model Architecture)

```text
Human Pose State (16 Joints 3D + Yaw, 49d) ───► [HumanPoseEncoder]   (32d) ┐
                                                                              ├─► [Fusion MLP] ──► Scalar Score Q_hat(v|H) ∈ [0, 1]
Candidate View Features (13d, incl. 7 parts)───► [ViewFeatureEncoder] (32d) ┘   (64d -> 64d -> 32d -> 1d)
```

- **HumanPoseEncoder** (`models/pose_encoder.py`): $49\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
- **ViewFeatureEncoder** (`models/view_encoder.py`): $13\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d}$；
  - 包含几何信息（距离、相对偏角 $\sin/\cos$）与观测信息（全局覆盖率、覆盖损失、投影面积及 **7 大关键身体部位可见性**：`head`, `torso`, `pelvis`, `left_hand`, `right_hand`, `left_leg`, `right_leg`）；
- **Fusion MLP** (`models/view_scorer.py`): $(32 + 32 = 64)\text{d} \rightarrow 64\text{d} \rightarrow 32\text{d} \rightarrow 1\text{d}$ (Sigmoid 激活)。

---

## 3. 训练目标与科学真实效用 (Ground-Truth Utility)

训练标签严格基于物理几何与解剖可见性计算，**绝非模仿规则打分**：
$$Q^*(v) = w_1 \cdot \text{global\_visibility} + w_2 \cdot \text{pose\_coverage} + w_3 \cdot \text{body\_part\_visibility} - w_4 \cdot \text{distance\_penalty}$$
- 采用 **Pairwise Ranking Loss** 结合辅助 Smooth L1 损失，专注于候选视角的正确相对排序；
- 严格执行 **Spatial-Level / Instance-Level 数据集隔离划分**，严禁训练集与测试集共享相同位姿。

---

## 4. 五大基线对比体系 (Five Baselines)

1. **Random View**: 随机选择可行视点；
2. **Nearest View**: 选择离人体最近的可行视点；
3. **Geometry-based View Selection (v8)**: 纯几何可见性与距离驱动选择；
4. **Rule-based Task-aware View Selection (v9.0)**: 基于先验规则知识库的打分基线；
5. **Human-state-aware Learnable View Selection (v9.1 Ours)**: 基于纯人体姿态驱动的神经网络打分选择。

---

## 5. 实验复现与运行指令 (Execution Guide)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 训练 v9.1 人体物理状态感知打分模型:
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
