# ACTIVEVIEW v9.1: Perception-Aware Active View Selection

> **ACTIVEVIEW v9.1 Scientific Specification & Experimental Closure Guide**  
> *"Given incomplete human observations, the robot actively selects informative viewpoints to improve future human state estimation."*  
> *(在人体观测不完整条件下，机器人主动选择能够提升人体状态估计质量的观察视角。)*

---

## 1. 科研定位与核心假设 (Scientific Scope & Problem Formulation)

在真实室内主动感知环境中，由于**环境遮挡、人体自遮挡与视角局限**，机器人当前捕获的人体观测始终是不完整的，无法直接获得 Ground-Truth 人体状态或动作标签。

### 核心科学问题 (Core Research Problem):
机器人如何根据**当前不完整的观测感知质量 $O_{\text{curr}}$**（估计关节坐标、关节置信度与身体部位可见性），主动规划下一个观察视角 $v$，以**最大化未来的信息增益（Information Gain）与状态感知提升**：
$$v^* = \arg\max_{v \in \mathcal{V}_{\text{feasible}}} \hat{G}(v \mid O_{\text{curr}})$$

### 严格研究边界 (Strict Research Boundaries):
- ❌ **No GT Human State / No SMPL-X GT in Forward Pass**: 模型输入严格来自视觉估计感知状态；
- ❌ **No Action Label in Model Forward**: 动作标签严禁输入模型，仅用于离线实验分组与数据统计；
- ❌ **No Temporal Multi-step / RL / End-to-end Policy**: 严格聚焦于单步主动视角信息增益规划。

---

## 2. 感知状态与网络架构 (Perception State & Model Architecture)

### 当前观测状态特征向量 (Observation State $O_{\text{curr}}$, 71d):
1. **16 关节点估计 3D 坐标** (相对估计骨盆归一化坐标)：$16 \times 3 = 48\text{d}$；
2. **16 关节点估计置信度** ($c_i \in [0.0, 1.0]$)：$16\text{d}$；
3. **7 大身体关键解剖部位可见性置信度** (`head`, `torso`, `pelvis`, `left_hand`, `right_hand`, `left_leg`, `right_leg`)：$7\text{d}$。

### 候选视角描述子向量 (View Descriptor $v$, 13d):
包含视点空间距离、相对偏角 $\sin/\cos$、视锥几何关系等，**严禁输入未来真实观测状态（No Future Leakage）**。

### 神经网络架构:
```text
Observation State O_curr (71d) ──► [ObservationEncoder]   (32d) ┐
                                                                   ├─► [Fusion MLP] ──► Predicted Info Gain G_hat(v|O_curr) ∈ [0, 1]
Candidate View Descriptor v (13d) ──► [ViewFeatureEncoder] (32d) ┘   (64d -> 64d -> 32d -> 1d)
```

---

## 3. 训练目标与信息增益标签 (Information Gain Target)

真实目标由视角移动后的观测质量提升量决定：
$$\text{Gain}(v) = \max\left(0.0, \text{ObservationQuality}_{\text{after}}(v) - \text{ObservationQuality}_{\text{before}}(v_{\text{curr}})\right)$$
其中：
$$\text{ObservationQuality} = 0.50 \cdot \bar{c}_{\text{joints}} + 0.40 \cdot \bar{c}_{\text{parts}} - 0.10 \cdot \text{dist\_penalty}$$
采用 **Pairwise Ranking Loss** 进行排序训练，专注于发现信息增益最大的最佳视角。

---

## 4. 实验验证套件 (Validation Suite & Benchmarks)

所有实验结果均结构化保存在 `ea_avs_mvp_v9/experiments/v9.1_validation/`：
- **`training/`**: 40 轮损失下降与 Top-1 准确率日志；
- **`baseline/`**: 5 大基线（Random, Nearest, Geometry v8, Rule v9.0, Perception-aware v9.1）横向对比；
- **`perception_degradation/`**: 4 种感知退化基准实验（无遮挡、自遮挡、家具遮挡、低置信度）；
- **`information_gain/`**: 关节置信度提升量与缺失关节点恢复率分析；
- **`ablation/`**: 4 项特征消融实验；
- **`visualization/`**: 包含收敛曲线、信息增益排序极坐标图、置信度改善柱状图等 4 张科研高清图表；
- **`V91_EXPERIMENT_REPORT.md`**: 包含 10 项核心维度的完整科研报告。

---

## 5. 快速复现指令 (Execution Guide)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 运行完整实验验证闭环流水线:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_validation_suite
```

### 3. 运行端到端感知驱动演示:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v91_demo
```
