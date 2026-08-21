# ACTIVEVIEW v9.0: Action-conditioned Active View Scoring

> **ACTIVEVIEW v9.0 Scientific Baseline Specification & Experimental Closure Guide**  
> *The optimal observation viewpoint is not only determined by geometry, but also depends on the current human activity state: $Q(v \mid A) \neq Q(v)$.*

---

## 1. 科研定位与研究目标 (Scientific Scope & Objectives)

ACTIVEVIEW v9.0 建立了首个面向动作状态的主动视角选择可解释基准（**Action-conditioned Active View Scoring Baseline**）。

### 核心科研假设 (Core Hypothesis):
在老人监护与物理感知任务中，不同的行为动作对观察视角具有截然不同的先验需求与解剖关注点：
$$Q(v \mid A) \neq Q(v)$$
$$v^* = \arg\max_{v \in \mathcal{V}} Q(v \mid H, A, E)$$
其中：
$$Q(v \mid A) = w_{\text{geom}} \cdot Q_{\text{geom}}(v) + w_{\text{act}} \cdot \Delta Q(A, v)$$

- **Fall (摔倒)**：优先保证全身轮廓、骨盆高度与地面接触关系的侧前方/侧面观察 ($20^\circ \sim 75^\circ$)，最优距离 $2.5\text{m}$；
- **Sitting (静坐)**：优先观察下肢膝关节弯曲与座椅表面交互 ($40^\circ \sim 90^\circ$)；
- **Standing (站立)**：优先保证躯干与面部正面端正观测 ($0^\circ \sim 30^\circ$)；
- **Bending (弯腰)**：优先侧向捕获脊柱弯曲度与躯干前倾剖面 ($45^\circ \sim 90^\circ$)；
- **Reaching (伸手)**：优先侧前方观察手臂前伸与抓取轨迹 ($15^\circ \sim 60^\circ$)。

### 明确排除的非目标 (Explicit Non-Goals):
- ✗ **No Deep Learning / RL**: 不训练神经网络或 RL policy，保持透明、可解释的先验规则打分；
- ✗ **No Action Recognition Model**: 直接接收动作状态输入，不训练动作分类器；
- ✗ **No Global Search / Navigation**: 聚焦于已知人体周边的局部主动视点优化。

---

## 2. 方法流程与架构组织 (Methodology & Pipeline)

```text
Human State (3D Joints) + Action Label (YAML Prior)
                  │
                  ▼
   1. Action Prior Encoding (ActionEncoder)
      - Vector: One-hot representation
      - Prior: critical_regions, preferred_angle_range, optimal_distance
                  │
                  ▼
   2. View Feature Extraction (ViewFeatureExtractor)
      - Distance, viewing angle, full-body pose coverage, visibility loss
      - Region coverages: head, torso, pelvis, upper_body, lower_body
                  │
                  ▼
   3. Action-conditioned Scoring (ActionConditionedScorer)
      - Q_geom(v): v8 geometry score
      - Delta_Q(A, v): RegionScore + AspectScore + DistanceScore
      - Q(v|A) = w_geom * Q_geom(v) + w_act * Delta_Q(A, v)
                  │
                  ▼
   4. Four Baseline Selection & Evaluation (ViewpointSelector)
      - Random View / Nearest View / Geometry Best (v8) / Action-conditioned (v9)
                  │
                  ▼
   5. Best Viewpoint Rendering & Quantitative Reports
```

---

## 3. 模块目录结构 (Module Map)

```text
ea_avs_mvp_v9/
├── configs/
│   ├── v9_demo.yaml            # 演示与传感器配置文件
│   ├── action_prior.yaml       # 动作解剖关键部位与视角先验知识库 (解耦配置)
│   └── v9_experiment.yaml      # 批量对比与消融实验参数
├── core/
│   ├── paths.py                # 物理数据根路径解析
│   ├── types.py                # ActionEmbedding, ViewFeature, ActionViewpointScore
│   └── config.py               # 统一配置加载器
├── action/
│   ├── action_types.py         # 动作分类与别名标准化 (支持 BABEL/AMASS)
│   └── action_encoder.py       # YAML 驱动的动作编码器 (无 Python 硬编码)
├── features/
│   └── view_feature_extractor.py # 16 关节解剖身体分区覆盖率提取器
├── scoring/
│   └── action_scorer.py        # Q(v|a) 综合质量评价器
├── selection/
│   └── viewpoint_selector.py   # 四基线视点选择调度器
├── dataset/
│   └── v9_dataset_loader.py    # v8 episode 数据读取与 action.json 导出
├── evaluation/
│   ├── action_metrics.py       # 动作特定指标评测 (关键区域覆盖率、朝向对齐度)
│   └── baseline_comparison.py  # 4 策略横向定量对比生成器
├── visualization/
│   ├── action_view_plotter.py      # 报表格式化与终端表格打印
│   └── action_comparison_plotter.py # 多动作对比图与消融图绘制器
├── scripts/
│   ├── run_v9_demo.py              # 端到端单动作 Demo 运行入口
│   ├── run_action_comparison.py    # 核心实验：5 类动作最佳视角对比 (证明 Q(v|A) != Q(v))
│   ├── run_weight_ablation.py      # 权重敏感性与消融实验 (0.8/0.2 ~ 0.2/0.8)
│   └── compare_baselines.py        # 纯 Python 批量基线对比
└── tests/
    └── unit/                       # 11 个单元测试集
```

---

## 4. 实验复现与快速运行 (Experiments & Quick Start)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 运行端到端视点选择与图像渲染 Demo:
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v9_demo --action fall
```

### 3. 运行多动作对比核心实验 ($Q(v \mid A) \neq Q(v)$):
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_action_comparison
```
- 输出文件：`data/ActiveView/results/v9_action_comparison.json` 与 `action_comparison.png`。

### 4. 运行权重敏感性与消融实验 ($w_{\text{geom}} / w_{\text{act}}$):
```bash
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_weight_ablation
```
- 输出文件：`data/ActiveView/results/weight_ablation_report.json`。

---

## 5. 核心实验结论 (Experimental Results Summary)

1. **视角迁移现象 (Viewpoint Shift)**：
   - 在 `SITTING` 与 `BENDING` 动作下，最佳视点从纯正面（$1^\circ$）显著迁移至侧前方（$44^\circ$），以便捕获下肢弯曲或脊柱剖面前倾；
   - 动作感知增益达 $+0.010 \sim +0.020$。
2. **权重敏感性 (Weight Ablation)**：
   - 当 $w_{\text{act}} \ge 0.4$ 时，动作先验对视角选择起到关键主导作用；在 $0.4/0.6$ 强动作感知下，`FALL` 动作选择的最优距离从 $2.0\text{m}$ 自动扩展至 $2.5\text{m}$ 以覆盖全身展开轮廓。
