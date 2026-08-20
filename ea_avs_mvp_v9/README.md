# ACTIVEVIEW v9.0: Action-conditioned Active View Scoring

> **ACTIVEVIEW v9.0 Specification & Execution Guide**  
> *The optimal observation viewpoint is not only determined by geometry, but also depends on the current human activity state.*

---

## 1. 科研定位与核心假设 (Scientific Scope & Core Hypothesis)

### 核心科研假设 (Core Hypothesis):
纯几何视角规划方法假定 $Q(v)$ 仅由可见性与距离决定。然而在老人监护与主动感知任务中，不同的人体动作（如 *摔倒*、*静坐*、*站立*、*弯腰*、*伸手*）对观察视角有截然不同的生理与任务先验要求：

$$Q(v \mid A) \neq Q(v)$$

$$v^* = \arg\max_{v \in \mathcal{V}} Q(v \mid H, A, E)$$

其中 $Q(v \mid A) = w_{\text{geom}} \cdot Q_{\text{geometry}}(v) + w_{\text{act}} \cdot \Delta Q(A, v)$。

- **Fall (摔倒)**：优先保证全身轮廓、骨盆重心与地面接触关系的侧前方/侧面观察 ($30^\circ \sim 75^\circ$)；
- **Sitting (静坐)**：优先观察下肢膝关节弯曲与座椅交互 ($40^\circ \sim 90^\circ$)；
- **Standing (站立)**：优先保证躯干与面部正面端正观测 ($0^\circ \sim 30^\circ$)；
- **Bending (弯腰)**：优先侧向捕获脊柱弯曲度与躯干角度 ($45^\circ \sim 90^\circ$)；
- **Reaching (伸手)**：优先侧前方观察手臂前伸与抓取轨迹 ($15^\circ \sim 60^\circ$)。

---

## 2. 研究边界 (Research Boundaries)

### 包含的科研能力 (Included):
- ✓ 动作标签解析与先验嵌入 (`ActionEncoder`)；
- ✓ 视点多维身体分区特征提取 (`ViewFeatureExtractor`)；
- ✓ 动作条件视角综合打分 (`ActionConditionedScorer`)；
- ✓ 四大基线横向对比与量化评测 (`ViewpointSelector`, `baseline_comparison.py`)；
- ✓ 选定动作感知视角 RGB 渲染与结构化报表生成。

### 明确排除的非目标 (Explicitly Excluded):
- ✗ **No Human Search**: 不进行全场人体搜索；
- ✗ **No Global Navigation**: 不包含 SLAM 与全局路径规划；
- ✗ **No Action Recognition Model**: 不训练动作分类网络，直接使用外部动作先验/标注；
- ✗ **No RL / End-to-End Policy Network**: 保持可解释性动作打分规则，为后续 v9.1+ 学习模型提供 Ground Truth 与对比基准。

---

## 3. 四大基线对比体系 (Four Baselines)

1. **Random View**: 在全部空间与几何可行视点中均匀随机采样；
2. **Nearest View**: 选择距离人体中心最近的可行视点；
3. **Geometry Best (v8 Baseline)**: 仅基于几何可见性、全身覆盖率、距离惩罚与未观测损失选择最优视点；
4. **Action-conditioned (v9 Ours)**: 基于动作关键部位覆盖、推荐视角朝向偏好与距离适配的综合最优视点。

---

## 4. 架构组织与目录结构 (Architecture Map)

```text
ea_avs_mvp_v9/
├── configs/
│   ├── v9_demo.yaml            # 默认演示配置
│   ├── action_weights.yaml     # 动作观测先验与身体分区权重
│   └── v9_experiment.yaml      # 批量实验参数
├── core/
│   ├── paths.py                # 物理数据根路径解析
│   ├── types.py                # ActionEmbedding, ViewFeature, ActionViewpointScore
│   └── config.py               # 统一配置加载器
├── action/
│   ├── action_types.py         # 动作分类与别名标准化
│   └── action_encoder.py       # One-hot 编码与动作先验提取器
├── features/
│   └── view_feature_extractor.py # 16 关节解剖身体分区覆盖率提取
├── scoring/
│   └── action_scorer.py        # Q(v|a) 综合质量评价器
├── selection/
│   └── viewpoint_selector.py   # 四基线视点选择调度器
├── dataset/
│   └── v9_dataset_loader.py    # v8 数据集读取与 action.json 导出
├── evaluation/
│   ├── action_metrics.py       # 动作特定指标评测
│   └── baseline_comparison.py  # 4 策略横向定量对比生成
├── visualization/
│   └── action_view_plotter.py  # 报表格式化与打印工具
├── scripts/
│   ├── run_v9_demo.py          # 端到端 Demo 运行入口
│   └── compare_baselines.py    # 5 类动作批量评测脚本
└── tests/
    └── unit/                   # 纯 Python 单元测试集
```

---

## 5. 快速运行与验证 (Quick Start)

### 1. 运行纯 Python 单元测试集:
```bash
python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"
```

### 2. 运行 v9.0 演示入口:
```bash
# 运行默认摔倒动作感知视点选择
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v9_demo --action fall

# 切换为静坐动作
conda run -n habitat python -m ea_avs_mvp_v9.scripts.run_v9_demo --action sitting
```

### 3. 查看输出成果 (`data/ActiveView/visualizations/v9_demo/`):
- `action_view_scores.json`: 候选视点动作感知综合评分；
- `best_view.json`: 选定视点外参与动作观测特征；
- `comparison_report.json`: 四大基线量化指标对比表；
- `best_view_rgb.png`: 选定视点渲染图像；
- `action.json`: 动作先验配置。
