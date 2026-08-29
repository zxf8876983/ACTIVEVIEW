# ACTIVEVIEW v10.0 Phase 3.1: Scientific Consistency Validation Report

> **报告版本**：ACTIVEVIEW v10.0 Phase 3.1  
> **日期**：2026-08-23  
> **状态**：`PHASE 3 FROZEN & READY FOR PHASE 4`  
> **核心使命**：对 Phase 3 进行科学一致性审计与修正，建立完全统一的 RGB 驱动动作识别基准，为 Phase 4 Active View Selection Policy 提供固定的动作评价器与不确定度量化支撑。

---

## 1. 核心修改内容清单 (Modifications Summary)

| 修改项 | 修改前状态 | Phase 3.1 修正后状态 | 科学意义与影响 |
|---|---|---|---|
| **动作识别输入源** | 曾有 SMPL GT 混用风险 | **严格强制仅允许 `Pose3DEstimator` 输出进入 ST-GCN** | 彻底消除信息泄漏，保证科研严谨性 |
| **Oracle 基准命名** | "Clean Oracle" / "GT upper bound" | **"Clean Perception Oracle" (Ideal Perception Condition)** | 明确是理想感知条件下的上界，绝非 GT 骨骼 |
| **感知估计器类型** | 包含历史 Depth back-projection 逻辑 | **主链路完全为 RGB-only 3D Pose Estimator (MediaPipe3D)** | 符合现实服务机器人单目 RGB 感知设定 |
| **Schema 运行时断言** | 潜在的隐式维度映射 | **增加 `assert joints == configured_joints` 显式校验** | 杜绝关节数量或连接关系的隐式篡改 |
| **数据集样本规模** | 小规模冒烟测试 ($N=180$) | **规模化扩展至 $N=3,672$ 序列 (每类动作 612 条 $\ge 500$)** | 满足科研论文级实验与统计显著性要求 |
| **样本级 Metadata** | 仅保存基础 label | **保存完整元数据 (`viewpoint`, `camera_height`, `yaw`, `confidence` 等)** | 实现任意样本物理位姿与感知状态的高精度复现 |
| **评价指标体系** | 仅有 Accuracy 与 Entropy | **补充宏精确率 Precision, 宏召回率 Recall, 宏 F1-score, 归一化不确定度 $U_{\text{norm}}$** | 建立多维全面的分类与不确定度评测体系 |

---

## 2. 数据流程与物理边界确认 (Data Pipeline & Physical Boundary)

### 2.1 统一感知与动作识别数据链路
```
+-------------------------------------------------------------+
|                      AMASS Motion (.pkl)                    |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             RGB Image Rendering (T, H, W, 3)                |
|  - Clean Perception: Unobstructed, neutral background       |
|  - Habitat Perception: Indoor scene, multi-view, obstacles  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|        Unified 3D Pose Estimator (MediaPipe BlazePose 3D)   |
|         (Outputs V=33 joints in Camera Right-Hand Frame)    |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Skeleton Normalizer (Root-centered, Torso-scaled) |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               ST-GCN Action Recognition Network             |
|                (9-Block Spatial-Temporal GCN)               |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|     Action Probabilities p in R^6 + Shannon Entropy H(p)    |
+-------------------------------------------------------------+
```

### 2.2 隔离审计结论
1. **训练与测试链路 100% 同构**：无论是 `train/clean_perception` 还是 `test/habitat_perception`，输入 ST-GCN 的数据均由相同的 `SequencePoseExtractor` 提取自渲染 RGB 图像；
2. **SMPL GT 骨架物理隔离**：代码库中没有任何将 SMPL 真实关节直接送入 ST-GCN 的分支；
3. **RGB-Only 主链路**：主动作感知完全依赖 RGB 图像估计 3D 关节坐标，未引入未来观测或候选点真实渲染。

---

## 3. Clean / Habitat 一致性检查 (Consistency Audit)

| 检查维度 | Clean Perception | Habitat Perception | 一致性判定 |
|---|---|---|---|
| **关节数量 Joint Count** | 33 关节 (`configs/skeleton_definition.json`) | 33 关节 (`configs/skeleton_definition.json`) | **PASS (完全一致)** |
| **骨架拓扑 Topology** | 35 条肢体连接边 + $K=3$ 空间划分邻接矩阵 | 35 条肢体连接边 + $K=3$ 空间划分邻接矩阵 | **PASS (完全一致)** |
| **坐标系定义 Coordinate** | 相机右手坐标系 (+X 右, +Y 上, +Z 前) | 相机右手坐标系 (+X 右, +Y 上, +Z 前) | **PASS (完全一致)** |
| **归一化方案 Normalization** | 双髋中心置零 + 躯干对角线距离缩放至 1.0 | 双髋中心置零 + 躯干对角线距离缩放至 1.0 | **PASS (完全一致)** |
| **时序长度 Time Steps** | 30 帧 (采样对齐) | 30 帧 (采样对齐) | **PASS (完全一致)** |
| **网络输入张量形状** | $(3, 30, 33, 1)$ | $(3, 30, 33, 1)$ | **PASS (完全一致)** |

---

## 4. 规模化数据集统计 (Expanded Dataset Statistics)

数据物理存储根目录：`/home/zxf/WorkSpace/code/data/ActiveView/datasets/action/`

### 4.1 数据集分布
- **总时序样本数**：**3,672 sequences**（全面满足 $\ge 3000$ 样本科研要求）；
- **6 大动作类别分布**：
  - `standing`: 612 样本
  - `walking`: 612 样本
  - `sitting`: 612 样本
  - `bending`: 612 样本
  - `reaching`: 612 样本
  - `fall_related`: 612 样本

### 4.2 数据集拆分与用途
1. **`train/clean_perception/` (2,400 样本)**：
   - 训练 ST-GCN 动作识别网络；
   - 采用固定随机种子 (`seed=42`) 与多角度摄动；
2. **`test/clean_perception/` (600 样本)**：
   - 评测 **Clean Perception Oracle** 理想感知上界；
3. **`test/habitat_perception/` (672 样本)**：
   - 评测 **Habitat Perception Baseline** 真实室内多视点基准；
   - 覆盖 16 个标准化视点 ($r \in [1.5, 2.0]\,\text{m}$, $\theta \in [0^\circ, 45^\circ, \dots, 315^\circ]$)。

---

## 5. Benchmark 评测结果 (Benchmark Results)

### 5.1 双基准对比 (Oracle vs Habitat Baseline)

| 实验基准 Benchmark | 样本数 $N$ | 准确率 Accuracy | 宏精确率 Precision | 宏召回率 Recall | 宏 F1 分数 | 交叉熵损失 Loss | 平均预测信息熵 $H(p)$ | 归一化不确定度 $U_{\text{norm}}(p)$ |
|---|---|---|---|---|---|---|---|---|
| **实验一: Clean Perception Oracle** | 600 | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.0094** | **0.0551** | **0.0307** |
| **实验二: Habitat Perception Baseline** | 672 | **70.83%** | **0.6591** | **0.7083** | **0.6251** | **1.3125** | **0.3482** | **0.1943** |
| **环境与视点扰动退化 ($\Delta$)** | - | **$-29.17\%$** | $-0.3409$ | $-0.2917$ | **$-0.3749$** | $+1.3031$ | **$+0.2931$** | **$+0.1636$** |

### 5.2 视点不确定度分析 (Active Viewpoint Uncertainty Analysis)
- **优质视点 ($\theta \in [45^\circ, 225^\circ, 270^\circ, 315^\circ]$, $r=1.5\,\text{m}$)**：
  - 准确率：**83.33%**；
  - 宏 F1 分数：**0.7778**；
  - 能够同时捕捉冠状面与矢状面动作运动轨迹。
- **退化视点 ($r=2.0\,\text{m}$ 远距离或遮挡视点)**：
  - 准确率降至 **66.67%**；
  - 宏 F1 分数降至 **0.5556**；
  - 平均信息熵 $H(p)$ 显著升高。

---

## 6. 测试与系统状态 (Testing & Phase Status)

1. **自动化单元测试**：
   - `ea_avs_mvp_v10/tests/unit/`: **42 / 42 PASS (100%)**；
   - 全库跨版本回归测试 (v7, v8, v9, v10): **118 / 118 PASS (100%)**。
2. **Phase 3 状态判定**：
   - **Phase 3 已正式冻结 (`FROZEN`)**；
   - 动作识别模型权重已稳定固化至 `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/v10_st_gcn/best_st_gcn_model.pth`；
   - **系统已完全准备就绪，静待进入 Phase 4 (Active View Selection Policy)**。
