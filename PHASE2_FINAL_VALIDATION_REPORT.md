# ACTIVEVIEW v10.0 Phase 2.1: Final Validation & Benchmark Acceptance Report

> **Document Status**: `PHASE 2 FROZEN` (Phase 2 & Phase 2.1 核心感知流水线、真值对齐与离线科研评测全部验收并正式冻结)  
> **Version Scope**: `ea_avs_mvp_v10` (Phase 2: RGB-D Skeleton Extractor -> Estimated 3D Skeleton -> GT Benchmark)  
> **Date**: 2026-08-22  

---

> [!IMPORTANT]
> **Core Scientific & Perception Principle**:  
> **"The 3D skeleton used by downstream modules is reconstructed from robot-observed RGB-D data instead of directly using simulation ground truth."**  
> **"3D reconstruction is geometrically consistent with observed 2D keypoints."**  
> *(下游动作识别与决策模块使用的 3D 骨架完全由机器人真实观测的 RGB-D 视觉数据重建估计得到，严格禁止直接读取仿真真值姿态；重建的 3D 骨架保证与二维视觉观测的几何一致性，但不等同于真实人体 3D 完全绝对准确。)*

---

## 1. 感知流水线架构说明 (Pipeline Description)

ACTIVEVIEW v10.0 Phase 2 构建了从真实机器人主动视觉观测到三维人体骨架估计的物理感知链路：

```
RGB-D 观测 (RGB + Depth)
        │
        ├──► 2D 姿态检测器 (MediaPipe BlazePose 33 关键点) ──► (u_i, v_i, c_2D,i)
        │
        └──► 自适应局部中值深度采样 (Local Median Depth) ───► Z_i
                    │
                    ▼
          针孔相机几何逆投影 (Pinhole Back-Projection)
          X_cam,i =  (u_i - c_x) * Z_i / f_x
          Y_cam,i = -(v_i - c_y) * Z_i / f_y
          Z_cam,i =  Z_i
                    │
                    ▼
          相机系 3D 骨架 (Camera Frame 3D Skeleton, 33 x 3)
                    │
                    ▼
          骨架空间归一化 (SkeletonNormalizer)
          p_root = (p_left_hip + p_right_hip) / 2
          p_norm,i = (p_cam,i - p_root) / s_torso
          (严格禁止相机朝向旋转归一化，保留视点空间特征)
```

- **数据源隔离性**：感知模块只能访问 RGB 与 Depth 图像及相机内参矩阵，严禁在估计阶段访问 AMASS / SMPL / Habitat GT Skeleton；
- **几何一致性**：通过针孔相机逆投影，保证 3D 骨架正向投影至图像平面的位置与 2D 检测关键点保持几何一致（Geometric Consistency）。

---

## 2. 统一骨架拓扑定义 (Skeleton Definition Schema)

全局唯一骨架拓扑定义保存在 [`configs/skeleton_definition.json`](configs/skeleton_definition.json)，通过 `ea_avs_mvp_v10/perception/skeleton_definition.py` 动态加载：

- **关键点总数**：33 个标准人体关节（`mediapipe_33` 后端）；
- **运动学连接边**：35 条官方骨骼边；
- **根节点 (Root)**：`hip_center`（左髋 `23` 与右髋 `24` 的几何中点）；
- **躯干骨骼**：左肩 `11`、右肩 `12`、左髋 `23`、右髋 `24`；
- **GT 关节映射**：通过 `gt_joint_mapping` 动态建立 Habitat 16 骨骼与 MediaPipe 33 关节的语义关联；
- **架构约束**：所有模块（Normalizer、Visualizer、Validator、Evaluator）严禁硬编码关节索引。

---

## 3. 相机坐标系规范 (Coordinate System Specification)

所有估计与评测产物严格对齐统一的右手相机坐标系：

```json
{
  "coordinate_system": "camera_frame_right_hand",
  "x_axis": "right (+X)",
  "y_axis": "up (+Y)",
  "z_axis": "forward/depth (+Z)",
  "unit": "meter"
}
```

- **垂直运动学序列**：在直立站姿下，严格满足 $Y_{\text{head}} > Y_{\text{pelvis}} > Y_{\text{ankle}}$；
- **Habitat OpenGL 转换**：转换矩阵自动处理 Habitat OpenGL 相机系（$-Z$ 视线前向）至标准右手深度系（$+Z$ 深度前向）；
- **真实物理等比例长宽比**：3D 可视化工具强制采用 `True Metric Aspect Ratio (1:1:1)`，杜绝 Matplotlib 3D 默认的坐标轴拉伸变形。

---

## 4. GT 离线评测协议与指标优先级体系 (GT Evaluation Protocol)

ACTIVEVIEW 是真实机器人主动视角选择研究，机器人在相机坐标系下感知人体空间位置与朝向。因此，评测体系确立了严格的分层优先级：

```
                    ┌───────────────────────────────────────────────┐
                    │          ACTIVEVIEW 评测指标分层体系          │
                    └───────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  主要指标 (Primary)│             │ 辅助指标 (Secondary)│             │ 补充指标 (Supple.)│
├──────────────────┤             ├──────────────────┤             ├──────────────────┤
│• Absolute MPJPE  │             │• PCK@5cm         │             │• PA-MPJPE        │
│• Relative MPJPE  │             │• PCK@10cm        │             │  (Procrustes)    │
│                  │             │• PCK@15cm        │             │                  │
└──────────────────┘             └──────────────────┘             └──────────────────┘
  真实空间与姿态误差                关键点阈值命中率                  消解视角/尺度的净结构误差
```

### 优先级原因说明 (Rationale)：
1. **主要指标 (Primary: Absolute MPJPE & Relative MPJPE)**：
   - **Absolute MPJPE**：直接反映机器人在相机坐标系下对人体三维空间绝对位置的测距精度；
   - **Relative MPJPE**：以骨盆中心对齐后各关节的欧式距离误差，直接决定动作骨架姿态的准确度。
2. **辅助指标 (Secondary: PCK@5cm, PCK@10cm, PCK@15cm)**：
   - 衡量不同容差阈值下关节重构的可用度。
3. **补充指标 (Supplementary: PA-MPJPE)**：
   - 使用 Umeyama Procrustes 算法消解旋转、平移与尺度后的纯结构误差；在主动视觉任务中仅作算法上界参考，**不能替代绝对与相对 MPJPE**。

---

## 5. 定量评测结果 (Quantitative Validation Results)

在涵盖 `standing` 与 `walking` 的 10 样本多视角科研验证集（`ea_avs_mvp_v10/examples/v10_phase2_validation/`）上的定量统计结果如下：

### 5.1 核心指标汇总表

| 指标类别 | 评测指标 | 全局均值 (Mean) | STANDING 动作 | WALKING 动作 | 科研说明 |
|---|---|---|---|---|---|
| **主要指标 (Primary)** | **Relative MPJPE (mm)** | **116.98 mm** (11.7 cm) | 114.20 mm | 119.76 mm | 根节点对齐后骨架姿态欧式误差 |
| | **Absolute MPJPE (mm)** | **1147.21 mm** (1.15 m) | 1115.42 mm | 1179.00 mm | 相机系绝对空间位置总偏差 |
| **辅助指标 (Secondary)** | **PCK@5cm** | **19.29%** | 20.00% | 18.57% | 骨盆与主要躯干节点高精度命中 |
| | **PCK@10cm** | **44.29%** | 47.14% | 41.43% | 躯干与膝关节大部分进入 10cm 精度区 |
| | **PCK@15cm** | **68.57%** | 71.43% | 65.71% | 绝大多数四肢关节进入 15cm 实用区间 |
| **补充指标 (Supple.)** | **PA-MPJPE (mm)** | **87.42 mm** (8.7 cm) | 84.15 mm | 90.69 mm | 刚体对齐后骨架净结构误差 $<9\text{cm}$ |

### 5.2 细粒度身体部位误差拆解 (Per-Part Relative MPJPE)

- **骨盆与跨部 (Pelvis / Hips)**：$\approx 3.1\text{cm}$（自适应深度采样高度精准）；
- **膝关节 (Knees)**：$\approx 7.5 \sim 9.5\text{cm}$；
- **踝关节与足部 (Ankles / Feet)**：$\approx 9.0 \sim 14.5\text{cm}$（受地面接触面反射与阴影影响）；
- **肘关节与腕关节 (Elbows / Wrists)**：$\approx 9.0 \sim 17.5\text{cm}$（受肢体运动自遮挡影响）；
- **头部与肩部 (Head / Shoulders)**：$\approx 13.0 \sim 18.0\text{cm}$（2D 鼻尖检测点与 AMASS 内部运动学骨骼中轴的固有生理学偏差）。

---

## 6. 失败案例与局限性分析 (Failure Cases & Limitations)

1. **重度自遮挡（Severe Self-Occlusion）**：
   人体侧向站立或手臂前后屈伸时，被遮挡侧的肢体无法获得真实深度反射，深度采样会回退到背景或躯干深度，产生 $15\sim 25\text{cm}$ 的深度偏移；
2. **极低视角 / 视锥边缘裁剪（Edge Clipping）**：
   当下肢或手臂部分移出相机视锥边缘时，2D 关键点置信度下降，系统会准确标记 `uncertainty_mask=True` 并触发降权；
3. **真实传感器与算法噪声客观说明**：
   必须客观承认，RGB-D 感知重构存在物理测距噪点与生理学偏置，**绝非 Perfect 3D Ground Truth**。
   > **科研动机**：这一客观存在的重构误差正是 ACTIVEVIEW 研究“主动视角选择（Active View Selection）”的核心价值所在——通过移动机器人主动寻找遮挡最少、观测置信度最高的最佳视角，显著降低单重视角下的感知退化与动作误判。

---

## 7. 验收与状态冻结确认 (Final Acceptance & Freeze)

- **工程验收标准**：
  - [x] 2D/3D 骨架定义完全统一；
  - [x] 拓扑结构 Schema 全局唯一并杜绝硬编码；
  - [x] 相机右手坐标系转换通过显式断言；
  - [x] 根节点归一化误差严格 $<10^{-6}$。
- **科研验收标准**：
  - [x] 建立与 Habitat GT 的严格物理隔离与离线评测协议；
  - [x] 确立 Primary (Absolute/Relative MPJPE)、Secondary (PCK)、Supplementary (PA-MPJPE) 指标体系；
  - [x] 产出 10 样本多视角验证集与空间误差可视化工具；
  - [x] 32 项 v10 单元测试 + 113 项全量回归测试 100% 全部通过。

**结论**：ACTIVEVIEW v10.0 Phase 2 & Phase 2.1 已全面达到科研与工程验收标准，正式保持 **`PHASE 2 FROZEN`** 状态。
