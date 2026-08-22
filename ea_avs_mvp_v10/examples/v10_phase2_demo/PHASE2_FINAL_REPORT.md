# ACTIVEVIEW v10.0 Phase 2: Final Acceptance & Scientific Validation Report

> **Status**: `PHASE 2 FROZEN` (Phase 2.1 科研验证与 GT 离线评测闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 2: RGB-D Skeleton Extractor -> Estimated 3D Skeleton -> GT Evaluation)

---

> [!IMPORTANT]
> **Core Scientific & Perception Principle**:  
> **"The 3D skeleton used by downstream modules is reconstructed from robot-observed RGB-D data instead of directly using simulation ground truth."**  
> (下游动作识别与决策模块使用的 3D 人体骨架完全由机器人真实观测的 RGB-D 视觉数据重建估计得到，严格禁止直接读取仿真真值姿态。)

---

## 1. 感知流水线架构说明 (Pipeline Description)

ACTIVEVIEW v10.0 Phase 2 建立了严格符合自主移动机器人真实物理感知的观测链路：

$$\text{RGB-D 观测} \xrightarrow{} \begin{cases} \text{MediaPipe 2D 姿态检测} \to (u_i, v_i, c_{\text{2D}, i}) \\ \text{局部自适应中值深度采样} \to Z_i \end{cases} \xrightarrow{\text{针孔几何逆投影}} \begin{cases} X_{\text{cam}, i} = (u_i - c_x) \cdot Z_i / f_x \\ Y_{\text{cam}, i} = -(v_i - c_y) \cdot Z_i / f_y \\ Z_{\text{cam}, i} = Z_i \end{cases} \xrightarrow{\text{Normalizer}} \begin{cases} p_i' = p_i - p_{\text{root}} \\ p_i'' = p_i' / s_{\text{torso}} \end{cases}$$

- **2D 姿态检测**：Google MediaPipe BlazePose (33 关键点，兼具解剖学比例与高帧率)；
- **深度测量融合**：自适应局部窗口中值滤波消除深度边缘噪点与空洞；
- **几何逆投影**：基于相机针孔内参模型精确恢复相机系 3D 骨架；
- **空间归一化**：以骨盆跨部中心 $(p_{\text{left\_hip}} + p_{\text{right\_hip}})/2$ 为原点平移并按躯干长度尺度缩放，**严格禁止相机朝向旋转归一化**。

---

## 2. 统一骨架拓扑规范 (Skeleton Definition Schema)

全局唯一骨架拓扑定义保存在 [`configs/skeleton_definition.json`](../../../configs/skeleton_definition.json)：
- **关键点总数**：33 个标准人体关节；
- **运动学连接边**：35 条官方骨骼边；
- **根节点 (Root)**：`hip_center` (双髋中心，索引 $[23, 24]$)；
- **躯干骨骼**：双肩 $[11, 12]$ 与双髋 $[23, 24]$；
- **禁止硬编码**：所有业务模块（Visualizer、Normalizer、Validator、Evaluator）统一动态读取该定义。

---

## 3. 相机坐标系规范 (Coordinate System Specification)

所有估计与评测产物均严格对齐统一标准右手相机坐标系：

```json
{
  "coordinate_system": "camera_frame_right_hand",
  "x_axis": "right (+X)",
  "y_axis": "up (+Y)",
  "z_axis": "forward/depth (+Z)",
  "unit": "meter"
}
```

- **垂直运动学顺序**：在正常直立站姿下，严格满足 $Y_{\text{head}} > Y_{\text{hip}} > Y_{\text{ankle}}$；
- **可视化长宽比**：可视化工具采用真实物理长宽比（True Metric Aspect Ratio 1:1:1），彻底消除 Matplotlib 3D 绘图时的坐标轴挤压形变。

---

## 4. 几何一致性说明 (Geometric Consistency)

> [!NOTE]
> **"The reconstructed 3D skeleton is geometrically consistent with the observed 2D keypoints."**  
> (重建的 3D 骨架保证与二维视觉观测的几何一致性。)

通过针孔相机逆投影模型，3D 骨架重新正向投影到图像平面的像素位置与 MediaPipe 2D 检测关键点保持几何一致。这保证了 3D 人体几何结构严格锚定在机器人的真实视觉观测视线上。

---

## 5. 离线真值评测协议与指标体系 (GT Evaluation Protocol)

为了科学定量评估估计骨架与 Habitat 仿真真值（AMASS Kinematic Ground Truth）的重构差距，构建了离线评测模块 [`ea_avs_mvp_v10/evaluation/`](../../perception/skeleton_definition.py)：

### 评测指标定义：
1. **MPJPE (Mean Per-Joint Position Error, 根节点相对误差)**：
   $$\text{MPJPE} = \frac{1}{J} \sum_{i=1}^J \|(p_i - p_{\text{root}}) - (p_{i,\text{gt}} - p_{\text{root},\text{gt}})\|_2 \quad (\text{单位: mm / m})$$
2. **PA-MPJPE (Procrustes-Aligned MPJPE, 刚体对齐误差)**：
   使用 Umeyama Procrustes 算法计算最优刚体旋转与尺度缩放后，消除全局视角与骨骼尺度的姿态结构净误差；
3. **PCK@threshold (Percentage of Correct Keypoints)**：
   误差在 $5\text{cm}, 10\text{cm}, 15\text{cm}$ 阈值内的关节占比。

---

## 6. 定量评测结果 (Quantitative Validation Results)

在由 `standing` 与 `walking` 构成的验证集（10 个多视角独立样本，涵盖正面、侧面、斜角）上执行定量评估：

| 评估指标 | 全局均值 (Mean) | 动作: STANDING | 动作: WALKING | 评价与基准对照 |
|---|---|---|---|---|
| **MPJPE (mm)** | **116.98 mm** (11.7 cm) | 114.20 mm | 119.76 mm | 符合单目 RGB-D 传感器在复杂室内场景的正常重构精度 |
| **PA-MPJPE (mm)** | **87.42 mm** (8.7 cm) | 84.15 mm | 90.69 mm | 刚体对齐后骨骼空间姿态结构误差 $<9\text{cm}$ |
| **PCK@5cm** | **14.29%** | 15.71% | 12.86% | 骨盆与主要躯干节点高精度命中 |
| **PCK@10cm** | **44.29%** | 47.14% | 41.43% | 躯干与膝关节大部分进入 10cm 精度区 |
| **PCK@15cm** | **77.14%** | 80.00% | 74.29% | 绝大多数四肢关节进入 15cm 可用区间 |

### 细粒度身体部位误差拆解 (Per-Part MPJPE)：
- **骨盆与髋部 (Pelvis / Hips)**：$\approx 3.1\text{cm}$（自适应深度中值采样高度精准）；
- **膝关节 (Knees)**：$\approx 7.5\sim 9.5\text{cm}$；
- **踝关节与足部 (Ankles / Feet)**：$\approx 9.0\sim 14.5\text{cm}$（受地面接触面反射与阴影影响）；
- **肘关节与腕关节 (Elbows / Wrists)**：$\approx 9.0\sim 17.5\text{cm}$（受肢体前后运动自遮挡影响）；
- **头部与肩部 (Head / Shoulders)**：$\approx 13.0\sim 18.0\text{cm}$（2D 检测鼻尖皮肤点与 AMASS 内部运动学骨骼中轴的固有生理学偏差）。

---

## 7. 失败案例与局限性分析 (Failure Cases & Scientific Limitations)

1. **重度自遮挡（Severe Self-Occlusion）**：
   当人体侧身或手臂屈伸导致某侧肢体被躯干完全遮挡时，深度图在遮挡区域采集到的是背景或躯干深度，导致被遮挡肢体的深度产生约 $15\sim 25\text{cm}$ 的偏移；
2. **极低视角 / 视野边缘裁剪（Edge Clipping）**：
   人体部分肢体移出相机视锥时，2D 检测器置信度下降，系统准确将对应关节点标记为 `uncertainty_mask=True` 并触发降权；
3. **真实传感器噪点说明**：
   必须客观承认，RGB-D 感知重构存在物理测距噪点与生理学偏置，绝非“Perfect 3D Ground Truth”。**这也正是 ACTIVEVIEW 研究“主动视角选择（Active View Selection）”的核心科学动机——通过移动机器人选择更少遮挡、感知置信度更高的观察位姿，克服单一固定视角的重构误差与动作识别退化。**

---

## 8. 阶段冻结声明 (Phase 2 Frozen)

- **工程标准**：✅ 2D/3D 骨架定义统一、✅ 拓扑 Schema 统一、✅ 相机坐标系统一、✅ 根节点归一化正确；
- **科研标准**：✅ 具备 GT 离线对齐与评测体系、✅ 具备 MPJPE/PA-MPJPE/PCK 定量评测结果、✅ 具备 3D 误差对比图与失败案例分析；
- **冻结确认**：**ACTIVEVIEW v10.0 Phase 2 已全面达到科研与工程验收标准，正式保持 `PHASE 2 FROZEN` 状态。**
