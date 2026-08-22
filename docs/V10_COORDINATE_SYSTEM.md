# ACTIVEVIEW v10.0: Coordinate System & Kinematic Specification

> **Target Version**: `ea_avs_mvp_v10`  
> **Status**: `FROZEN & ENFORCED`  
> **Purpose**: 严格定义与冻结 ACTIVEVIEW v10.0 阶段从 2D 像素到 3D 骨架的坐标体系，杜绝坐标系混淆导致的人体空间位置错误与模型输入退化。

---

## 1. 坐标系全景概述 (Coordinate System Overview)

ACTIVEVIEW v10.0 感知流水线严格解耦并定义以下四个层次的坐标系统：

```text
[1. Image Coordinate] (u, v) [Pixels]
        │
        ▼ (Pinhole Back-Projection with Depth Z)
[2. Camera Coordinate] (X_cam, Y_cam, Z_cam) [Meters] (Local Robot Perception)
        │
        ├─────────────────────────────────────────┐
        ▼ (Extrinsic Transform R_c2w, T_c2w)       ▼ (Root & Scale Normalization)
[3. World Coordinate] (X_w, Y_w, Z_w)    [4. Normalized Coordinate] (X_norm, Y_norm, Z_norm)
  (Habitat Global Analysis & Vis)          (ST-GCN Action Recognition Input)
```

---

## 2. 各坐标系严格数学定义 (Mathematical Definitions)

### 2.1 图像坐标系 (Image Coordinate System)
- **定义**：$(u, v)$，单位为像素 (Pixels)。
- **原点**：图像左上角 $(0, 0)$。
- **轴向**：
  - $+u$：向右（水平宽度方向，范围 $0 \sim 639$）；
  - $+v$：向下（垂直高度方向，范围 $0 \sim 479$）。

---

### 2.2 相机局部坐标系 (Camera Coordinate System) —— Phase 2 核心 3D 输出
- **定义**：$(X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}})$，单位为米 (Meters)。
- **原点**：相机光心 (Optical Center)。
- **轴向（计算机视觉标准右手系）**：
  - $+X_{\text{cam}}$：指向相机**右方** (Right)；
  - $+Y_{\text{cam}}$：指向相机**上方** (Up, 注意在反投影中 $Y = -(v - c_y) \cdot Z / f_y$)；
  - $+Z_{\text{cam}}$：指向相机**正前方光轴方向** (Forward / Depth)。
- **逆投影恢复公式 (Depth-assisted Back-Projection)**：
  $$X_{\text{cam}} = \frac{(u - c_x) \cdot Z}{f_x}$$
  $$Y_{\text{cam}} = -\frac{(v - c_y) \cdot Z}{f_y}$$
  $$Z_{\text{cam}} = Z = \text{depth}(u, v)$$
  *其中 $(f_x, f_y, c_x, c_y)$ 为相机几何内参。*
- **用途**：作为机器人局部感知与主动视角规划的核心 3D 几何表征。

---

### 2.3 仿真世界坐标系 (World Coordinate System)
- **定义**：$(X_w, Y_w, Z_w)$，单位为米 (Meters)。
- **原点**：Habitat 仿真场景全局原点。
- **轴向**：Habitat 场景全局右手坐标系（$+Y_w$ 为重力垂直向上）。
- **坐标变换**：
  $$P_w = R_{\text{cam}\to\text{world}} \cdot P_{\text{cam}} + T_{\text{cam}\to\text{world}}$$
- **用途约束**：**仅用于全局环境可视化与 Habitat 场景分析，严禁作为下游动作识别模型的前向输入**。

---

### 2.4 归一化骨架坐标系 (Normalized Coordinate System) —— ST-GCN 动作识别专用输入
- **定义**：$(X_{\text{norm}}, Y_{\text{norm}}, Z_{\text{norm}})$，无量纲相对坐标。
- **归一化算法流程**：
  1. **根节点平移去中心化 (Root Normalization)**：
     以骨盆/跨部中心 $p_{\text{root}} = (\text{left\_hip} + \text{right\_hip}) / 2$ 为原点：
     $$p_i' = p_i - p_{\text{root}}$$
  2. **躯干尺度归一化 (Scale Normalization)**：
     以躯干长度 $s = \|\text{neck} - p_{\text{root}}\|$（或标准人体高度尺度）作为尺度因子：
     $$p_i'' = \frac{p_i'}{s + \epsilon}$$
  3. **严格禁止相机朝向旋转归一化 (No Camera-Facing Rotation Normalization)**：
     主动视角（Active View）研究的核心科学命题在于不同观察视角下人体投影形态的变化。若施加朝向旋转对齐，将抹除视角的差异性。
- **用途**：作为 Phase 3 ST-GCN 动作识别模型的纯净标准化输入。

---

## 3. 坐标健康度检查 (Sanity Check Protocol)

为防止因反投影噪声、深度边缘虚空或误检导致的坐标异常（如人体飞天、嵌入墙体、骨架上下颠倒），系统强制执行以下四阶几何健康度校验：

1. **深度测距范围检查**：$0.2\text{m} \le Z_{\text{cam}} \le 8.0\text{m}$；
2. **人体尺度范围检查**：$0.3\text{m} \le \text{Height}_{\text{skeleton}} \le 2.5\text{m}$；
3. **关键点相对运动学朝向检查**：头部中心 $Y_{\text{head}}$ 在正常站立/行走/弯腰姿态下应高于脚踝中点 $Y_{\text{ankle}}$（或在合理倾角容差内）；
4. **统计分类输出**：每一批处理均输出 `VALID`, `WARNING`, `INVALID` 状态统计。
