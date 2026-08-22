# ACTIVEVIEW v10.0 Phase 2: Final Acceptance & Freeze Report

> **Status**: `PHASE 2 FROZEN` (Phase 2 核心感知流水线研发与验证闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 2: Depth-assisted 3D Skeleton Reconstruction Pipeline)

---

> [!IMPORTANT]
> **Core Scientific & Perception Principle**:  
> **The 3D skeleton used by downstream modules is reconstructed from robot-observed RGB-D data instead of directly using simulation ground truth.**  
> (下游动作识别与决策模块使用的 3D 人体骨架完全由机器人真实观测的 RGB-D 视觉数据重建估计得到，严格禁止直接读取仿真真值姿态。)

---

## 1. 阶段目标与完成情况清单 (Completed Scope & Modules)

| 核心模块 | 对应代码 | 验收状态 | 说明 |
|---|---|---|---|
| **2D 姿态估计器封装** | `ea_avs_mvp_v10/perception/pose_estimator.py` | **PASS** | 开箱即用 Torchvision Keypoint R-CNN，原生输出 COCO-17 关键点 |
| **局部自适应深度逆投影** | `ea_avs_mvp_v10/perception/depth_projection.py` | **PASS** | 针孔反投影模型 + 局部邻域自适应中值滤波恢复 $(X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}})$ |
| **COCO-17 骨架生成与融合** | `ea_avs_mvp_v10/perception/skeleton_converter.py` | **PASS** | 原生保留 COCO-17 拓扑，$c = c_{\text{2D}} \cdot c_{\text{depth}}$ 复合感知置信度 |
| **3D 骨架空间归一化** | `ea_avs_mvp_v10/perception/skeleton_normalizer.py` | **PASS** | 根节点去中心化 + 躯干尺度归一化 (**严格禁止相机朝向旋转归一化**) |
| **坐标健康度校验器** | `ea_avs_mvp_v10/perception/coordinate_validator.py` | **PASS** | 物理深度范围、人体尺度、头部/脚踝相对上下位姿合理性检查 |
| **拓扑适配器接口** | `ea_avs_mvp_v10/perception/skeleton_adapter.py` | **PASS** | 可选 COCO-17 $\to$ NTU-25 适配器，Phase 2 默认保持 COCO-17 |
| **感知数据集持久化** | `ea_avs_mvp_v10/dataset/perception_dataset.py` | **PASS** | 独立保存 `pose2d/`, `pose3d_camera/`, `pose3d_world/`, `normalized_pose3d/`, `confidence/` |
| **自动化单样本检查 CLI** | `tools/v10/check_sample.py` | **PASS** | 支持按 ID、按动作或随机挑选生成 4 面板高分辨率诊断图 |
| **多模态可视化套件** | `ea_avs_mvp_v10/visualization/skeleton_visualizer.py` | **PASS** | 5 面板诊断图、归一化验证图、全量 6 动作全景图与遮挡测试图 |

---

## 2. 坐标系统与数据结构规范 (Coordinate Systems & Representations)

详细规范请参阅 [`docs/V10_COORDINATE_SYSTEM.md`](../../../docs/V10_COORDINATE_SYSTEM.md)。

系统严格解耦保存四个物理与模型维度的坐标体系：

1. **Image Coordinate $(u, v)$**：
   - 图像左上角原点，用于 2D 检测。
2. **Camera Coordinate (`joints_3d_camera`, 形状 $17 \times 3$)**：
   - 局部机器人相机坐标系（$+X$ 右, $+Y$ 上, $+Z$ 前/深度），单位为米 (Meters)；
   - 作为 Phase 2 主要 3D 输出，用于机器人局部主动感知与视点规划打分。
3. **World Coordinate (`joints_3d_world`, 形状 $17 \times 3$)**：
   - Habitat 仿真场景世界坐标系，结合相机位姿外参齐次变换恢复；
   - **仅用于 Habitat 全局可视化与场景几何分析，严禁作为下游动作分类器输入**。
4. **Normalized Coordinate (`joints_3d_normalized`, 形状 $17 \times 3$)**：
   - 根节点平移（以骨盆/跨部中心为原点 $(0, 0, 0)$）+ 躯干尺度归一化；
   - **严格禁止相机朝向旋转归一化**（完整保留主动视角在投影几何上的差异性）；
   - 作为后续 Phase 3 ST-GCN 动作分类器的直接输入。

---

## 3. 感知置信度与不确定性定义 (Perception Confidence & Uncertainty)

- **Perception Confidence ($c_i \in [0, 1]$)**：
  $$c_i = c_{\text{2D}, i} \cdot c_{\text{depth}, i}$$
  反映机器人当前的观测可靠性（而非真实物理可见性）。
- **Perception Uncertainty Mask (`uncertainty_mask`)**：
  当 $c_i < 0.35$ 时，标记该关节存在高感知不确定性（受自遮挡或环境遮挡影响）。

---

## 4. 自动化可视化检查工具使用说明 (Automated Inspection Tool)

为了应对未来大规模样本生成无法逐一人工核对的问题，提供自动化 CLI 检查工具：

```bash
# 1. 检查指定样本
python tools/v10/check_sample.py --sample_id v10_sample_000000

# 2. 随机挑选一个样本检查
python tools/v10/check_sample.py --random

# 3. 指定动作类别挑选
python tools/v10/check_sample.py --action sitting

# 4. 批量抽检 5 个样本
python tools/v10/check_sample.py --batch 5
```

---

## 5. 运行与验证指令 (Verification Commands)

### 单元测试回归：
```bash
python3 -m unittest discover -s ea_avs_mvp_v10/tests -p "test_*.py"
```
*(验证结果：23/23 PASS)*

### 运行 Phase 2 演示与全量数据处理：
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase2_demo
```

---

## 6. 当前阶段限制与冻结声明 (Limitations & Freeze Declaration)

- **当前限制**：
  1. Phase 2 专注于 RGB-D 观测到 3D 估计骨架的真实观测链构建，不包含动作分类器训练（Phase 3）或主动视角决策（Phase 4）；
  2. 姿态估计器采用开箱即用轻量模型，未针对室内特殊低照度场景进行微调。
- **状态冻结声明**：
  **ACTIVEVIEW v10.0 Phase 2 核心功能已全部通过验收并正式标记为 `FROZEN`。Phase 3 (ST-GCN Action Recognition) 将严格以 Phase 2 生成的 `normalized_pose3d` 序列与感知置信度作为输入。**
