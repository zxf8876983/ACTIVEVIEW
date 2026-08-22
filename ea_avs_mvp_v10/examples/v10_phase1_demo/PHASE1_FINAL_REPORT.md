# ACTIVEVIEW v10.0 Phase 1: Final Acceptance & Freeze Report

> **Status**: `PHASE 1 FROZEN` (Phase 1 研发与验证闭环完成并正式冻结)  
> **Date**: 2026-08-22  
> **Target Version**: `ea_avs_mvp_v10` (Phase 1: Habitat RGB-D Dataset Generation)

---

## 1. 阶段目标与完成情况清单 (Completed Scope & Modules)

| 核心模块 | 对应代码 | 验收状态 | 说明 |
|---|---|---|---|
| **路径与数据规范** | `ea_avs_mvp_v10/core/paths.py` | **PASS** | 物理数据隔离至 `/home/zxf/WorkSpace/code/data/ActiveView/datasets/v10/` |
| **核心数据实体** | `ea_avs_mvp_v10/core/types.py` | **PASS** | `ActionClassV10`, `CameraPose`, `CandidateViewpointV10`, `V10Sample` |
| **RGB-D 传感器捕获** | `ea_avs_mvp_v10/sensors/rgbd_capture.py` | **PASS** | RGB uint8 PNG (640x480), Depth float32 NPY (meters), 相机内参/外参矩阵 |
| **6 大动作资产管理** | `ea_avs_mvp_v10/motion/motion_manager.py` | **PASS** | 索引 20 个转换动作资产，覆盖 6 大核心动作类别 |
| **多视角极坐标采样** | `ea_avs_mvp_v10/viewpoint/candidate_generator.py` | **PASS** | 多半径、多方位采样，严格遵守 Habitat `-Z` 前向与注视角对齐 |
| **样本持久化与索引** | `ea_avs_mvp_v10/dataset/sample_builder.py` | **PASS** | 规范写入 `raw/`, `ground_truth/`, `metadata/samples.json` |
| **仿真流水线生成引擎** | `ea_avs_mvp_v10/dataset/v10_dataset_generator.py` | **PASS** | 集成 NavMesh 可达性与 LineOfSight 光线追踪硬阻挡过滤 |
| **演示与可视化套件** | `ea_avs_mvp_v10/examples/v10_phase1_demo/` | **PASS** | 6 大动作类别、多视角多模态对比大图与说明文档 |

---

## 2. 数据格式与物理存储规范 (Data Storage Specifications)

运行时数据物理存储根目录：
`/home/zxf/WorkSpace/code/data/ActiveView/datasets/v10/`

```text
datasets/v10/
├── raw/
│   ├── rgb/              # 原始 RGB 观测图像 (PNG, uint8, 640x480x3)
│   ├── depth/            # 原始深度数据 (NPY, float32, 单位为米, 640x480)
│   ├── camera_pose/      # 相机外参、世界坐标、四元数与内参 (JSON)
│   └── scene_meta/       # 场景、人体位置、朝向等元数据 (JSON)
├── ground_truth/
│   ├── skeleton/         # 16 关键点 3D 坐标真值 (JSON, 仅供 Oracle 评估，严禁模型推理读取)
│   └── action/           # 真实动作类别与子标签 (JSON)
└── metadata/
    └── samples.json      # 全量样本索引清单 (JSON)
```

---

## 3. 严格数据隔离与科研信息边界 (Information Boundary Protection)

1. **GT 骨骼隔离**：`ground_truth/skeleton/` 中的 3D 关节真值仅供离线数据集校验和后续 Oracle 理论上限对比；
2. **严禁在线输入**：Phase 2 感知流水线与 Phase 3 识别模型**严禁直接读取任何 GT 骨骼或真实可见性数据**；
3. **真实观测流**：后续所有感知与决策模块的输入严格限定为 `raw/rgb/`、`raw/depth/` 以及相机内参；
4. **历史数据保护**：严格保持 `/home/zxf/WorkSpace/code/data/ActiveView/` 目录结构，不修改 `v7`、`v8`、`v9` 既有实验数据。

---

## 4. 运行与验证指令 (Verification Commands)

### 单元测试回归：
```bash
python3 -m unittest discover -s ea_avs_mvp_v10/tests -p "test_*.py"
```
*(验证结果：9/9 PASS)*

### 运行 Phase 1 演示与可视化流水线：
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase1_demo
```

### 批量数据生成 CLI：
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.generate_v10_dataset --frame_step 10 --max_frames 5
```

---

## 5. 当前阶段限制与冻结声明 (Limitations & Freeze Declaration)

- **当前限制**：
  1. Phase 1 仅提供 Habitat 真实物理环境下的多视角 RGB-D 仿真数据底座，不包含任何姿态估计、动作识别分类器或主动视角选择模型；
  2. 候选视点采样仅生成几何观测候选，不执行视角优选。
- **状态冻结声明**：
  **ACTIVEVIEW v10.0 Phase 1 核心功能与数据格式已全部通过验收并正式标记为 `FROZEN`。下一阶段开发将严格基于 Phase 1 生成的 RGB-D 数据展开 Phase 2 感知流水线构建。**
