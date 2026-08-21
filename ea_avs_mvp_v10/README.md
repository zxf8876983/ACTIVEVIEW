# ACTIVEVIEW v10.0: RGB-D Driven Task-aware Active View Selection

> **Phase 1 Implementation & Research Baseline**  
> *"Where should the robot observe to understand human actions better?"*  
> *(机器人应如何主动选择观察视角以更准确地识别人体动作？)*

---

## 1. 版本定位与研究边界 (Version Scope & Research Boundary)

ACTIVEVIEW 从 v8/v9 的几何与感知质量驱动（"Where can the robot see better?"）正式迈向 **下游动作识别任务驱动的主动感知（"Where should the robot observe to recognize better?"）**。

### Phase 1 严格范围 (Phase 1 Scope):
- ✅ **Habitat RGB-D 多视角数据集采集引擎**：编排室内场景、KinematicHumanoid、MotionPlayer 与移动底盘相机；
- ✅ **支持 6 大核心动作类别**：`standing`, `walking`, `sitting`, `bending`, `reaching`, `falling`；
- ✅ **多视角空间候选生成**：多距离半径与极角方位；
- ✅ **多模态数据持久化**：RGB (PNG)、Depth (NPY / Vis PNG)、CameraPose (JSON) 与 GT 骨骼 (Oracle用)；
- ❌ **禁止提前实现**：ST-GCN 分类器、主动视角决策网络、强化学习或端到端策略网络。

---

## 2. 模块架构 (Package Architecture)

```text
ea_avs_mvp_v10/
├── core/                       # 路径解析、类型实体 (ActionClassV10, CameraPose, V10Sample) 与配置
├── configs/                    # YAML 配置文件 (default_v10_config.yaml)
├── sensors/                    # RGB-D 传感器捕获与相机位姿计算 (RGBDCapture)
├── motion/                     # 6 大动作类别资产管理与索引检索 (MotionManager)
├── viewpoint/                  # 多视角候选视点极坐标生成器 (CandidateViewpointGeneratorV10)
├── dataset/                    # 样本持久化构建与流水线生成器 (V10DatasetGenerator)
├── scripts/                    # 批量数据集生成与演示脚本
├── examples/v10_phase1_demo/   # 样本数据示例、说明与多模态可视化 (sample_visualization.png)
└── tests/unit/                 # 单元测试覆盖
```

---

## 3. 数据集结构规范 (Dataset Organization)

运行时数据集默认保存在：
`data/ActiveView/datasets/v10/`

```text
datasets/v10/
├── raw/
│   ├── rgb/              # 原始 RGB 观察图像 (PNG, uint8, 640x480)
│   ├── depth/            # 原始深度数据 (NPY float32 距离矩阵 + 可视化 PNG)
│   ├── camera_pose/      # 相机外参、世界位置、四元数与内参 (JSON)
│   └── scene_meta/       # 场景、人体位置、朝向等元数据 (JSON)
├── ground_truth/
│   ├── skeleton/         # 16 关键点 3D 坐标真值 (仅用于 Oracle/评估，严禁模型推理)
│   └── action/           # 真实动作类别与子标签 (JSON)
└── metadata/
    └── samples.json      # 全量样本索引清单 (JSON)
```

---

## 4. 运行与验证指令 (Execution & Verification)

### 1. 运行纯 Python 单元测试:
```bash
python3 -m unittest discover -s ea_avs_mvp_v10/tests -p "test_*.py"
```

### 2. 运行 Phase 1 演示与可视化流水线:
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase1_demo
```

### 3. 运行批量数据集生成 CLI:
```bash
conda run -n habitat python -m ea_avs_mvp_v10.scripts.generate_v10_dataset --frame_step 10 --max_frames 5
```
