# EA-AVS MVP7.0: Humanoid-driven Active Perception Simulation Environment

## 1. 概述 (Overview)

`ea_avs_mvp_v7` 是面向老人监护与拟人化动态人体的室内主动感知仿真环境与实验平台。
本版本旨在构建可复现、严谨解耦的实验基础设施：

```text
BABEL Action Annotation
        ↓
AMASS Motion Asset
        ↓
Motion Normalization (NormalizedMotion)
        ↓
Humanoid Agent (KinematicHumanoid)
        ↓
Habitat Indoor Scene (apartment_1.glb)
        ↓
Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
        ↓
Episode Dataset
```

---

## 2. 目录架构与职责划分 (Directory Architecture)

```text
ea_avs_mvp_v7/
├── configs/
│   ├── habitat_config.yaml          # Habitat 场景、NavMesh 与物理配置
│   ├── humanoid_config.yaml         # Humanoid URDF 资产配置 (neutral_0)
│   ├── motion_config.yaml           # AMASS 清单与帧率配置
│   └── sensor_config.yaml           # 机器人 RGB-D 相机参数配置
│
├── core/
│   ├── paths.py                     # 数据根目录解析 (../../data/ActiveView 或 ACTIVEVIEW_DATA_ROOT)
│   ├── episode.py                   # Episode / EpisodeFrame 数据结构
│   └── config.py                    # 统一配置加载器
│
├── environment/
│   ├── scene_manager.py             # 场景与 NavMesh 资产定位
│   └── habitat_env.py               # Habitat-Sim 生命周期管理 (严禁 SLAM/导航规划)
│
├── robot/
│   ├── robot_agent.py               # 移动机器人底盘位姿控制
│   └── rgbd_sensor.py               # 相机内外参矩阵与 RGB-D 传感器读取
│
├── human/
│   ├── action_state.py              # 动作类别与 BABEL 语义标注
│   ├── human_state.py               # 人体空间状态与 16 关节 3D 世界坐标真值 (Ground-Truth)
│   └── humanoid_agent.py            # KinematicHumanoid 实例化与姿态驱动
│
├── motion/
│   ├── amass_loader.py              # AMASS npz 读取与 Schema A/B 标准化 (NormalizedMotion)
│   ├── motion_converter.py          # Habitat SMPL-X 关节/刚体四元数格式转换
│   └── motion_player.py             # 动作序列单步步进与帧控制
│
├── observation/
│   ├── metadata.py                  # 单帧与序列元数据结构
│   └── recorder.py                  # 观测落盘 (PNG/NPY/JSON)
│
├── dataset/
│   └── episode_generator.py         # 多动作多视角 Episode 数据集生成器
│
├── evaluation/
│   └── basic_metrics.py             # 数据集完整性统计指标
│
├── scripts/
│   ├── run_smoke_test.py            # 最小闭环端到端验证 Smoke Test
│   ├── convert_motions.py           # AMASS 动作批量转换工具
│   └── generate_dataset.py          # 多动作多视角数据集批量生成
│
└── tests/                           # 纯 Python 单元测试套件
```

---

## 3. 运行指南 (Execution Guide)

### 3.1 运行纯 Python 单元测试 (无需 GPU / Habitat)
```bash
python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"
```

### 3.2 批量转换 AMASS 动作资产
```bash
/home/zxf/anaconda3/envs/habitat/bin/python -m ea_avs_mvp_v7.scripts.convert_motions --action fall_related
```

### 3.3 运行端到端最小闭环 Smoke Test
```bash
/home/zxf/anaconda3/envs/habitat/bin/python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 5
```

### 3.4 批量生成多动作多视角 Episode 数据集
```bash
/home/zxf/anaconda3/envs/habitat/bin/python -m ea_avs_mvp_v7.scripts.generate_dataset --actions standing sitting fall_related --num-views 4
```

---

## 4. 科学边界与非目标声明 (Scientific Boundary & Non-Goals)

本版本严格遵循 `EA_AVS_MVP70_Code_Generation_Document.md`：
- ❌ **不实现** NBV 视点选择算法；
- ❌ **不实现** Action-aware utility 效用函数；
- ❌ **不实现** 强化学习 (RL) 与多步主动感知；
- ❌ **不实现** 动作识别网络 (HAR) 与真实机器人部署；
- ❌ **历史版本保护**：`ea_avs_mvp_v6/` 及更早版本保持只读；
- ❌ **零大文件入库**：所有数据集与运行时产物必须置于外部数据目录 `../../data/ActiveView`。
