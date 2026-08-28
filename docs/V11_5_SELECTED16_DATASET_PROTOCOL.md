# ACTIVEVIEW v11.5 selected16 数据集与评估协议

本文是 v11.5 的机器可执行协议摘要。完整的代码变更、数据字段和评估解释见 [`V11_5_DEVELOPMENT.md`](V11_5_DEVELOPMENT.md)。旧 10 类、旧 14 类 `fall` 版本、source-sequence split、GT-skeleton shortcut 和 Utility Predictor 历史协议均不属于当前默认流程。

## 1. 当前 16 类

固定 label mapping：

```text
0:t pose, 1:cartwheel, 2:knock, 3:play instrument,
4:crawl, 5:a pose, 6:kick, 7:sit,
8:move up/down incline, 9:jog, 10:stand up, 11:jump,
12:walk, 13:throw, 14:lie, 15:stumble
```

`lie`、`stumble` 是为老人安全感知保留的辅助标签；`fall` 当前不作为独立类。类别顺序来自 `activeview/dataset/babel_selected16_manifest.py`，不得从旧数据集推断。

## 2. Manifest 规则

- 数据源：BABEL `train.json` → Train，BABEL `val.json` → Val。
- BABEL `test.json` 为空，不构造所谓官方 Test 分数。
- 只保留单标签源区间，且严格 `num_frames > 30`。
- 相同 AMASS 源区间对应冲突标签时全部剔除。
- 官方 14 类：Train 每类最多 400、Val 每类最多 100；`lie`/`stumble` 不设上限。
- 随机种子默认为 42。

入口：

```bash
python -m activeview.scripts.prepare_selected16_manifests
```

当前实际结果：Train 3,240、Val 980；类别计数和排除原因分别保存在运行时数据根目录的 `train.json`、`val.json`、`summary.json`、`excluded.json`。

Stage A policy records 使用 canonical `train/val/test = 6:2:2`。Episode
builder 唯一读取 split 目录 `summary.json` 中的 `split_ratios`，不再重复
硬编码比例；ST-GCN 预训练的 BABEL train/val 数据不因此增加独立 Test split。

## 3. ST-GCN 预训练数据

```text
AMASS/SMPL
→ male_0 纯色 Habitat
→ RGB 256×256（仅 COLOR sensor）
→ Ultralytics YOLO26n-Pose
→ VideoPose3D
→ Y/Z 相机轴转换 + camera-to-gravity
→ root center + torso scale + yaw-only
→ H36M-17 ST-GCN tensor
```

生成入口：

```bash
python -m activeview.scripts.generate_selected16_habitat_dataset --split train --device cuda:0
python -m activeview.scripts.generate_selected16_habitat_dataset --split val --device cuda:0
```

默认输出形状为 `(N,3,30,17,1)`，序列统一均匀采样 30 帧。ST-GCN 只接收 RGB 估计骨架，绝不读取 AMASS/SMPL GT joints。贴地偏移使用 URDF visual geometry 和场景支持面计算，并跨候选视点缓存；归一化只移除 root、体型尺度和水平 yaw，保留重力相关 roll/pitch。

训练入口：

```bash
python -m activeview.scripts.train_selected16_habitat_stgcn --device cuda:0
```

训练使用 `WeightedRandomSampler(count^-0.5)`、归一化 `sqrt(N/count)` class weights、Adam 和监控全量 Train loss 的 `ReduceLROnPlateau(mode=min)`。early stopping 只使用 Train loss（最多 200 epochs、`patience=20`、`min_delta=1e-4`），保存停止 epoch 的最后模型。Val 不进入训练循环；训练结束后仅对 Val 做一次 post-hoc 性能上限诊断，之后才进入冻结 checkpoint 的主动视角评估。

## 4. 主动视角离线数据

目录：

```text
data/ActiveView/datasets/offline/
├── hm3d-minival/<原始场景文件夹>/
└── hm3d-train/<原始场景文件夹>/
```

每个场景包含 `bedroom`、`living_room`、`kitchen`、`dining_area` 四类语义家具放置点。每个放置点生成半径 1.5/2.0/2.5/3.0 m、每圈 8 个方位的 32 个候选视点。离线阶段保存骨架、confidence 和完整导航元数据，不保存 RGB/Depth。

候选元数据入口：

```bash
python -m activeview.scripts.generate_semantic_region_candidate_metadata
```

视点骨架生成入口：

```bash
python -m activeview.scripts.generate_semantic_region_offline_views --workers 4 --device cuda:0
```

每个 NPZ 必须包含：场景/区域/放置点、navmesh 路径、32 个原始位置、snap 位置、实际 agent 位置、相机旋转、`is_navigable`、`is_reachable_from_placement`、路径代价，以及 32 个 `(3,30,17)` 估计骨架。离线 schema 为 `semantic-region-offline-v2`；候选 manifest 为 `semantic-region-v2`。

`is_reachable_from_placement` 只表示放置点到候选点的静态参考。连续策略评估必须重新加载场景 navmesh，从机器人当前点向所有待选点计算 `ShortestPath`，不能把静态标志当作下一步可达性。

## 5. 策略评估

当前评估策略：

- `NoMove`：停留在初始点；32-grid 协议下为精确基线。
- `Fixed`：动态可达池中最小 viewpoint ID。
- `Random`：固定种子随机选择动态可达视点。
- `Nearest`：选择当前点到候选点 geodesic cost 最小者。
- `Oracle`：事后查看真实标签，在同一动态候选池中选择预测正确且熵最低的视点；仅为候选池理论上限，不能用于真实决策。

入口：

```bash
python -m activeview.scripts.evaluate_semantic_region_offline
python -m activeview.scripts.evaluate_hm3d_train_dynamic_reachability --device cuda:0
python -m activeview.scripts.evaluate_hm3d_train_random_initializations --device cuda:0
python -m activeview.scripts.evaluate_hm3d_train_grid_initializations --device cuda:0
```

策略决策不得读取候选视点未来 RGB、真实遮挡、标签或后验 ST-GCN 结果；这些信息只用于评估。场景/navmesh、预测矩阵和路径矩阵可缓存，固定场景加载时间不重复计入每个视点。

## 6. 当前状态

- 冻结 ST-GCN：`checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`。
- HM3D-minival：当前离线策略数据使用 `00800-TEEsavR23oF`。
- HM3D-train：21 个目标场景中已有 12 个完整 `manifest.json`，另有一个不完整场景目录；恢复生成时必须断点续跑并检查 v2 schema。
- v11.3 Utility Predictor 未在当前语义区域协议上训练或评估；当前报告只包含基线和 Oracle。
