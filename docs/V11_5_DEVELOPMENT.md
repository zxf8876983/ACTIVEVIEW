# ACTIVEVIEW v11.5 开发与实验状态

更新时间：2026-08-27
当前源码目录：`ea_avs_mvp_v11/`

本文记录 v11.5 当前实际代码、数据链路、离线数据结构、评估协议和相对于历史版本的清理/修正。它与 [`V11_5_SELECTED16_DATASET_PROTOCOL.md`](V11_5_SELECTED16_DATASET_PROTOCOL.md) 一起构成 v11.5 的开发文档；若历史 README、脚本或结果与本文冲突，以本文和实际 canonical entry points 为准。

## 1. 研究边界与当前结论

ACTIVEVIEW 的研究对象是主动视角选择，而不是重新设计动作识别网络。ST-GCN 只是冻结的下游动作识别 backbone：机器人根据当前可用信息选择下一个可达观察点，候选点的真实观测只能在评估阶段产生，不能在决策阶段泄漏给策略。

当前 v11.5 的工作区只实现并报告几何/随机基线：`NoMove`、`Fixed`、`Random`、`Nearest`，以及仅用于上界对比的候选池 `Oracle`。v11.3 的 `Utility Predictor`/`ActiveViewSelector` 已从当前 v11 工作区移除，没有对应的训练脚本、推理入口或 checkpoint，因此当前结果不能称为 learned Ours 结果。

## 2. Canonical 运行时边界

- 源码：`/home/zxf/WorkSpace/code/code/ActiveView/`。
- 运行时数据：`/home/zxf/WorkSpace/code/data/ActiveView/`，由 `ea_avs_mvp_v11/core/paths.py` 解析，也可用 `ACTIVEVIEW_DATA_ROOT` 覆盖。
- Habitat 场景和语义文件只从 `/home/zxf/WorkSpace/code/code/robot/DATA/` 读取，也可用 `ACTIVEVIEW_HABITAT_DATA_ROOT` 覆盖；禁止扫描其它磁盘。
- 人体资产使用项目内副本 `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`，不再从 Habitat-Lab 源目录搜索。
- 大型数据集、RGB、视频、NPY/NPZ、权重和缓存不进入 Git。

## 3. 16 类 ST-GCN 预训练数据集

### 3.1 标签与划分

当前 label mapping 固定为官方筛选的 14 类加两个安全辅助类 `lie`、`stumble`；`fall` 不再是独立类别：

```text
0  t pose                 8  move up/down incline
1  cartwheel              9  jog
2  knock                 10  stand up
3  play instrument       11  jump
4  crawl                 12  walk
5  a pose                13  throw
6  kick                  14  lie
7  sit                   15  stumble
```

数据来自本地 BABEL `train.json` 和 `val.json`，不是 BABEL 空的 `test.json`：

1. 只保留单一目标标签的源区间。
2. 要求源区间严格 `num_frames > 30`。
3. 相同 AMASS 源区间若对应冲突标签，全部剔除。
4. 官方 14 类分别限制为 Train 每类最多 400、Val 每类最多 100；`lie` 和 `stumble` 不设上限。
5. 固定随机种子 42，并写出 `train.json`、`val.json`、`label_mapping.json`、`summary.json` 和 `excluded.json`。

当前实际 manifest 规模为 Train 3,240、Val 980，张量分别为 `(3240, 3, 30, 17, 1)` 和 `(980, 3, 30, 17, 1)`。实际类别计数写入 `train.json`/`val.json` 与 summary，不允许根据旧 10 类或旧 14 类数据推断。

### 3.2 Manifest 代码

唯一入口：

```bash
python ea_avs_mvp_v11/scripts/prepare_selected16_manifests.py
```

实现分工：

- `ea_avs_mvp_v11/dataset/babel_selected16_manifest.py`：类别、过滤、采样上限和 label mapping。
- `babel_official150_true_skeleton.py`：读取已审计的官方 150 类映射和记录。
- `babel_segment_utils.py`：去重、短区间过滤、冲突区间剔除及辅助标签读取。
- `babel_source_utils.py`：BABEL 记录到本地 AMASS 源文件的解析。

### 3.3 RGB/Habitat 感知生成

Train 和 Val 都经过同一条真实感知链路，ST-GCN 不读取 AMASS/SMPL GT joints：

```text
BABEL interval
→ AMASS SMPL motion
→ male_0 articulated object
→ 纯色 Habitat + 人体贴地/地面网格
→ RGB-only 256×256
→ Ultralytics YOLO26n-Pose（多人时按检测置信度和时序框选择目标人）
→ VideoPose3D（COCO-17/H36M-17 lifting）
→ Y/Z 相机轴转换 + Habitat camera-to-world/gravity frame
→ root centering + torso-scale normalization + yaw-only alignment
→ ST-GCN tensor
```

关键实现位于：

- `ea_avs_mvp_v11/dataset/babel_clean_dataset_generator.py`
- `ea_avs_mvp_v11/perception/ultralytics_pose3d_estimator.py`
- `ea_avs_mvp_v11/perception/skeleton_normalizer.py`
- `ea_avs_mvp_v11/dataset/humanoid_grounding.py`

人体根姿态保留 AMASS 的 roll/pitch/yaw；归一化阶段只消除水平 yaw，不把躺倒/跌倒人体旋转回站立。VideoPose3D 的 Human3.6M `+Y down, +Z depth` 先翻转 Y/Z，再使用逐帧 Habitat 相机旋转；平移由 root centering 消除。贴地偏移由 URDF visual geometry 的包围盒和场景支持面计算，并缓存到同一动作的所有视点。

生成器只注册 COLOR sensor，不保存深度图。每条记录保存独立的估计骨架 NPZ（含 skeleton 和 confidence），再合并成 `*_data.npy`、`*_labels.npy` 和 metadata/summary。默认支持断点续跑；只有 pose backend 一致时才复用已有骨架，避免 YOLO11/YOLO26 混用。

单进程入口：

```bash
python ea_avs_mvp_v11/scripts/generate_selected16_habitat_dataset.py --split train --device cuda:0
python ea_avs_mvp_v11/scripts/generate_selected16_habitat_dataset.py --split val --device cuda:0
```

多进程入口 `generate_selected16_habitat_parallel.py` 将 manifest 分片，每个 worker 独立 Habitat simulator 和 CUDA pose estimator，成功后合并元数据与张量：

```bash
python ea_avs_mvp_v11/scripts/generate_selected16_habitat_parallel.py --split train --workers 2 --device cuda:0
```

### 3.4 ST-GCN 训练与冻结

入口：`ea_avs_mvp_v11/scripts/train_selected16_habitat_stgcn.py`。

- 输入：H36M-17、3 通道、30 帧、单人 `(N,3,30,17,1)`。
- 模型：spatial ST-GCN，启用 edge-importance weighting。
- 类别不平衡：`WeightedRandomSampler` 使用 `count^(-0.5)` tempered oversampling；交叉熵同时使用归一化的 `sqrt(N/count)` class weights。
- 优化：Adam、`ReduceLROnPlateau`，调度器监控确定性的全量 Train loss（`mode=min`）。
- 收敛：最多 200 epochs，仅按 Train loss 早停（`patience=20`、`min_delta=1e-4`）；Val 在训练完成后只做一次 post-hoc 性能上限诊断，不参与训练、学习率调整或 checkpoint 选择。
- checkpoint 保存停止 epoch 的最后模型，而不是最低 Train loss epoch 的模型；同时记录最低 Train loss epoch、Train/Val 数据 SHA-256、类别计数、采样参数、预处理协议和 `frozen=true`。文件名保留 `_best.pth` 仅用于兼容现有评估入口。

当前冻结权重：

```text
/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/
stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/
```

本次 canonical 训练实际在第 165 epoch 因 Train loss 连续 20 个 epoch 未改善而停止；第 145 epoch 的最低 Train loss 为 `0.006555`，最终保存第 165 epoch 权重。Train 指标为 Accuracy `99.81%`、Macro-F1 `99.90%`；独立纯色 Habitat Val（980 条）post-hoc 上限为 Accuracy `76.33%`、Macro-F1 `73.58%`、Balanced Accuracy `73.57%`。Val 不参与 ST-GCN 训练或 checkpoint 选择，也不参与 Habitat 主动视角策略训练。

## 4. 主动视角离线数据集

### 4.1 两阶段设计

离线阶段不提前保存 32 张 RGB，也不保存深度；它保存候选视点的几何、导航元数据和从该视点得到的估计骨架。评估阶段使用冻结 ST-GCN 的每视点预测，按策略选择一个可达的下一视点。

场景目录规范：

```text
/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/
├── hm3d-minival/<原始 Habitat 场景文件夹>/
└── hm3d-train/<原始 Habitat 场景文件夹>/
```

每个语义区域从家具语义位置选择一个人体放置点：`bedroom`、`living_room`、`kitchen`、`dining_area`。每个放置点生成 4 个半径（1.5/2.0/2.5/3.0 m）× 8 个方位，共 32 个候选视点。候选元数据生成入口为 `generate_semantic_region_candidate_metadata.py`；4 区域场景编排入口为 `generate_hm3d_train_four_region_offline.py`。

### 4.2 Candidate metadata 与 v2 记录

候选 manifest 使用 `semantic-region-v2`，每个记录的 NPZ 至少包含：

- `scene_id`、`region`、`placement_id`、`placement_position`；
- `navmesh_path` 和 candidate manifest 引用；
- 32 个原始 `viewpoint_positions`；
- navmesh snap 后的 `viewpoint_snapped_positions`；
- 渲染器实际使用的 `viewpoint_agent_positions`；
- `viewpoint_rotations_wxyz`；
- `viewpoint_is_navigable`；
- `viewpoint_is_reachable_from_placement`；
- `viewpoint_navigation_cost_m`；
- 32 个 `(3,30,17)` 估计骨架和每视点 confidence。

`is_reachable_from_placement` 只是人体放置点到候选点的静态参考，不能代替机器人连续移动时的路径判断。这样保存信息后，评估时可以从机器人当前真实位置重新判断任意候选点是否可达。

### 4.3 离线生成实现

`generate_semantic_region_offline_views.py` 的每个 worker：

1. 加载一个 Habitat 场景和 navmesh，创建 4 个 COLOR 相机。
2. 对动作序列执行一次 AMASS 转换和逐帧贴地偏移预计算。
3. 每次批量设置 4 个相机视点，渲染同一动作的 30 帧 RGB。
4. 对每个视点执行 YOLO26n → VideoPose3D → 坐标转换 → 归一化。
5. 保存骨架、confidence、32 视点几何和导航字段，不保存 RGB/Depth。

已批准的 HM3D-train 场景集合为 21 个语义场景，编排器按场景串行、每场景 4 个 Habitat worker，以控制显存并利用多相机。当前文件系统审计显示 12 个场景有完整 `manifest.json`（每场景 980×4×32=125,440 个视点样本），第 13 个场景目录尚不完整；没有检测到正在运行的生成进程，后续可从场景目录断点续跑。

## 5. 策略评估流程

### 5.1 在线决策边界

对每个场景/区域，先从人体放置点可达的候选视点中选择一个初始机器人位置。选择下一视点前，重新用 Habitat navmesh 计算当前机器人位置到全部候选点的 `ShortestPath`，剔除不可达点和当前点，然后在动态可达池中应用策略。策略选择不能读取候选点未来 RGB、真实遮挡或候选点标签。

### 5.2 当前策略

- `NoMove`：停留在初始观察点；32-grid 协议下为精确基线，连续随机起点协议下只能使用最近缓存视点代理。
- `Fixed`：动态可达池中最小 viewpoint ID。
- `Random`：按固定种子从动态可达池随机选择。
- `Nearest`：选择当前点到候选点 geodesic cost 最小者。
- `Oracle`：评估后查看冻结 ST-GCN 预测和真实标签，在同一动态候选池中选预测正确且熵最低的视点；若没有正确视点，退化为最低熵视点。它是候选池理论上限，不是可执行策略。

入口：

- 单场景/静态参考：`evaluate_semantic_region_offline.py`。
- 动态可达性：`evaluate_hm3d_train_dynamic_reachability.py`。
- 500 次连续随机初始点：`evaluate_hm3d_train_random_initializations.py`。
- 32 网格初始点：`evaluate_hm3d_train_grid_initializations.py`。

预测矩阵和导航矩阵可缓存；场景/navmesh 只加载一次，避免把固定加载时间重复计入每个视点。

### 5.3 已记录结果

当前结果文件位于运行时 `results/`：

- `semantic_region_offline_baselines.json`：单场景离线基线。
- `hm3d_train_dynamic_reachability_10scenes.json`：前 10 个完整场景动态评估，NoMove 39,200 条，移动策略 38,220 条；Fixed 39.00%、Random 34.84%、Nearest 40.33%、Oracle 88.91%。
- `hm3d_train_random_initializations_500_train_only.json`：使用 Train-only 收敛 checkpoint；35 个可行场景/区域放置点 × 500 起点 × 980 动作，共 17,150,000 条；NoMove 39.49%、Fixed 40.10%、Random 34.95%、Nearest 39.61%、Oracle 92.44%，墙钟时间 187.1 s。
- `hm3d_train_grid_initializations_32_train_only.json`：使用 Train-only 收敛 checkpoint；259 个有效网格起点，共 253,820 条；NoMove 39.76%、Fixed 40.54%、Random 34.63%、Nearest 41.68%、Oracle 93.95%，墙钟时间 130.8 s。
- 不带 `_train_only` 后缀的随机/网格结果是旧 checkpoint 的历史对照，不作为当前 canonical 结果。

这些数字是基线/上界诊断，不能解释为 learned active-view policy 的性能。

## 6. v11.5 代码清理与科学修正记录

相对于 v10/v11.3，当前工作区做了以下有意修改：

1. 移除旧 v10 RGB-D 数据生成器、旧 v11 household/official150 入口、旧 MediaPipe/RTMPose/Keypoint-R-CNN 入口以及 v11.3 Utility Predictor/closed-loop 入口；v10 目录只作为只读历史参考。
2. 删除深度传感器和深度文件保存逻辑，统一 RGB-only。
3. 将姿态后端统一为 Ultralytics YOLO26n-Pose，并保留项目内 VideoPose3D lifter。
4. 将完整 3D canonical alignment 改为 yaw-only，保留人体相对重力的 roll/pitch，避免把 `lie`/`stumble` 旋转成站立。
5. 修正 Human3.6M/ Habitat 相机 Y/Z 轴转换，并在归一化前执行 camera-to-gravity 变换。
6. 用 URDF visual geometry 和支持面 raycast 修正人体飘浮/穿模；贴地偏移按动作帧缓存并跨视点复用。
7. 将主动视角离线记录升级为 `semantic-region-offline-v2`，持久化原始、snap、实际 agent 位置、旋转、导航性和静态可达性字段。
8. 将策略评估改为动态可达性：每次从机器人当前点重算候选路径，禁止用放置点静态可达性替代连续路径判断。
9. 将场景路径和人体路径集中到 `core/paths.py`，禁止全盘扫描；加入断点续跑、预测缓存和场景级缓存。
10. 增加归一化、相机重力变换、manifest 冲突过滤和 ST-GCN 输入形状测试，并把数据 SHA-256、训练参数和预处理协议写入 checkpoint。

## 7. 当前限制与下一步

- 当前正式 learned active-view strategy 尚未实现；结果只能用于基线和 Oracle 上限比较。
- `lie`/`stumble` 等安全动作样本仍少，纯 skeleton 对物体交互动作的语义区分有限。
- 纯色 Habitat 预训练域与真实 HM3D 家具遮挡域存在 domain gap。
- HM3D-train 四区域离线数据仍需完成剩余场景；恢复生成前必须检查场景 manifest 是否为 v2，不能混用早期 v1 产物。
- 任何新策略都必须只使用决策时可见状态，并在同一动态可达候选池、同一 ST-GCN checkpoint 和同一随机种子协议下与基线比较。
