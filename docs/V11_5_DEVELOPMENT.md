# ACTIVEVIEW v11.5 开发与实验状态

更新时间：2026-08-29
当前唯一源码目录：`activeview/`

> 2026-08-29 repository consolidation：v1–v10 源码树已从当前工作树移除，
> `activeview/` 是唯一正式源码包。历史报告中的旧路径只表示当时的事实，
> 不再作为可执行入口。

本文记录 v11.5 当前实际代码、数据链路、离线数据结构、评估协议和相对于历史版本的清理/修正。它与 [`V11_5_SELECTED16_DATASET_PROTOCOL.md`](V11_5_SELECTED16_DATASET_PROTOCOL.md) 一起构成 v11.5 的开发文档；若历史 README、脚本或结果与本文冲突，以本文和实际 canonical entry points 为准。

## 1. 研究边界与当前结论

ACTIVEVIEW 的研究对象是主动视角选择，而不是重新设计动作识别网络。ST-GCN 只是冻结的下游动作识别 backbone：机器人根据当前可用信息选择下一个可达观察点，候选点的真实观测只能在评估阶段产生，不能在决策阶段泄漏给策略。

当前 v11.5 已实现并冻结几何/随机基线：`NoMove`、`Fixed`、`Random`、`Nearest`，以及仅用于上界对比的候选池 `Oracle`；同时已完成 Stage C current-conditioned Utility Predictor（Pairwise MLP 与 Set Ranker）的离线训练、评估和独立验收。Stage C 结果属于离线 learned policy 诊断，尚不能替代 Habitat 在线 learned-policy 评估。

## 2. Canonical 运行时边界

- 源码：`/home/zxf/WorkSpace/code/code/ActiveView/`。
- 运行时数据：`/home/zxf/WorkSpace/code/data/ActiveView/`，由 `activeview/core/paths.py` 解析，也可用 `ACTIVEVIEW_DATA_ROOT` 覆盖。
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
5. 固定随机种子 42，并写出 `train.json`、`val.json`、`label_mapping.json`、`summary.json` 和 `excluded.json`。ST-GCN 预训练仍只使用 BABEL train/val；Stage A policy split 的比例另行定义如下。

当前实际 manifest 规模为 Train 3,240、Val 980，张量分别为 `(3240, 3, 30, 17, 1)` 和 `(980, 3, 30, 17, 1)`。实际类别计数写入 `train.json`/`val.json` 与 summary，不允许根据旧 10 类或旧 14 类数据推断。

### 3.2 Manifest 代码

唯一入口：

```bash
python -m activeview.scripts.data.prepare_selected16_manifests
```

实现分工：

- `activeview/data/motion/babel_selected16_manifest.py`：类别、过滤、采样上限和 label mapping。
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

- `activeview/data/motion/babel_clean_dataset_generator.py`
- `activeview/perception/pose/ultralytics.py`
- `activeview/perception/normalization.py`
- `activeview/data/motion/humanoid_grounding.py`

人体根姿态保留 AMASS 的 roll/pitch/yaw；归一化阶段只消除水平 yaw，不把躺倒/跌倒人体旋转回站立。VideoPose3D 的 Human3.6M `+Y down, +Z depth` 先翻转 Y/Z，再使用逐帧 Habitat 相机旋转；平移由 root centering 消除。贴地偏移由 URDF visual geometry 的包围盒和场景支持面计算，并缓存到同一动作的所有视点。

生成器只注册 COLOR sensor，不保存深度图。每条记录保存独立的估计骨架 NPZ（含 skeleton 和 confidence），再合并成 `*_data.npy`、`*_labels.npy` 和 metadata/summary。默认支持断点续跑；只有 pose backend 一致时才复用已有骨架，避免 YOLO11/YOLO26 混用。

单进程入口：

```bash
python -m activeview.scripts.data.generate_selected16_habitat_dataset --split train --device cuda:0
python -m activeview.scripts.data.generate_selected16_habitat_dataset --split val --device cuda:0
```

多进程入口 `generate_selected16_habitat_parallel.py` 将 manifest 分片，每个 worker 独立 Habitat simulator 和 CUDA pose estimator，成功后合并元数据与张量：

```bash
python -m activeview.scripts.data.generate_selected16_habitat_parallel --split train --workers 2 --device cuda:0
```

### 3.4 ST-GCN 训练与冻结

入口：`activeview.scripts.train.train_selected16_habitat_stgcn.py`。

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

离线阶段不提前保存 32 张 RGB，也不保存深度；它保存候选视点的几何、导航元数据和从该视点得到的估计骨架。评估阶段使用冻结 ST-GCN 的每视点预测，按策略选择一个可达的下一视点。Stage A policy records 使用 canonical `train/val/test = 6:2:2`；实际比例唯一读取 split 目录的 `summary.json -> split_ratios`。当前 980 条 policy records 已重新划分为 589/197/194；旧的 Stage A Episode JSONL 不会自动随 split 文件更新，必须重新运行 `build_policy_episodes.py`。加载 split 时会同时核对 summary 中的 split counts、unique record count、各类别计数和 canonical ratios 与实际 JSON 文件；任何不一致都会在 Episode 构建入口直接失败。

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

已批准的 HM3D-train 场景集合为 21 个语义场景，编排器按场景串行、每场景 4 个 Habitat worker，以控制显存并利用多相机。当前文件系统审计显示 21 个场景均有完整 `manifest.json`（每场景 980×4×32=125,440 个视点样本）；没有检测到正在运行的生成进程。

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

### 5.4 Stage A 科研级验收

仅运行 unit tests 不足以证明 Stage A 的最终 Episode 合法。当前验收分为三层：

1. `audit_episode_files()` 重新读取最终的 `train/val/test_episodes.jsonl`，检查跨 split record、一致的 `candidate_count`、current 不进入候选池、候选 ID/几何/代价、record-level skeleton path 一致性，以及递归 schema 中是否混入未来 RGB、Depth、pose、prediction 或 entropy 等信息。
2. 开启 `validate_cached_skeletons=True` 时，对所有被最终 Episode 引用的 NPZ 做缓存级检查：skeleton 必须为 `(32,3,30,17)`，导航数组必须存在且形状为 `(32,3)/(32,4)`，viewpoint ID 唯一有效，导航字段必须有限，current/candidate viewpoint 必须对应有效骨架帧。单个 viewpoint 的骨架失败不会使整个 archive 作废；失败 viewpoint 只进入信息性计数 `nonfinite_cached_skeleton_viewpoints`，并由 Episode 级有效集合过滤。
3. 审计还要求每个 `record_id × scene_id × region` 只有一个 Episode、`episode_id` 全局唯一，并将 Episode 中的 current/candidate 几何按 viewpoint ID 与 NPZ 中的 position、snapped position、agent position 和 rotation 做 `1e-5` 容差交叉核对。
4. Coverage audit 同时比较目标场景集合与实际成功进入生成的场景集合，检查 `target_scene_count`、`used_scene_count`、`failed_scene_count` 和 `missing_scene_ids`；`all_target_scenes_used` 必须为 `true`，不允许场景级失败被静默移出 expected set。
5. `validate_policy_episodes.py --verify-habitat` 在真实 HM3D NavMesh 上重新调用 Habitat `ShortestPath`，验证最终 Episode 的 current→candidate geodesic 与序列化代价一致。场景只加载一次，并缓存 `(scene, region, current_id, candidate_id)` 路径结果；支持 `--max-habitat-episodes` 做快速 smoke；论文级验收应运行全量模式。

验收入口：

```bash
python -m activeview.scripts.eval.validate_policy_episodes
conda run --no-capture-output -n habitat python -m activeview.scripts.eval.validate_policy_episodes \
  --verify-habitat
```

`stage_a_summary.json` 中的 `episode_audit` 是最终 JSONL 的实际统计，`coverage_audit` 进一步验证所有 `record × complete_scene × selected_region` 恰好由一个 Episode 或一个 record-level exclusion 覆盖；`missing_tuple_count`、`duplicate_accounted_tuple_count` 和 `episode_and_exclusion_overlap` 必须为 0。`integrity_checks` 由这些统计推导而来，不再使用硬编码 `True`。其中 `split_overlap=false` 表示 split 之间没有重叠；`nonfinite_cached_skeleton_viewpoints` 是允许部分失败视点时的信息性计数，不作为失败条件；其余验收布尔值为 `true` 才表示通过。`no_forbidden_future_perception_fields` 只表示 Episode schema 没有混入未来感知字段，不代表 Episode 可原封不动提供给在线策略。真实 Habitat 集成测试和全量缓存审计应与 unit tests 分开记录，不能将被跳过的集成测试视为通过。

## 6. Stage C current-conditioned Utility Predictor 结果

Stage C 使用已验收的 Stage A/B 产物，不修改 Stage A/B 和冻结 ST-GCN。当前输入为 275-D current-only ST-GCN 状态，候选输入为 11-D 几何特征：使用 Stage A `relative_position` 转换到当前机器人 yaw 的 ego frame，并使用 snapped position 计算候选半径。候选未来 RGB、skeleton、label、entropy、confidence 和 utility 均未进入模型输入。

训练设置为 record-balanced sampling、SmoothL1 utility regression 和 stay-inclusive listwise ranking。checkpoint 只依据 Val Stage C recognition Macro-F1 选择；Test 只在最终诊断时使用。当前特征和预测产物位于运行时目录：

```text
/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/
```

### 6.1 Recognition 结果

| 方法 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| NoMove | 41.25% | 37.13% | 41.27% | 38.18% |
| Pairwise MLP | 63.27% | 57.75% | 61.45% | 55.33% |
| Set Ranker | **64.91%** | **59.80%** | **62.54%** | **56.37%** |
| CandidateOracle | 83.36% | 78.86% | 81.64% | 77.79% |
| SafeOracle | 85.85% | 81.85% | 84.49% | 81.11% |

`CandidateOracle` 和 `SafeOracle` 使用 Stage B 的后验信息，只是候选池理论上限，不是可执行策略。相对 NoMove，Set Ranker 在 Test 上提升 21.27 个百分点 Accuracy 和 18.19 个百分点 Macro-F1；Pairwise 分别提升 20.18 和 17.15 个百分点。

### 6.2 Regret、动作匹配和 headroom

| 方法 | Val regret mean / median / p90 | Test regret mean / median / p90 | Val headroom capture | Test headroom capture |
|---|---:|---:|---:|---:|
| Pairwise MLP | 1.674 / 0.0088 / 6.326 | 1.802 / 0.0108 / 6.582 | 72.88% | 70.83% |
| Set Ranker | **1.450 / 0.0051 / 5.608** | **1.614 / 0.0075 / 6.143** | **77.80%** | **74.93%** |
| SafeOracle | 0 / 0 / 0 | 0 / 0 / 0 | 100%（定义上限） | 100%（定义上限） |

Set Ranker 相比 Pairwise 在 Test 上 candidate hit rate 为 33.93%（Pairwise 32.41%），safe-action match rate 为 29.85%（Pairwise 28.63%），mean regret 降低 0.188，aggregate headroom capture 提高 4.10 个百分点。Headroom capture 定义为正 SafeOracle utility episode 上的 `sum(max(0, selected utility)) / sum(SafeOracle utility)`；Val/Test 分别包含 11,879/11,644 个正 headroom episode。由极小 SafeOracle utility 导致的 `raw_mean` 不作为主统计量。

### 6.3 科研解释与限制

Stage C 已显示 current-conditioned view selection 明显优于 NoMove，Set Ranker 略优于 Pairwise；但 Set Ranker Test 仍比 SafeOracle 低 21.95 个百分点 Accuracy、24.74 个百分点 Macro-F1，说明 utility ranking 尚未接近候选池上限。Regret 的 median 接近 0 而 p90 较高，表明总体分布存在明显长尾。

这些结果来自一次固定 split/训练运行（Val 13,987、Test 13,774 episodes），没有多 seed 置信区间或显著性检验。Train/Val/Test 按 motion record 划分且共享 HM3D 场景，因此不能解释为跨场景泛化结果。Stage D 真实 Habitat 在线 learned-policy 评估尚未开始。

## 7. v11.5 代码清理与科学修正记录

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

## 8. 当前限制与下一步

- 当前 Stage C learned active-view strategy 仅完成离线训练与诊断；Stage D 的真实 Habitat 在线 learned-policy 闭环尚未实现。
- `lie`/`stumble` 等安全动作样本仍少，纯 skeleton 对物体交互动作的语义区分有限。
- 纯色 Habitat 预训练域与真实 HM3D 家具遮挡域存在 domain gap。
- HM3D-train 四区域离线数据的 21 个目标场景均已完成；恢复任何任务前仍必须检查场景 manifest 是否为 v2，不能混用早期 v1 产物。
- 任何新策略都必须只使用决策时可见状态，并在同一动态可达候选池、同一 ST-GCN checkpoint 和同一随机种子协议下与基线比较。
