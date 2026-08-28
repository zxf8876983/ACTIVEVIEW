# ACTIVEVIEW v11.5

ACTIVEVIEW 研究机器人在 HM3D 室内环境中主动选择观察视角，以降低人体姿态估计不确定性并提升老人动作感知。

## 当前唯一实验协议

- 动作标签：BABEL 官方筛选的 14 类 + `lie` + `stumble`，共 16 类；不使用 `fall`。
- 数据划分：BABEL `train.json` → Train，`val.json` → Val；单标签、源区间 `>30` 帧；官方类别上限为 Train 400、Val 100。
- 视觉链路：

  `AMASS/SMPL → male_0 Habitat 纯色场景 → RGB → Ultralytics YOLO26n-Pose → VideoPose3D → root/scale/yaw-only normalization → ST-GCN`

- 输入：RGB-only，`256×256`，均匀采样 30 帧；ST-GCN 冻结后用于主动视角评估。
- 主动视角离线评估：HM3D-train 目标场景中的卧室、客厅、厨房和 dining area 四个家具语义区域，每个动作 32 个候选视点；只保存骨架和元数据，不保存 RGB/Depth。旧场景 `00800-TEEsavR23oF` 仅作历史兼容，不属于当前评估集。
- 策略对比包含 Fixed、Random、Nearest；`Oracle` 为同一候选视点池上的事后 GT-correctness 理论上限，不参与实际决策。

完整协议见 [`docs/V11_5_SELECTED16_DATASET_PROTOCOL.md`](docs/V11_5_SELECTED16_DATASET_PROTOCOL.md)，代码变更、数据字段和评估流程见 [`docs/V11_5_DEVELOPMENT.md`](docs/V11_5_DEVELOPMENT.md)。

## 运行时数据

运行时根目录为 `/home/zxf/WorkSpace/code/data/ActiveView/`，也可通过 `ACTIVEVIEW_DATA_ROOT` 覆盖。原始 AMASS/BABEL 不放入 Git。

Habitat 场景和语义文件只从 `/home/zxf/WorkSpace/code/code/robot/DATA/` 读取，也可通过
`ACTIVEVIEW_HABITAT_DATA_ROOT` 覆盖；禁止为了寻找场景扫描其它磁盘。`male_0` 模型已复制到
`data/ActiveView/assets/habitat_humanoids/male_0/`，运行时不再依赖 Habitat-Lab 源目录。

- 训练/验证估计骨架：[stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed](../../data/ActiveView/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed)
- 冻结 ST-GCN：[stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled](../../data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled)
- 离线策略数据：[offline/hm3d-minival/00800-TEEsavR23oF](../../data/ActiveView/datasets/offline/hm3d-minival/00800-TEEsavR23oF)；后续 `hm3d-train` 和其它 Habitat 原场景使用同级目录。
- 基线结果：[semantic_region_offline_baselines.json](../../data/ActiveView/results/semantic_region_offline_baselines.json)

## 主要入口

```bash
python -m activeview.scripts.prepare_selected16_manifests
python -m activeview.scripts.generate_selected16_habitat_dataset --split train
python -m activeview.scripts.generate_selected16_habitat_dataset --split val
python -m activeview.scripts.train_selected16_habitat_stgcn
python -m activeview.scripts.generate_semantic_region_candidate_metadata
python -m activeview.scripts.generate_semantic_region_offline_views --workers 4
python -m activeview.scripts.evaluate_semantic_region_offline
```

`activeview/` 是唯一正式源码包；v1–v10 目录已从当前工作树移除，历史报告仅作只读科研记录。

## Repository layout

- `activeview/`: 唯一正式 Python 源码和 CLI。
- `tests/`: 仓库级 unit/integration tests。
- `docs/`: 协议、开发记录和精选科研结果。
- `experiments/`: 后续实验定义与小型结果（大型运行时产物不入库）。
- `.ai/`: AI/人类研究状态与交接记录。
- `ACTIVEVIEW_DATA_ROOT`: 外部 datasets、checkpoints、缓存和运行时结果根目录。

Directory-versioned source development has ended. Future model iterations are
tracked with Git commits/branches, experiment IDs, configs and provenance
manifests rather than copied source trees.
