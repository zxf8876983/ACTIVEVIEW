# Reduced-15 BABEL ST-GCN

独立的 15 类动作识别数据协议与 checkpoint。每类从 Official BABEL Train
最多采样 300 条、从 Official BABEL Val 最多采样 100 条；采样优先保证
unique source、subject、AMASS dataset 与 duration-bin 多样性。Official Train
按 90/10 生成 ST-GCN development Train/Val，本实验不生成或读取 Test。

类别顺序固定为：`walk`, `sit`, `stand up`, `crawl`, `stumble`, `wave`,
`clap`, `throw`, `clean something`, `jump`, `stretch`, `kick`, `knock`,
`punch`, `take/pick something up`。

## 运行入口

```bash
/home/zxf/anaconda3/envs/habitat/bin/python activeview/scripts/data/prepare_reduced15_dataset.py
/home/zxf/anaconda3/envs/habitat/bin/python activeview/scripts/data/generate_reduced15_skeleton_dataset.py --subset stgcn_development --split train --split val --device cuda:0
```

训练入口复用 `train_selected16_habitat_stgcn.py`，通过 label mapping 自动
构建 15 类 ST-GCN；本轮使用固定 30-frame resampling，未启用 temporal
segment jitter。
