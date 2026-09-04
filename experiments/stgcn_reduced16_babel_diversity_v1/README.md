# Reduced-16 BABEL ST-GCN

本实验在新的独立 runtime root 中使用 16 类动作：`walk`, `sit`, `stand up`,
`bend`, `crawl`, `stumble`, `wave`, `clap`, `throw`, `clean something`,
`jump`, `kick`, `knock`, `punch`, `eat`, `telephone call`。每类 Official
Train/Val 上限分别为 300/100，按 unique source、subject、AMASS dataset 和
duration-bin 优先选择；Official Train 再按 90/10 划分 ST-GCN Train/Val。

数据使用固定 30-frame resampling，不启用 temporal segment jitter。生成入口
固定 16 个 Habitat worker：

```bash
/home/zxf/anaconda3/envs/habitat/bin/python activeview/scripts/data/prepare_reduced16_dataset.py
/home/zxf/anaconda3/envs/habitat/bin/python activeview/scripts/data/generate_reduced16_skeleton_dataset.py --split train
/home/zxf/anaconda3/envs/habitat/bin/python activeview/scripts/data/generate_reduced16_skeleton_dataset.py --split val
```

训练复用 `train_selected16_habitat_stgcn.py`，通过新 label mapping 自动构建
16 类模型。本轮只训练 development Train 并记录 posthoc Val，不进行 Test
评估。
