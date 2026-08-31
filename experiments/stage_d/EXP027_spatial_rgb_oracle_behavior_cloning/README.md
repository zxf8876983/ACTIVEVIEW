# EXP027 — Spatial-RGB Oracle Behavior Cloning

EXP027 directly predicts the three second-step actions `{Stay, p2, p3}` from
legal current-state skeleton/delta features and the existing EXP025 spatial
DINOv2 RGB features for visited `s0` and `s1` only. The first Stage C-v0
decision and `p1` remain frozen. Train labels are generated and regression
checked against the frozen Fixed-first Second-Step Oracle; true U2 is never a
model input. The model is trained with unweighted CrossEntropyLoss for 30
Train-only epochs (seed 42, Adam, learning rate 1e-3, batch 256), then applied
once to Val. Test is locked.

Runtime checkpoints and predictions are external under
`/home/zxf/MG08/robot/ActiveView/experiments/stage_d/EXP027_spatial_rgb_oracle_behavior_cloning/`.
