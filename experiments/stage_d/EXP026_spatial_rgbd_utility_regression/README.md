# EXP026 — Spatial RGB-D Utility Regression

EXP026 is preregistered before EXP025 Val results are inspected. It augments
the fixed EXP025 spatial DINOv2 representation with compact Habitat metric
depth features for already visited Stage-D `s0` and `s1` observations only.
The first-step Stage C-v0 action, EXP014 `c_hat` ranking, RGB dataset and all
perception artifacts remain frozen.

Depth is rendered only for the unique Train/Val `s0`/`s1` observation keys,
using frame 15, the saved skeleton camera states and the canonical Habitat
sensor configuration. Each depth image is represented as 4×4 pooled
`[mean,min,std,valid_ratio]` values (16×4 float16); p2/p3 depth is never
rendered or loaded. The depth branch is shared `Linear(4,32) → GELU` with
mean pooling. The final 609-D input is passed to a fixed 609→128→64→1
SmoothL1 regression head predicting raw `true_U2(c_hat)`.

Training is fixed at 30 Train-only epochs (Adam, lr 1e-3, batch 256, seed 42),
with one Val evaluation and no Test access or tuning. EXP025's spatial RGB
cache is reused only if its provenance matches this configuration.

The configuration is frozen before EXP025 results are read. Runtime depth,
RGB cache, checkpoints and predictions remain external under
`/home/zxf/MG08/robot/ActiveView/`.
