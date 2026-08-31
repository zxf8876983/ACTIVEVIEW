# EXP025 — DINOv2 Spatial RGB Utility Regression

EXP025 tests whether spatial RGB context improves over EXP024's global CLS
embedding. It uses frozen `facebook/dinov2-base` ViT-B/14 patch tokens from
already visited Stage-D `s0` and `s1` RGB observations only. The native 16×16
patch grid is adaptively pooled to 4×4 (16×768 float16 tokens), passed through
a shared 768→128 projector and one shared 1-layer spatial Transformer, then
mean-pooled into `z0`, `z1` and `z1-z0`.

The 513-D input concatenates the frozen EXP014 contextual token and predicted
utility with these spatial features. Only the spatial projector, Transformer
and fixed 513→128→64→1 SmoothL1 regression head are trainable; the target is
raw `true_U2(c_hat)`. EXP014 candidate ranking and the Stage C-v0 first-step
decision remain frozen.

Training uses 30 fixed Train-only epochs (Adam, lr 1e-3, batch 256, seed 42),
with no Val tuning and one Val evaluation. p2/p3 RGB is never requested and
Test remains locked. Runtime cache, checkpoint and predictions are external
under `/home/zxf/MG08/robot/ActiveView/`.
