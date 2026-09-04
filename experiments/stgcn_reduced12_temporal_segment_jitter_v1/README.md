# Reduced-12 temporal segment jitter

This experiment extends `reduced12_babel_diversity_v1` without changing its
labels or validation data. Train motions are divided into 30 temporal
segments. Each segment contains `m` temporally ordered candidates: `m=1` for
intervals no longer than 30 frames, `m=3` for 31--300 frames, and `m=5` for
longer intervals. The Train loader samples one candidate per segment on each
access, while development Val remains the original fixed 30-frame uniform
sampling.

The runtime data is at
`/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced12_babel_temporal_jitter_v1/`.
The independent checkpoint is at
`/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced12_babel_temporal_jitter_v1/stgcn_reduced12_temporal_jitter_best.pth`.
No selected-16 or prior reduced12 artifacts were modified.
