# Handoff

Status: CLEAN

EXP024 DINOv2 RGB Context Utility Regression Pilot is complete. Runtime cache,
checkpoint and predictions are under
`/home/zxf/MG08/robot/ActiveView/`; source summaries are under
`experiments/stage_d/EXP024_dinov2_rgb_context/`. The cache uses 77,750 unique
visited `s0`/`s1` RGB observations (58,266 Train; 19,484 Val), float16 768-D
DINOv2 global CLS embeddings, and no future-candidate RGB.

Val EXP024: Accuracy 0.657682, Macro-F1 0.609503, mean regret 1.414353,
P90 5.431195, headroom 0.782434; candidate identity mismatches = 0. Relative
to EXP022, Accuracy delta = -0.002216 and mean-regret delta = -0.002142.
Test remained locked. No upstream artifact, RGB dataset, perception pipeline
or Habitat rendering was modified. Do not start EXP025 automatically.
