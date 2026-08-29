# Handoff

Status: CLEAN

EXP002 was evaluated on Val only and rejected: P90 regret was 5.6153066 versus
the 5.6078177 baseline, with no consistent improvement in the secondary
metrics. Test was not used; the detailed conclusion is in the EXP002
`analysis.md`.

EXP003 relative-geometry representation is recorded as REJECTED because its
2.094% mean-regret improvement missed the pre-registered 5% target. Its
Val-only metrics and geometry-bias diagnostic remain in
`experiments/stage_c_v1/EXP003_relative_geometry/` as evidence only.

EXP004–EXP007 completed independent Train→Val runs against frozen Stage C-v0
and are recorded as rejected diagnostic directions. No Test evaluation was run.

Stage C-v2 preparation is complete in `experiments/stage_c_v2/`:
EXP008 uses mean-pooled frozen ST-GCN joint tokens, EXP009 uses one
candidate-conditioned cross-attention layer over those tokens, and EXP010
encodes the current `[3,30,17]` skeleton with a lightweight Transformer and
candidate queries. Shared cache construction and Train→Val entry points are
implemented but were not executed. No Stage A/B/C-v0 artifacts changed, and no
training, Test, Habitat, YOLO or VideoPose3D run was started.
