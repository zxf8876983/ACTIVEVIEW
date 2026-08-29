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

Stage C-v2 EXP008–EXP010 completed authorized Train→Val runs in
`experiments/stage_c_v2/`. EXP008 uses mean-pooled frozen ST-GCN joint tokens,
EXP009 uses one candidate-conditioned cross-attention layer, and EXP010
encodes the current `[3,30,17]` skeleton with a lightweight Transformer and
candidate queries. Selected epochs were 26, 22 and 54. Runtime outputs are
under `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/`; compact results and
comparisons are in the experiment READMEs and registry. No Test evaluation was
performed. No Stage A/B/C-v0 artifacts changed, and no Habitat, YOLO or
VideoPose3D rerun occurred.

The limitation was preregistered before running: the current skeleton
representation is body-yaw canonicalized and therefore does not preserve
explicit body-to-candidate directional alignment. EXP008–EXP010 are recorded as
rejected diagnostic directions. Stage C-v3 EXP011–EXP013 predictability
diagnostics completed under the authorized Train→Val / read-only Val protocol.
EXP011 used the corrected 17-D schema (predicted-label one-hot plus entropy;
`logp_true` excluded), EXP012 used exact k=5 Train-reference/Val-query
analysis, and EXP013 used frozen-v0 Val Top-K analysis. No Test evaluation,
Habitat or perception rerun occurred. Results are recorded in
`experiments/stage_c_v3/` and the external runtime root. Await human scientific
review; do not start another experiment automatically.
