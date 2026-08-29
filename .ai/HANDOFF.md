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

EXP004–EXP007 have completed independent Train→Val runs against frozen Stage
C-v0. Compact results are in each `experiments/stage_c_v1/EXP00X_*/result.json`;
full checkpoints, predictions and analyses are under the corresponding
`ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v1/` directories. All four decisions
remain pending user review. No Test evaluation was run.
