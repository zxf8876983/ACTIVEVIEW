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

EXP004–EXP007 are prepared as independent PLANNED protocols against frozen
Stage C-v0. Their feature-cache builders, Move/Stay model support, configs and
Val-only run scripts are present but have not been executed. No training or
Test evaluation is authorized in this handoff.
