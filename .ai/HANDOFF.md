# Handoff

Status: CLEAN

EXP017 preparation is complete but its real Train calibration and Val
evaluation have not been executed. It calibrates one strict scalar
`gate_score > tau` threshold from frozen EXP014 Train predictions and applies
that frozen threshold once to Val. Candidate ranking and the Stage C-v0 first
decision remain unchanged; Test is rejected and never read.

New source:
`activeview/active_view/stage_d_gate_calibration.py` and
`activeview/scripts/analyze_stage_d_gate_calibration.py`.
Experiment record:
`experiments/stage_d/EXP017_second_step_gate_calibration/`.

No training, Habitat rendering, perception regeneration, ST-GCN retraining or
upstream artifact modification was performed. Human code review and explicit
execution authorization are required before running `run.sh`.
