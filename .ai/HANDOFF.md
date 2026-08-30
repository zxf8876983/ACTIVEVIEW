# Handoff

Status: CLEAN

EXP017 second-step gate calibration is complete under the Train→Val protocol
and is **REJECTED** as a deployable global-threshold intervention. Train chose
`tau=-0.08218251913785934`; Val trajectory performance worsened despite better
gate balanced accuracy and Move recall. Candidate identity was unchanged when
both tau variants moved, and the frozen EXP014/OracleGate references matched.

Runtime result and calibration artifact:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP017_second_step_gate_calibration/`.
Compact experiment record:
`experiments/stage_d/EXP017_second_step_gate_calibration/`.

Test remains locked and was not read. No Stage A/B/C-v0, EXP014/015/016,
perception or Habitat artifact was modified. No follow-up experiment has been
started automatically; human review is required before any next intervention.
