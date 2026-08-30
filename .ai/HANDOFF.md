# Handoff

Status: CLEAN

EXP019 executed-candidate-aware gate is complete as a Train→Val,
single-model, fixed-threshold experiment and is **INCONCLUSIVE**. It trained
on 29,133 Train second-step examples for 30 fixed epochs and evaluated 9,742
Val v0-Move episodes. EXP019 reached 0.656681 Accuracy / 1.429851 mean
regret, slightly worse than EXP014 (0.658254 / 1.422463); candidate identity
mismatch count was zero.

Runtime result:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP019_executed_candidate_gate/`.
Compact experiment record:
`experiments/stage_d/EXP019_executed_candidate_gate/`.

Test remains locked and was not read. No EXP014 retraining, perception
regeneration, Habitat rendering, ST-GCN retraining or Stage A/B/C-v0/
EXP014–EXP018 artifact modification was performed. EXP020 must not be started
automatically; human review is required before any next experiment.
