# Handoff

Status: CLEAN

EXP022 and EXP023 are complete Train→Val experiments. EXP022 used the frozen
EXP014 contextual token plus predicted utility for raw executed-candidate U2
regression and achieved 0.659898 Accuracy / 1.416495 mean regret; decision
**ACCEPTED as research-direction evidence**. EXP023 used a supervised
warm-start followed by fixed full-information contextual-bandit optimization,
avoided all-Stay collapse, and achieved 0.660470 Accuracy / 1.374664 mean
regret; decision **ACCEPTED as research-direction evidence**.

Runtime results:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP022_executed_utility_gate/`
and
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP023_warmstarted_contextual_bandit/`.
Compact experiment records are under the corresponding directories in
`experiments/stage_d/`.

Test remains locked and was not read. No EXP014 retraining, perception
regeneration, Habitat rendering, ST-GCN retraining or Stage A/B/C-v0/
EXP014–EXP021 artifact modification was performed. No Val tuning or threshold
search was used. EXP024 must not be started automatically; human review is
required before any next experiment.
