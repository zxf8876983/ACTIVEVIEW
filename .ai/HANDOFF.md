# Handoff

Status: CLEAN

EXP020 and EXP021 are complete Train→Val experiments. EXP020 used the frozen
EXP014 contextual token with a small binary gate and achieved 0.661757 Accuracy
/ 1.453868 mean regret; decision **INCONCLUSIVE**. EXP021 used an offline
full-information contextual-bandit objective and collapsed to all-Stay,
achieving 0.649103 Accuracy / 1.450498 mean regret; decision **REJECT**.

Runtime results:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP020_contextual_latent_gate/`
and
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP021_contextual_bandit/`.
Compact experiment records are under the corresponding directories in
`experiments/stage_d/`.

Test remains locked and was not read. No EXP014 retraining, perception
regeneration, Habitat rendering, ST-GCN retraining or Stage A/B/C-v0/
EXP014–EXP019 artifact modification was performed. EXP022 must not be started
automatically; human review is required before any next experiment.
