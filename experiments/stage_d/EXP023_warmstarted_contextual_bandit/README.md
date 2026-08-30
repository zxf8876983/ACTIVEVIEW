# EXP023 — Supervised-Warm-Started Contextual Bandit

EXP023 gives the one-step full-information contextual-bandit formulation a
single controlled warm-start test after EXP021 collapsed to all-Stay.  Phase A
trains the existing Stage-D contextual scorer to regress both candidate U2
targets on Train for 20 fixed epochs.  Phase B starts from those weights and
optimizes expected Train reward under `softmax([0,q2,q3])` for 10 fixed epochs
with the fixed entropy bonus `beta=0.001` and learning rate `1e-4`.

Stay remains a fixed zero score, the Stage C-v0 first action/p1 is frozen, and
Val is evaluated exactly once after both Train phases.  True U2 is used only
as the Train target/reward and offline diagnostics; it never enters the model
input.  No Test, PPO/DQN, perception, Habitat or ST-GCN processing is used.

Run:

```bash
bash experiments/stage_d/EXP023_warmstarted_contextual_bandit/run.sh
```

Runtime checkpoints, predictions and full summaries are kept outside Git at
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP023_warmstarted_contextual_bandit/`.
