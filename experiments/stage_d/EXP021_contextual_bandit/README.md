# EXP021 — Offline Contextual-Bandit Joint Second-Step Policy

EXP021 treats the frozen second-step decision as a one-step,
full-information contextual bandit with actions `{Stay, p2, p3}`. The model
uses legal Stage-D `s0_feature`, `s1_feature`, `delta_semantic` and candidate
geometry inputs, scores Stay as fixed zero, and learns candidate scores by
maximizing the expected Train utility under `softmax([0, q2, q3])`.

Training is fixed at 30 epochs, batch size 256, Adam learning rate `1e-3`,
seed 42, with no entropy bonus, no Val selection and no Test access. The
Stage C-v0 first action/p1 remains frozen. Unlike EXP019/020, EXP021 may
select either candidate after choosing Move; true U2 is a Train reward only
and is never a model input or Val action selector.

Run:

```bash
bash experiments/stage_d/EXP021_contextual_bandit/run.sh
```

The full checkpoint/predictions are external runtime artifacts. The compact
`result.json` and `analysis.md` are versioned for review.
