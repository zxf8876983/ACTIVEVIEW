# EXP022 — Executed-Candidate Utility Regression Gate

EXP022 replaces EXP020's binary `y_exec` BCE target with direct raw
`true_U2(c_hat)` regression.  The frozen EXP014 contextual token (128-D)
plus frozen predicted utility (129-D) is used as input to a fixed
`Linear(129,64) → GELU → Linear(64,1)` head.  The head is trained on Train
for 30 fixed epochs with default `SmoothL1Loss`, then applied once on Val with
the strict rule `predicted_U_exec > 0`.

The Stage C-v0 first decision, EXP014 candidate ranking, Stage D cache,
perception and ST-GCN are frozen.  True U2 is a Train target/offline
diagnostic only and never an input.  Test is locked and is not read.

Run:

```bash
bash experiments/stage_d/EXP022_executed_utility_gate/run.sh
```

The checkpoint, predictions and full runtime summaries are outside Git under
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP022_executed_utility_gate/`.
