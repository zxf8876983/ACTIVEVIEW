# EXP019 — Executed-Candidate-Aware Second-Step Gate

**Status: COMPLETED; Val result is an analysis of a frozen-candidate gate.**

EXP019 trains one small binary MLP to predict whether the candidate selected
by the frozen EXP014 ranking has positive true second-step utility:

```text
c_hat = order_candidates(predicted utility, geodesic, viewpoint ID)[0]
y_exec = 1[true_U2(c_hat) > 0]
```

The input is the normalized existing 11-D Stage D candidate geometry for
`c_hat` plus its frozen EXP014 predicted utility (12-D total). True U2 is a
Train target only and is never a model input or candidate selector. The model
is a fixed `Linear(12,64) → ReLU → Linear(64,1)` trained with
`BCEWithLogitsLoss` for exactly 30 Train epochs. Val uses the fixed
`sigmoid(logit) > 0.5` rule once; no Val selection or threshold tuning occurs.

The Stage C-v0 first action/p1 and EXP014 p2/p3 ranking remain frozen. No
Stage A/B/C-v0 or EXP014 artifact is modified, and Test is not accepted.

## Reproduction

```bash
bash experiments/stage_d/EXP019_executed_candidate_gate/run.sh
```

The checkpoint, full Train log and Val gate predictions are external runtime
artifacts. The compact `result.json` and `analysis.md` are versioned here.
