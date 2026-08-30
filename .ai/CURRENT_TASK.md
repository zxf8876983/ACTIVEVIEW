# Current Task

## EXP019 — executed-candidate-aware second-step gate completed

Stage C-v0 and corrected EXP014–EXP018 remain frozen. EXP019 trained one
small `Linear(12,64) → ReLU → Linear(64,1)` binary gate on 29,133 Train
second-step examples for exactly 30 epochs. The target was
`1[true_U2(c_hat)>0]`, where `c_hat` is selected only by frozen EXP014
predicted-utility/geodesic/viewpoint-ID ranking. Val used the fixed
probability threshold 0.5 exactly once.

## Result

On 13,987 Val episodes (9,742 v0-Move), EXP019 reached Accuracy `0.656681`,
Macro-F1 `0.607936`, mean regret `1.429851`, P90 regret `5.532555` and
headroom capture `0.780325`, versus EXP014 Accuracy `0.658254`, Macro-F1
`0.610153` and mean regret `1.422463`. The executed-candidate oracle remained
at Accuracy `0.743119` / mean regret `0.761339`. EXP019 changed 984 Stay→Move
and 253 Move→Stay decisions; candidate identity mismatch count was zero.

## Protocol boundaries

- EXP019 trained only on Train and evaluated once on Val; Test was not read or
  used.
- No EXP014 retraining, Habitat rendering, perception regeneration, ST-GCN
  retraining, or Stage A/B/C-v0/EXP014–EXP018 artifact modification was
  performed.
- EXP019 is an analysis result with decision **INCONCLUSIVE**; no policy was
  accepted and EXP020 must not be started automatically.

## Status

EXP019 completed. Await human scientific review before authorizing any next
experiment.
