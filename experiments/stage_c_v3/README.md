# Stage C-v3 Predictability Diagnostics

Stage C-v3 addresses the scientific question left open by EXP001–EXP010:
whether the approximately 64–65% one-step policy plateau is primarily a model
limitation or an information limitation caused by future perception.

These are read-only diagnostics over frozen Stage A/B/C-v0 artifacts. The
canonical split remains 589/197/194 records. Only Train and Val are permitted;
Test is locked.

| Experiment | Question | Status |
|---|---|---|
| EXP011 | How much easier is utility prediction when actual future candidate perception is revealed? | COMPLETED (review pending) |
| EXP012 | How predictable is utility from online-available state and geometry alone? | COMPLETED (review pending) |
| EXP013 | Does frozen v0 rank a useful proposal set even when Top-1 is wrong? | COMPLETED (review pending) |

No Test evaluation was performed. EXP011 used the corrected 17-D
future-feature schema. No Habitat, RGB, YOLO, VideoPose3D or ST-GCN rerun is
part of these diagnostics.

## Interpretation boundaries

EXP011 is intentionally a future-perception teacher upper bound. It is
diagnostic-only and must never be reported as a deployable online policy.
EXP012 uses Train candidates as the only nearest-neighbour reference database
for Val queries. EXP013 is a post-hoc Val audit and its Top-K oracle quantities
are not online policy performance.

All three diagnostics retain the preregistered limitation from Stage C-v2:
the current skeleton representation is body-yaw canonicalized and therefore
does not preserve explicit body-to-candidate directional alignment.

## Runtime locations

Large cache and JSON outputs belong under:

```text
ACTIVEVIEW_DATA_ROOT/datasets/policy_v11_5/stage_c_v3/
ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v3/
```

The run scripts are intentionally not invoked during this preparation task.
