# Current Task

# Historical Route-1 × Shared Adapted Head Synergy Audit — completed 2026-09-12

Replayed the highest compliant reduced12 Route-1 checkpoint (Stay-aware
GTMargin Listwise / `margin_listwise`) on the matched 10,080 Moving-Val
contexts, then evaluated the identical selected viewpoints with the original
frozen ST-GCN head and the frozen shared adapted head.  No Policy Test was
read; no model, perception data or cache was modified.

The audit found no deployment-legal, matched-protocol historical Route-1
result above 50%.  The 0.502778 RealEvidence-GTMarginListwise result was
excluded because it consumes future candidate evidence, and old EXP036
16-class results were protocol mismatches.  The selected closest compliant
method's archived result is 0.458730 Accuracy / 0.446342 Macro-F1.

On the matched replay, Stay/current was 0.254266/0.235500 with the original
head and 0.302579/0.292976 with the shared head.  The historical policy was
0.458730/0.446342 with the original head and 0.509524/0.509806 with the
shared head.  Recognizer gain at Stay was +4.831pp; policy gain was +20.446pp
under the original head and +20.694pp under the shared head.  Combined gain
was +25.526pp and additive synergy was +0.248pp.  Shared candidate-only Legal
AnyCorrect Coverage was 0.771726 and its GT-TrueLogP Oracle was
0.753175/0.755358.

The +2pp gate for a possible policy retraining is passed, but no second-stage
retraining was started automatically.  Full artifacts are under
`experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/`.

Task status: **CLEAN**.  Await explicit approval before retraining the
historical policy for the shared head.
