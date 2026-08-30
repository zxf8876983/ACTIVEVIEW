# Current Task

## EXP018 — executed-candidate gate alignment completed

Stage C-v0, corrected EXP014, EXP015, EXP016 and EXP017 remain frozen. EXP018
was a Val-only, no-training audit that froze the Stage C-v0 first action/p1
and the corrected EXP014 learned p2/p3 ranking. It compared
`y_any=max(true U2)>0` with `y_exec=true U2(c_hat)>0`, where `c_hat` is the
candidate selected by the frozen learned ranking.

## Result

On 13,987 Val episodes (9,742 v0-Move), `y_any` was positive for 5,477 and
`y_exec` for 4,281. The ranking-induced mismatch count was 1,196 (21.84% of
any-positive episodes). The executed-candidate oracle reached Accuracy
0.743119 / Macro-F1 0.693231 / mean regret 0.761339, compared with the
any-positive oracle 0.720026 / 0.670190 / 0.969138 and EXP014 0.658254 /
0.610153 / 1.422463. EXP017's 2,838 extra moves contained 1,497
executed-nonpositive cases.

## Protocol boundaries

- EXP018 read frozen Val artifacts only; Test was not read or used.
- No training, Habitat rendering, perception regeneration, ST-GCN retraining,
  or Stage A/B/C-v0/EXP014/EXP015/EXP016/EXP017 modification was performed.
- EXP018 is an offline diagnostic with decision **INCONCLUSIVE**; no policy
  was accepted and EXP019 was not started.

## Status

EXP018 completed. Await human scientific review before authorizing a next
experiment.
