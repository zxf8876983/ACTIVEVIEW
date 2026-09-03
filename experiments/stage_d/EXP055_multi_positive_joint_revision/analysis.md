# EXP055 analysis

EXP055 was trained once on 29,133 Train second-step contexts (seed 42, 20
epochs, AdamW, batch 512).  The multi-positive objective was used for 27,077
contexts with at least one recognition-correct action; 25,362 had multiple
positives and 2,056 used the no-positive fallback.  Final loss was 0.854134.
The final GPU checkpoint SHA256 is
`8a6ef93ded8df94154f2045d6cf7d297c23e587ac8cf2601a83fcf3c82f1383c`.

## Real H2 Val metrics

| Variant | Population | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| EXP051-R2 | moving 9,742 | 0.675529 | 0.642612 |
| EXP055 multi-positive | moving 9,742 | **0.683022** | **0.647338** |
| EXP051-R2 | full 13,987 | 0.695503 | 0.649220 |
| EXP055 multi-positive | full 13,987 | **0.700722** | **0.650955** |

Relative to EXP051-R2, moving Accuracy/Macro-F1 increase by +0.007493/+0.004727
(+0.749/+0.473 percentage points), and full-population Accuracy/Macro-F1
increase by +0.005219/+0.001735 (+0.522/+0.173 percentage points).

## Action and class audit

There were 5,372 changed H2 trajectories; the frozen Stage C first action
changed in 0 cases.  The first and second second-step actions changed in 4,984
and 3,626 contexts, respectively.  Terminal recognition rescued 922 contexts
and harmed 849 (net +73).  Candidate identity mismatch inside the new rollout
was 0; candidate ordering/legality remained frozen, while joint revision was
allowed to choose among legal candidates.

Largest class-recall gains were crawl (+9.155 pp), stumble (+4.225 pp),
cartwheel (+3.521 pp), jog (+2.734 pp), and play instrument (+2.465 pp).
Largest declines were lie (-9.859 pp), throw (-7.512 pp), knock (-3.756 pp),
sit (-1.056 pp), and a pose (-0.302 pp).

## Scientific decision

Both moving-subset Accuracy and Macro-F1 improve, and both full-population
metrics improve.  Under the pre-registered rule this is **CASE A**: the
multi-positive joint revision shows a small but consistent real-observation H2
gain over EXP051-R2.  This is an analysis of one frozen Train→Val run, not a
Test or deployment claim; no additional loss variant was attempted.

`test_used=false`; no Test split, perception regeneration, Habitat rendering,
or ST-GCN retraining was performed.
