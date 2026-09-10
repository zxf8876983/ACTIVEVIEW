# Reduced12 action-agnostic transition utility

Train contexts: 30580; Moving Val contexts: 10080

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| H1 baseline | 0.454266 | 0.444782 | 1.000000 |
| BCE-H2 | 0.410813 | 0.402678 | 1.000000 |
| BCE-StayAware | 0.482341 | 0.468785 | 0.355952 |
| Listwise-H2 | 0.408333 | 0.401269 | 1.000000 |
| Listwise-StayAware | 0.456548 | 0.450949 | 0.450496 |
| Listwise-S0-only | 0.411012 | 0.403548 | 1.000000 |
| Full GT-margin Oracle | 0.629861 | 0.622230 | 0.608036 |

Real-H1 vs S0-only Listwise accuracy gain: -0.268 pp.
Best independent scorer: BCE-StayAware (0.482341); set-level condition: False.
A true H1 observation is used as an allowed sequential input; future candidates are targets/terminal evaluation only.

policy_test_used=false; no RGB/skeleton/DINO regeneration; frozen ST-GCN unchanged.
Tiny set-level ranker skipped because the predefined learnability gate was not met.
