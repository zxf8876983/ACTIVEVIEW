# EXP048

Using the same CF_CORRECTNESS_MLP weights, the empirical Accuracy quantities
were:

| Quantity | Accuracy |
|---|---:|
| A: best legal imagined observation | 0.664403 |
| B: same decision with true observation | 0.629013 |
| C: imagined observation + GT-label oracle | 0.730535 |
| D: true observation + GT-label/fixed-first oracle | 0.771502 |

Thus `world-model gap = -0.035390`, `decision/label gap = 0.066133`, and
`residual oracle gap = 0.040967`; these are descriptive empirical quantities,
not an additive causal decomposition.  The best legal method rescued 796
episodes and caused 582 harmful correctness changes relative to EXP014
(net +214).  Subgroup results (current correctness, Train-defined entropy and
confidence quartiles, special groups, and all 16 classes) are in
`subgroup_analysis.json`; rescue/harm counts are in `rescue_harm.json`.
Paired McNemar/bootstrap statistics versus EXP014, EXP023, and the fixed-first
oracle are in `paired_statistics.json`.  Privileged controls are marked
non-deployable and Test was not accessed.
