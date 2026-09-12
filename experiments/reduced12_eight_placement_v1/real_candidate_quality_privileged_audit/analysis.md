# Real Candidate Quality Privileged Audit

Moving Val only: 10,080 contexts; recognizer is the frozen reduced12 encoder plus shared adapted head. Quality selectors are privileged because they read archived future candidate quality, while GT action/correctness are excluded from quality scores.

## Main results

| Selector | Type | HAR Acc | Macro-F1 | Move rate |
|---|---|---:|---:|---:|
| Stay | deployable baseline | 0.302579 | 0.292976 | 0.000000 |
| Random legal | baseline | 0.365079 | 0.364567 | 0.843254 |
| RealPoseConfidence | privileged action-agnostic | 0.513194 | 0.513235 | 0.871230 |
| SceneVisibility | privileged action-agnostic | 0.521131 | 0.520336 | 1.000000 |
| HumanVisibility | privileged action-agnostic | 0.302877 | 0.300860 | 1.000000 |
| TotalVisibility | privileged action-agnostic | 0.504365 | 0.507568 | 1.000000 |
| ProjectedHumanArea | privileged action-agnostic | 0.483929 | 0.479323 | 1.000000 |
| VisibleJointRatio | privileged action-agnostic | 0.521131 | 0.520336 | 1.000000 |
| TemporalMotionRetention | privileged action-agnostic | 0.391071 | 0.378826 | 1.000000 |
| Historical Route-1 | learned task-aware | 0.509524 | 0.509806 | 0.972024 |
| GT-TrueLogP Oracle | privileged task-aware | 0.753175 | 0.755358 | 0.886409 |
| Legal AnyCorrect | coverage only | — | — | — |

Best action-agnostic selector: **SceneVisibility**, 0.521131 Acc / 0.520336 Macro-F1.
Against Stay: +21.855pp; against Random: +15.605pp; versus Historical Route-1 + Shared: +1.161pp (higher); gap to GT-TrueLogP Oracle + Shared: +23.204pp.

## Ranking diagnostics

Best selector within-context Spearman with shared GT true-logp: mean 0.247205, median 0.261905; candidate-level Spearman: 0.402764; GT-TrueLogP top-1/top-3 overlap: 0.230754/0.562599.

## Occlusion subset

No occlusion-stratified result is reported: existing Scene/Human artifacts have legal-candidate values but no current/stay value, so no synthetic stay score or unsupported quantile split was introduced.

## Decision

A quality selector exceeds Random by at least 5pp and is 1.161pp above the historical task-aware route on this matched audit, but it remains privileged rather than deployable. Keep future observability as a possible target, while combining it with task evidence rather than relying on quality alone.

Observation quality is not recognition utility, and a privileged real-quality selector is not a deployable future-quality predictor.

No Policy Test was read; no RGB, skeleton, DINO, recognizer, policy or runtime artifact was modified.
