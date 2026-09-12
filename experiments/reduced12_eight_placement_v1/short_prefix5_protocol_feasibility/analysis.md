# Short-prefix5 mixed-view protocol feasibility audit

Split: Moving Val only; contexts: 10,080 when run without `--limit`; training: none; Policy Test: false.
Recognizer: frozen reduced12 ST-GCN plus frozen shared adapted head. Prefix5 uses current frames 0..4 and selected candidate frames 5..29; Stay uses current frames 0..29.
This is a discrete-time view-switch approximation, not continuous navigation.

## Results

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Stay | 0.302579 | 0.292976 | 0.000000 |
| Random-ShortPrefix5 | 0.342262 | 0.329111 | 0.843254 |
| Mixed5-GT-TrueLogP | 0.687599 | 0.671197 | 0.869345 |
| Mixed5-GT-Margin | 0.720040 | 0.709917 | 0.850496 |
| FullView-GT-TrueLogP | 0.753175 | 0.755358 | 0.886409 |
| FullView-GT-Margin | 0.791964 | 0.800900 | 0.875496 |
| Mixed5 AnyCorrect Coverage | 0.720040 | — | — |
| FullView AnyCorrect Coverage | 0.791964 | — | — |

Mixed5-to-FullView GT-TrueLogP ceiling drop: 0.065575 Acc / 0.084161 F1.
Candidate ranking: within-context Spearman mean/median 0.794074/0.890909; top-1 viewpoint agreement 0.678770; top-3 overlap 0.832573.
Full-correct→mixed-wrong: 0.226582 conditional (0.085063 of all candidate samples); switch/normal 4→5 mean displacement amplification: 9.395924.

## High-occlusion subset

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stay | 0.102667 | 0.052429 |
| Random-ShortPrefix5 | 0.221107 | 0.201432 |
| Mixed5-GT-TrueLogP | 0.533123 | 0.506643 |
| FullView-GT-TrueLogP | 0.636650 | 0.649576 |
| Mixed5 AnyCorrect Coverage | 0.581015 | — |

Largest FullView→Mixed5 GT-TrueLogP per-class F1 drops: knock 0.306717, stand up 0.154958, touching face 0.130562, punch 0.114357.

## Decision

Decision: **KEEP** under the preregistered thresholds (Mixed5 oracle Acc=0.687599, FullView−Mixed5=0.065575).
A subsequent selector-training experiment is allowed only as a separately authorized step; this audit itself trained no selector.
The mixed-view boundary is substantially larger than normal same-view transitions; if the short-prefix ceiling is low, representation discontinuity is a plausible protocol ceiling.

The candidate skeleton is used only to construct the privileged mixed terminal evaluation and oracle targets. No future candidate observation is available to a deployable selector in this audit.
