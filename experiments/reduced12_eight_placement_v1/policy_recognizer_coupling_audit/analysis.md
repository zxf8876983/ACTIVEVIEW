# Policy–Recognizer Coupling Audit

Experiment: Policy–Recognizer Coupling Audit
Task: strict Frame0 single-step NBV
Final HAR: selected O1 alone
Selector candidate observation access: false
Policy Test: false
Purpose: determine whether the 52.6984% result comes primarily from fixed viewpoint prior, recognizer-policy distribution matching, or instance-conditioned Frame0 information.

Moving Val contexts: 10080; Train contexts used only for static priors: 46324.

## Main results

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Random + shared | 0.365079 | 0.364567 |
| Historical selector + shared | 0.506151 | 0.503722 |
| Historical selector + old adaptive | 0.523115 | 0.542017 |
| StaticViewPrior + old adaptive | 0.522917 | 0.546040 |
| ViewPairPrior + old adaptive | 0.519940 | 0.543270 |
| Adaptive-aware selector + shared | 0.508333 | 0.507541 |
| Adaptive-aware selector + old adaptive | 0.526984 | 0.547002 |
| Adaptive-aware selector RGB-shuffled + old adaptive | 0.484425 | 0.511408 |

## Distribution and shuffles

Adaptive-aware vs historical s1 JSD: 0.014136; top-5 overlap: 0.800; entropy bits adaptive/historical-s1: 3.3590/3.5666.
RGB shuffle: normal 0.526984, shuffled 0.484425, accuracy drop 4.256pp. Geometry shuffle accuracy drop: 15.159pp.
Adaptive-aware selected-set shared/old: 0.508333/0.526984; adaptive head gain on identical selected views: +1.865pp.

## Interpretation

Primary conclusion: **MOSTLY FIXED VIEWPOINT PRIOR**; secondary: RGB carries signal, but the adaptive-aware gain over fixed priors is below 1pp.
The old adaptive head's matched s1 gain is +1.696pp, while its legal-candidate specialization audit is recorded in the JSON artifacts.
Static and pair prior best Accuracy is 0.522917; adaptive-aware Accuracy is 0.526984. This makes the adaptive-aware minus best-prior gap +0.407pp.

Decision on 52.6984%: retain it as an informative strict Frame0 result only when accompanied by the distribution-matching caveat; do not claim a purely instance-conditioned NBV gain.

```text
policy_test_used=false
training_used=false
new_rgb_generated=false
future_candidate_observation_used_for_selector=false
future_candidate_recognizer_output_used_for_selector=false
gt_action_used_for_selector=false
terminal=selected real O1 alone
```
