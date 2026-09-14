# Adaptive Head × Frame0 NBV Combination Audit

Task: strict single-step next-best-view selection

Input before selection: current frame0 RGB + legal candidate geometry
Full current action before selection: false
Candidate observation before selection: false
Final evaluation: selected O1 alone
Multi-view fusion: false
Policy Test: false
Encoder: frozen ST-GCN
Compared recognition heads: shared / historical adaptive / legal-candidate-balanced adaptive
Selector architecture: historical Frame0 RGBGlobal + Geometry + VisibilityAux

Train contexts: 46324; Moving Val contexts: 10080; records: 313 / 105

## Moving-Val strict results

| Method | Head | Accuracy | Macro-F1 | Move rate |
|---|---|---:|---:|---:|
| A_random_shared | H_shared | 0.365079 | 0.364567 | 0.843254 |
| B_random_adapt_old | H_adapt_old | 0.361310 | 0.381130 | 0.843254 |
| C_random_adapt_balanced | H_adapt_balanced | 0.354167 | 0.355880 | 0.843254 |
| D_frame0_shared | H_shared | 0.506151 | 0.503722 | 0.840476 |
| E_frame0_shared_selector_adapt_old | H_adapt_old | 0.523115 | 0.542017 | 0.840476 |
| F_frame0_shared_selector_adapt_balanced | H_adapt_balanced | 0.490079 | 0.491019 | 0.840476 |
| G_frame0_adaptive_old_selector | H_adapt_old | 0.526984 | 0.547002 | 0.910417 |
| H_frame0_adapt_balanced | H_adapt_balanced | 0.493452 | 0.497652 | 0.934623 |

## Head distribution

H_shared s1: 0.507044 / 0.507893; legal candidates: 0.375418 / 0.378473.
H_adapt_old s1: 0.531052 / 0.556059; legal candidates: 0.369800 / 0.392751.
H_adapt_balanced s1: 0.488393 / 0.493137; legal candidates: 0.364633 / 0.367584.
Specialization flag: OLD ADAPTIVE HEAD IS DISTRIBUTION-SPECIALIZED.

## Gain decomposition

Random shared baseline: 0.365079; head gain (random balanced − random shared): -0.010913; historical selector gain: +0.141071; adaptive selector gain: +0.139286; combined gain: +0.128373; synergy: -0.001786.
Balanced adaptive-aware selector versus historical strict selector: -0.012698 Accuracy.
Decision: **KILL ADAPTIVE-HEAD × NBV COMBINATION**.

## Interpretation

The old and balanced heads are evaluated on the same current/Stay + Stage-A legal candidate action set. Candidate observations and recognizer outputs are never selector inputs; they are used only for Train targets or terminal/oracle diagnostics.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
future_candidate_observation_used_for_selector=false
future_candidate_recognizer_output_used_for_selector=false
gt_action_used_for_selector=false
frozen_stgcn_encoder_modified=false
deployable_selector_inputs=true
```
