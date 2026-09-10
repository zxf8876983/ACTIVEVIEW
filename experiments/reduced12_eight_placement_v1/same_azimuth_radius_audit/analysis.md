# Reduced12 Same-Azimuth Radius / Image-Scale Audit

Val Moving only: 10080 contexts and 68702 legal candidate samples. Test was not read; no model or perception artifact was changed.

## Radius candidate quality

| Radius | Candidate correct | Mean GT-logp | Mean GT-margin | Coverage | Oracle Acc | Oracle Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 m | 0.455008 | -3.941413 | -1.136297 | 0.894544 | 0.608295 | 0.597911 |
| 2.0 m | 0.365816 | -4.386552 | -2.129537 | 0.872619 | 0.518190 | 0.508525 |
| 2.5 m | 0.271216 | -5.141721 | -3.560899 | 0.821925 | 0.386361 | 0.383883 |
| 3.0 m | 0.202971 | -5.864999 | -4.681740 | 0.762599 | 0.300247 | 0.298186 |

## Full four-radius same-azimuth groups

Groups: 3193; best-radius distribution: 1.5m=0.339, 2.0m=0.258, 2.5m=0.197, 3.0m=0.207

| Radius | Mean margin | Median margin | Correctness |
|---:|---:|---:|---:|
| 1.5 m | -0.663430 | -0.321071 | 0.486376 |
| 2.0 m | -0.818292 | -1.084934 | 0.447541 |
| 2.5 m | -2.006595 | -2.103936 | 0.376762 |
| 3.0 m | -3.407546 | -3.480226 | 0.303476 |

## Same-azimuth pairwise effects

| Pair | Groups | Δ margin (far-near) | P far better | Wrong→correct | Correct→wrong |
|---|---:|---:|---:|---:|---:|
| 1.5_vs_2.0 | 10803 | -0.129652 | 0.521152 | 0.083773 | 0.118208 |
| 1.5_vs_2.5 | 7043 | -1.373595 | 0.451086 | 0.083771 | 0.197501 |
| 1.5_vs_3.0 | 5598 | -2.735635 | 0.387996 | 0.083780 | 0.272955 |
| 2.0_vs_2.5 | 8970 | -1.060554 | 0.485396 | 0.069677 | 0.131327 |
| 2.0_vs_3.0 | 6142 | -2.446050 | 0.409313 | 0.084500 | 0.217193 |
| 2.5_vs_3.0 | 8147 | -1.030967 | 0.512950 | 0.091445 | 0.152081 |

## SceneVisibility-matched subset

Fixed threshold: 0.05; matched four-radius groups: 2887; combined pair groups: 42766; mean Δ margin (far-near): -1.171220; wrong→correct: 0.082846.

## Oracle comparison

BestRadiusOracle Accuracy/Macro-F1: 0.709623/0.704598; Same-Azimuth-BestRadius group Accuracy/Macro-F1: 0.495409/0.489181; FrozenStageCv0: 0.454266/0.444782; AnyCorrect Oracle: 0.728274/0.729637.
Same-Azimuth-BestRadius covers 19386 groups; closest-radius→best-radius wrong→correct=0.108532, correct→wrong=0.000000.
Only radius was changed inside the radius oracle; all terminal labels came from the selected real archived skeleton's frozen ST-GCN cache.

## Image-scale proxies

Spearman radius→projected-area: -0.867799; projected-area→GT-margin: 0.180444; distance→GT-margin: -0.175286.

## Qualitative cases

- case_0001: outside_legal_candidate_set (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0002: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0003: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0004: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0005: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0006: outside_legal_candidate_set (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0007: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0008: outside_legal_candidate_set (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0009: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0010: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0011: outside_legal_candidate_set (selected/oracle viewpoint is outside this context's legal candidate set).
- case_0012: unavailable (selected/oracle viewpoint is outside this context's legal candidate set).

## Scientific judgment

Q1. Is 1.5m systematically biased? 1.5m is systematically the strongest radius in this archive (higher mean margin/correctness than farther radii), rather than being the weakest.
Q2. Is there a preferred 2.0/2.5/3.0m interval? No; the aggregate and matched tables favor the near 1.5m condition, while the best-radius distribution remains heterogeneous.
Q3. Does radius still matter at matched azimuth and SceneVisibility? The matched mean far-near margin is -1.171220; this is consistent with a material residual radius effect.
Q4. Are recent same-azimuth failures global or isolated? They are consistent with a broader tendency for farther radii to lose margin, although individual flips remain heterogeneous.
Q5. Most reasonable explanation: D. distance / image scale / perspective is the strongest remaining explanation: radius differences persist in the SceneVisibility-matched subset.

Flags: `test_used=false`; `training_used=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `future_candidate_skeleton_used_for_terminal_diagnostic_only=true`; `new_rgb_rendered=false`; `selector_modified=false`; `deployable=false.
