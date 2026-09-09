# Reduced12 Future Recognition Evidence Prediction

Train/Val only; no policy Test was read. Existing visited s0 DINO and archived skeleton caches were reused; no RGB/skeleton/DINO regeneration was performed.

## Moving Val references

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S0-only | 0.254266 | 0.235500 |
| FrozenStageCv0 | 0.454266 | 0.444782 |
| Candidate-Conditioned Spatial H1 | 0.471528 | 0.463723 |
| SceneVisibility | 0.469544 | 0.456516 |
| AnyCorrect Oracle | 0.728274 | 0.729637 |

## Moving Val evidence selectors

| Selector | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |
|---|---:|---:|---:|---:|
| logp_entropy | 0.325397 | 0.315914 | -12.887pp | -12.887pp |
| logp_belief | 0.314881 | 0.308777 | -13.938pp | -13.600pp |
| feature_entropy | 0.322718 | 0.314119 | -13.155pp | -13.066pp |
| feature_belief | 0.312798 | 0.309522 | -14.147pp | -13.526pp |
| feature_logp_entropy | 0.327083 | 0.321841 | -12.718pp | -12.294pp |
| feature_logp_belief | 0.318155 | 0.313815 | -13.611pp | -13.097pp |

## Full Val

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S0-only | 0.329601 | 0.329175 |
| FrozenStageCv0 | 0.460167 | 0.450040 |
| AnyCorrect Oracle | 0.731338 | 0.731446 |
| Evidence predictors | NOT_AVAILABLE | NOT_AVAILABLE |

Best evidence selector: **feature_logp_entropy**. Evidence selectors follow the requested direct argmax over legal candidates (no added stay score). Full-Val evidence predictors are **NOT_AVAILABLE** because the existing DINO cache covers only Moving Val; no fallback/regeneration was used.

## Diagnostics

- **logp**: logp Pearson/Spearman 0.610836/0.611599; top-1/top-3 class agreement 0.437979/0.684129; score Spearman entropy/belief 0.099428/0.135518.
- **feature**: logp Pearson/Spearman 0.528697/0.533461; top-1/top-3 class agreement 0.441778/0.698539; score Spearman entropy/belief 0.087362/0.131447.
- **feature_logp**: logp Pearson/Spearman 0.613218/0.614686; top-1/top-3 class agreement 0.443932/0.690140; score Spearman entropy/belief 0.104018/0.132593.

## Scientific judgment

Simple evidence predictors do not establish a clear >=2pp gain over Candidate-Conditioned Spatial; s0+DINO+geometry may be insufficient for candidate-specific future evidence.
No larger network, hyperparameter sweep, or formal WM/JR modification was performed.

`test_used=false`; `future_candidate_skeleton_used_for_target_only=true`; `future_candidate_rgb_used=false`; `future_candidate_dino_used=false`; `deployable=true`.
