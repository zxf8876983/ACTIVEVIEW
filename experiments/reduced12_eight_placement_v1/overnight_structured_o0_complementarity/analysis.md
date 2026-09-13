# Overnight Structured O0 Complementarity Sweep

## Protocol

- Policy Test: false; recognizer and shared head frozen; no RGB/skeleton/DINO generated.
- O0 is an already acquired full 30-frame current-view observation. The selector sees only O0 representations and legal candidate geometry; candidate observation evidence is restricted to Train utility targets and privileged Val references.
- Train: 46324 contexts; Moving Val: 10080 contexts; Train candidate samples: 305175; Val candidate samples: 68702.
- B2 chooses exactly one non-stay candidate from the Stage-A legal candidate pool; `val_all32.json` is provenance only and is not an action set.

## Main Val results

| Method | Acc | Macro-F1 | ΔAcc vs current |
|---|---:|---:|---:|
| Random B2 | 0.429762 | 0.424594 | -8.760 pp |
| StaticViewPairPrior | 0.508234 | 0.497486 | -0.913 pp |
| SoftBelief-ViewPairPrior | 0.505952 | 0.495751 | -1.141 pp |
| CurrentBaseline | 0.517361 | 0.504616 | +0.000 pp |
| PosteriorOnly+Geometry | 0.510020 | 0.497620 | -0.734 pp |
| IntermediateGlobal+Geometry | 0.511905 | 0.497953 | -0.546 pp |
| PartPool+Geometry | 0.511806 | 0.498844 | -0.556 pp |
| PartPool+Posterior+Geometry | 0.512996 | 0.499961 | -0.437 pp |
| Temporal3+Geometry | 0.508631 | 0.496651 | -0.873 pp |
| Temporal3+Posterior+Geometry | 0.513294 | 0.499310 | -0.407 pp |
| PartTime15+Geometry | 0.514385 | 0.503291 | -0.298 pp |
| PartTime15+Posterior+Geometry | 0.512302 | 0.500314 | -0.506 pp |
| MotionDescriptor+Posterior+Geometry | 0.515675 | 0.501470 | -0.169 pp |
| RawO0-SmallEncoder | 0.517460 | 0.503193 | +0.010 pp |
| PartTime-FiLM | 0.511905 | 0.502259 | -0.546 pp |
| PartTime-Bilinear | 0.515179 | 0.503945 | -0.218 pp |
| CandidatePartAttention | 0.509226 | 0.495568 | -0.813 pp |
| CandidateTemporalAttention | 0.512202 | 0.501095 | -0.516 pp |
| CurrentBaseline-PairLogP | 0.513294 | 0.500702 | -0.407 pp |
| PartTime15+Posterior+Geometry-PairLogP | 0.514187 | 0.500742 | -0.317 pp |
| DirectTop1Ranker | 0.518056 | 0.508831 | +0.069 pp |
| PairMargin Oracle | 0.706647 | 0.701819 | +18.929 pp |

Current baseline (historical matched branch): 0.517361 Acc / 0.504616 Macro-F1.
Best structured branch (RawO0-SmallEncoder): 0.517460 Acc / 0.503193 Macro-F1; gain over current +0.010 pp.
Pair-margin oracle: 0.706647 Acc / 0.701819 Macro-F1; remaining gap from best structured 18.919 pp.

## Priors, interactions, and formulation

Static pair prior=0.508234; soft-belief pair prior=0.505952. Their proximity to the current branch is a direct check for fixed viewpoint-pair bias.
PairLogP target ablation: Current=0.513294, D2=0.514187; PairMargin remains the registered primary target.
DirectTop1Ranker=0.518056; it uses masked candidate-set CE and is a formulation diagnostic.

## Shuffle and structured sensitivity
O0 shuffle: 0.506944 (drop 1.052 pp); geometry shuffle: 0.424206 (drop 9.325 pp); both shuffle: 0.423413.
Permutation source branch: PartTime-Bilinear; body-part and temporal drops are stored in the dedicated JSON files.

## Stratification and interpretation

The O0-correct/O0-wrong, confidence-quartile, and low-frame0-visibility subsets are reported without changing the main protocol. Candidate true observations are not selector inputs. The best structured branch is selected by Moving-Val utility-learning checkpoint loss, while the summary table is descriptive.
Decision: **KILL STRUCTURED O0 REPRESENTATION**.
Conditional B3 gate: SKIPPED_GATE_NOT_MET (requires best Acc >= 0.54 and gain >= +2 pp).
Structured body/time representations are interpreted as useful only if they exceed both the current selector and the static/soft pair-prior baselines by the preregistered margins.

## Scientific conclusion

The best structured O0 branch is effectively tied with the current selector and remains far below the PairMargin/AnyCorrect ceiling. This sweep therefore does not support claiming that frozen final/intermediate O0 representations recover substantial candidate complementarity; the preregistered next-step decision is to stop expanding this representation family unless a later protocol change is explicitly approved.
