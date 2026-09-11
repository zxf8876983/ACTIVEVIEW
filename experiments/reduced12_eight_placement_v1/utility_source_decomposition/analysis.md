# Reduced12 candidate utility source decomposition

Moving Val contexts: 10080; candidate map samples: 68702

## Alignment

Body-relative convention confirmed: `True`; max position/azimuth error 0.000019 deg.

## Consistency

| Comparison | Usable pairs | Spearman mean | Spearman median | Sign agreement mean | Correct-map Jaccard mean |
|---|---:|---:|---:|---:|---:|
| Same motion across scene/placement | 50635 | 0.268070 | 0.400000 | 0.738559 | 0.520610 |
| Different motion matched | 37188 | 0.146669 | 0.187879 | 0.619279 | 0.304816 |
| Same scene/placement | 264551 | 0.147943 | 0.190476 | 0.618859 | 0.304401 |
| Same scene/placement, same action | 21647 | 0.280596 | 0.356643 | 0.730829 | 0.459485 |

## Leave-one-group additive decomposition

| Model | Explained variance | Pearson | Spearman | Residual variance / total | Samples |
|---|---:|---:|---:|---:|---:|
| motion_only | 0.479493 | 0.693387 | 0.681225 | 0.520507 | 68702 |
| scene_only | 0.104153 | 0.325211 | 0.318382 | 0.895847 | 68702 |
| motion_scene | 0.558848 | 0.748154 | 0.737853 | 0.441152 | 68702 |

## Interpretation

The report separates motion consistency, scene/placement consistency, and additive motion+scene effects. Low additive explained variance with a large residual indicates motion×scene×viewpoint interaction; this audit does not launch a new policy.

## Explicit source questions

1. **Same motion across scenes:** utility maps retain moderate but imperfect
   consistency (Spearman mean 0.268), clearly above the matched
   different-motion baseline (0.147).
2. **Same scene/placement across motions:** consistency is weak when actions
   are mixed (0.148), but action-matched pairs reach 0.281; the scene is not
   an action-independent template.
3. **Stronger effect:** motion-only explains 47.9% of variance, scene-only
   10.4%, and the additive motion+scene model 55.9%; the combined model is
   strongest.
4. **Interaction residual:** 44.1% of variance remains after the additive
   model, so motion×scene×viewpoint interaction is substantial.
5. **Three/four hops:** K=3 and K=4 reach 0.6548 and 0.6884 Accuracy versus
   0.7096 Full; they are useful but not close enough to replace global
   access.
6. **Correct-view basin:** correct candidates are neither all isolated nor a
   single smooth basin: the largest component contains 76.3% of correct
   nodes on average, while 1.07 correct nodes are isolated per context with
   a correct view.
7. **Greedy versus reachability:** privileged greedy search reaches only
   0.5582 Accuracy after four steps, below the K=4 reachability ceiling
   0.6884, so monotonic local search is not sufficient.
8. **Supported route:** the evidence most supports **D, strong
   motion×scene×view interaction**, with motion as a secondary source. A
   sequential information-acquisition protocol is better motivated than a
   one-shot scalar predictor, but should not be started automatically from
   this audit.

policy_test_used=false; training_used=false; future_candidate_observation_used_at_inference=false.
