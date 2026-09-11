# Reduced12 candidate utility source decomposition

Moving Val contexts: 10080; candidate map samples: 68702

## Alignment

Body-relative convention confirmed: `True`; max position/azimuth error 0.000019 deg.

## Consistency

| Comparison | Usable pairs | Spearman mean | Spearman median | Sign agreement mean | Correct-map Jaccard mean |
|---|---:|---:|---:|---:|---:|
| Same motion across scene/placement | 50635 | 0.268070 | 0.400000 | 0.738559 | 0.520610 |
| Different motion matched | 264551 | 0.147943 | 0.190476 | 0.618859 | 0.304401 |
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

policy_test_used=false; training_used=false; future_candidate_observation_used_at_inference=false.
