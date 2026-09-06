# Reduced14 WM-E Val Diagnostics

Only the frozen reduced14 Val Stage-D contexts and frozen counterfactual cache were read. No model was retrained and no Test file was accessed.

- Contexts: 14809; legal candidate samples: 426474.
- Candidate recognition agreement: 0.477298 over WM-E imagined versus archived-candidate ST-GCN classes.
- Ground-truth-class logp correlation: Pearson 0.4988114169980131, Spearman 0.6159245198318315.
- Positive ranking hit: Top-1 0.516780, Top-3 0.681748; oracle-positive contexts 12912.
- Conditional on an oracle-positive candidate: Top-1 0.5927044609665427, Top-3 0.781908302354399.
- Weakest action agreement: punch (0.371819); weakest action Spearman: knock (0.024442).

## Body-relative yaw

Yaw bins group candidates by `wrap(candidate azimuth - placement yaw)`. Within each bin, Top-1 positive hit ranks only that bin's candidates for a context; agreement and correlations remain candidate-level.

| Bin | Samples | Contexts | Agreement | Pearson | Spearman | Top-1 positive hit |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 53898 | 14732 | 0.483562 | 0.510274 | 0.630031 | 0.393022 |
| ±45 | 106343 | 14732 | 0.483793 | 0.504812 | 0.621062 | 0.442778 |
| ±90 | 106510 | 14709 | 0.469712 | 0.493313 | 0.612350 | 0.414916 |
| ±135 | 106323 | 14723 | 0.473914 | 0.496576 | 0.612247 | 0.395707 |
| 180 | 53400 | 14710 | 0.479906 | 0.487471 | 0.600495 | 0.302311 |

## Interpretation

The limiting stage is identified by comparing WM-E-to-real candidate recognition agreement and true-class ranking correlation. Low agreement with low ranking hit indicates the world-model observation prediction is the dominant bottleneck; high agreement but weak positive ranking indicates the downstream candidate scoring/utility alignment is the remaining bottleneck. Yaw-bin gaps are evidence for a body-relative viewpoint sensitivity only when they are materially larger than the other bins.
Here the overall agreement is 0.477298 and positive Top-1 hit is 0.516780; this indicates substantial WM-E candidate-recognition mismatch plus a remaining ranking/utility loss. The 180-degree bin has lower Top-1 hit (0.302311) and Spearman (0.600495) than the ±45-degree bin (0.442778/0.621062), but agreement is similar across bins, so body-relative yaw is a secondary rather than sole bottleneck.

`test_used=false`; `future_candidate_observation_input=false`.
