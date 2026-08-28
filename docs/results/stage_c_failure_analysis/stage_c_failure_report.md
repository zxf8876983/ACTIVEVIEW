# Stage C-v0 Failure Analysis (Set Ranker Test)

## Scope and frozen-artifact checks

- Test episodes: **13774**; independent motion records: **194**.
- Analysis uses frozen Stage A/B/C JSONL and Set Ranker predictions only; no Habitat, YOLO, VideoPose3D, retraining or upstream regeneration was run.
- The Stage C validator report was checked before analysis and must be `passed=true`, `error_count=0`.
- Episode-level rows are repeated observations, not IID samples; record-level aggregation is reported separately.

## Regret distribution and groups

- Thresholds derived from Test regret: p50=0.007464, p75=1.967694, p90=6.143025, p95=8.420851, p99=13.031287.
- G0 (≤1e-3): 5922 (42.99%); G1 (1e-3–p75): 4408 (32.00%); G2 (p75–p90]: 2066 (15.00%); G3 (>p90): 1378 (10.00%).
- Catastrophic top 5%: 689 (5.00%); extreme top 1%: 138 (1.00%).

## Decision-failure taxonomy

| Type | Meaning | Count | Ratio | Mean regret | P90 regret |
|---|---|---:|---:|---:|---:|
| A_missed_move | Move required, model stayed | 3024 | 21.95% | 1.7777 | 6.7174 |
| B_unnecessary_move | Stay required, model moved | 736 | 5.34% | 3.0731 | 8.2106 |
| C1_wrong_near_optimal | Move/move wrong candidate, near-equivalent | 1435 | 10.42% | 0.0019 | 0.0060 |
| C2_wrong_high_utility_loss | Move/move wrong candidate, high loss | 4468 | 32.44% | 3.2648 | 8.3750 |
| D_correct_safe_action | SafeOracle action matched | 4111 | 29.85% | 0.0000 | 0.0000 |

The dominant failure type is **C2_wrong_high_utility_loss**. High-regret action rates are: lie (41.31%), play instrument (31.92%), knock (26.76%), stumble (22.07%), throw (15.18%). Region high-regret rates are: dining_area (11.05%), living_room (10.26%), kitchen (9.41%), bedroom (9.08%).

## Candidate miss versus utility quality

- Candidate exact miss rate: 66.07% (9100 episodes).
- For misses, CandidateOracle utility minus selected utility: mean 2.1013, median 0.2298, p90 7.1201.
- Among misses with CandidateOracle utility >1e-6, selected utility reaches ≥90%/75%/50% of CandidateOracle in 29.45%/36.63%/44.67%.
This directly tests whether low exact hit is mainly near-equivalent selection or materially bad utility loss.

## Action-class and region findings

Lowest Set Accuracy classes: stumble (16.90%), play instrument (21.60%), lie (26.76%), knock (36.15%), throw (48.36%).
Highest mean-regret classes: lie (5.022), play instrument (4.490), stumble (3.330), knock (3.289), throw (2.285).
Region breakdown is in the machine-readable summary; it should not be treated as a scene-held-out result.

## Current state and geometry diagnostics

- Spearman correlations with regret (absolute ordering): current_logp_true=-0.237, current_entropy=+0.184, current_margin=-0.180, current_pose_confidence=+0.023.
- State-group summaries compare entropy, margin, pose confidence, current correctness, move rate and SafeOracle headroom in the JSON artifact.
- Geometry separates CandidateOracle geometry (all Episodes) from SafeOracle move geometry (Move-only Episodes); selected geometry and candidate-set utility-gap bins are also reported.
- Symmetric-geometry analysis uses explicit tolerances and a data-derived q90 utility-difference threshold; it is indirect evidence only and does not estimate body yaw.

## Record-level concentration

- Top 10% records account for 40.20% of catastrophic top-5% episodes (4.02x the uniform 10% baseline).
- Worst and best 20 motion records are included in `record_failure_table.csv` and the JSON summary.

## Scientific answers

1. Catastrophic failures are substantially but not exclusively concentrated: the worst 10% of records account for 40.20% of top-5% failures.
2. The dominant decision failure is C2_wrong_high_utility_loss; wrong-candidate severity is separated into C1/C2.
3. Exact candidate hit versus aggregate headroom is explained by the miss utility-gap and ratio statistics above.
4. The hardest action classes by Set accuracy are stumble, play instrument, lie, knock, throw.
5. The strongest state correlation by absolute Spearman magnitude is current_logp_true (-0.237); this is descriptive, not causal.
6. Geometry and azimuth results are descriptive binned evidence; no single geometry variable is assumed causal.
7. Symmetric-geometry ambiguity affects 1053 episodes under the stated tolerance, with 2350 large-difference pairs; explicit enrichment ratio=1.22x versus the overall high-regret baseline.
8. Evidence for adding perceived body orientation is **weak / inconclusive** unless symmetric ambiguity is clearly enriched among high-regret cases; no body yaw was added here.
9. Current evidence most directly supports hard-example/long-tail analysis (E) and improved current-state/candidate-set representation (B/A) as hypotheses for later review. It does not justify changing Stage C-v0 in this task.

## Artifacts

- `summary`: `/tmp/activeview_stage_c_failure_analysis_v2/stage_c_failure_summary.json`
- `episode_table`: `/tmp/activeview_stage_c_failure_analysis_v2/episode_failure_table.csv`
- `record_table`: `/tmp/activeview_stage_c_failure_analysis_v2/record_failure_table.csv`
- `figures`: `/tmp/activeview_stage_c_failure_analysis_v2/figures`

### Per-action detail

| Action | n | NoMove Acc | Set Acc | SafeOracle Acc | Set gain | Mean regret | P90 regret | Headroom | G3 rate | Top-5% rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stumble | 639 | 9.08% | 16.90% | 41.63% | +7.82% | 3.330 | 8.549 | 60.49% | 22.07% | 10.80% |
| play instrument | 213 | 15.96% | 21.60% | 62.44% | +5.63% | 4.490 | 12.296 | 50.94% | 31.92% | 20.19% |
| lie | 213 | 40.85% | 26.76% | 82.16% | -14.08% | 5.022 | 11.991 | 31.61% | 41.31% | 26.76% |
| knock | 213 | 14.08% | 36.15% | 55.40% | +22.07% | 3.289 | 9.659 | 71.01% | 26.76% | 15.02% |
| throw | 639 | 24.73% | 48.36% | 76.06% | +23.63% | 2.285 | 7.832 | 72.41% | 15.18% | 8.45% |
| walk | 1420 | 32.61% | 51.97% | 81.83% | +19.37% | 1.863 | 6.155 | 71.14% | 10.07% | 5.35% |
| jump | 1420 | 38.80% | 54.65% | 85.85% | +15.85% | 1.887 | 6.346 | 75.21% | 10.77% | 4.30% |
| kick | 852 | 30.99% | 58.45% | 84.74% | +27.46% | 1.666 | 6.150 | 73.14% | 10.33% | 4.11% |
| stand up | 1420 | 45.07% | 61.83% | 81.13% | +16.76% | 1.453 | 5.779 | 69.65% | 9.15% | 4.01% |
| cartwheel | 142 | 50.70% | 64.08% | 93.66% | +13.38% | 1.583 | 6.174 | 73.09% | 10.56% | 4.23% |
| jog | 1136 | 41.20% | 68.13% | 89.79% | +26.94% | 1.389 | 5.609 | 83.39% | 8.71% | 3.70% |
| move up/down incline | 1349 | 32.10% | 68.42% | 87.03% | +36.32% | 1.091 | 3.845 | 83.79% | 5.19% | 2.37% |
| crawl | 284 | 43.31% | 71.48% | 90.85% | +28.17% | 0.983 | 3.673 | 80.37% | 4.58% | 2.11% |
| sit | 1420 | 58.87% | 80.07% | 90.85% | +21.20% | 1.058 | 3.926 | 67.94% | 6.69% | 4.01% |
| a pose | 994 | 58.75% | 82.49% | 96.48% | +23.74% | 0.928 | 3.471 | 85.29% | 5.53% | 3.02% |
| t pose | 1420 | 62.18% | 83.03% | 96.55% | +20.85% | 0.793 | 3.211 | 83.80% | 4.65% | 2.25% |

### Semantic-region detail

| Region | n | NoMove Acc | Set Acc | SafeOracle Acc | Set gain | Mean regret | P90 regret | Headroom | Set stay | Safe stay | G3 rate | Top-5% rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bedroom | 2522 | 37.55% | 61.74% | 82.12% | +24.19% | 1.539 | 5.755 | 79.48% | 28.35% | 15.82% | 9.08% | 5.11% |
| dining_area | 3492 | 39.60% | 63.66% | 88.17% | +24.05% | 1.723 | 6.428 | 73.85% | 28.64% | 11.60% | 11.05% | 5.64% |
| kitchen | 3880 | 38.58% | 61.52% | 80.80% | +22.94% | 1.554 | 5.872 | 77.29% | 31.70% | 14.30% | 9.41% | 4.82% |
| living_room | 3880 | 47.86% | 63.07% | 86.42% | +15.21% | 1.623 | 6.222 | 68.46% | 36.60% | 18.51% | 10.26% | 4.54% |

### Current-state group detail

| Group | n | Entropy mean | Margin mean | Pose confidence mean | Current correct | Move rate | Safe move rate | Safe utility mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_near_optimal | 5922 | 0.330 | 0.798 | 0.598 | 50.79% | 60.99% | 76.85% | 3.957 |
| G1_low_regret | 4408 | 0.417 | 0.743 | 0.617 | 47.87% | 71.12% | 92.08% | 3.723 |
| G2_moderate_regret | 2066 | 0.505 | 0.697 | 0.606 | 17.38% | 79.38% | 89.16% | 5.191 |
| G3_high_regret | 1378 | 0.434 | 0.735 | 0.618 | 15.02% | 74.17% | 90.35% | 6.903 |

### Geometry detail

- CandidateOracle absolute azimuth mean/median (all Episodes): 68.58° / 45.00°.
- SafeOracle move absolute azimuth mean/median: 71.60° / 45.00°; model-selected move: 81.62° / 90.00°.
- CandidateOracle geodesic mean/median: 2.725 / 2.210 m.
- SafeOracle move geodesic mean/median: 2.874 / 2.357 m; model-selected move: 3.130 / 2.879 m.
- Radius direction counts (closer/same/farther): CandidateOracle={'closer': 6954, 'same': 2853, 'farther': 3967}; SafeOracle move={'closer': 6284, 'same': 2366, 'farther': 3047}; model-selected move={'closer': 7412, 'same': 1625, 'farther': 372}.

| SafeOracle move geodesic bin | n | Mean regret | Headroom capture |
|---|---:|---:|---:|
| q0_q25 | 2782 | 1.678 | 52.33% |
| q25_q50 | 3056 | 1.740 | 71.82% |
| q50_q75 | 2931 | 1.450 | 80.12% |
| q75_q100 | 2928 | 1.957 | 79.45% |

### Candidate-set difficulty

| Gap bin | n | Exact hit | Mean regret | P90 regret | Headroom |
|---|---:|---:|---:|---:|---:|
| very_small | 3444 | 19.51% | 0.495 | 0.841 | 89.72% |
| small | 3443 | 26.26% | 1.157 | 4.437 | 81.86% |
| medium | 3443 | 32.79% | 2.193 | 7.119 | 71.65% |
| large | 3444 | 57.17% | 2.610 | 7.870 | 64.91% |

### Representative cases

- `correct_stay`: `babel_val_00273_000__00006-HkseAnWCgqk__bedroom__v07`, action=kick, region=bedroom, regret=0.0000, predicted=stay, SafeOracle=stay.
- `correct_move`: `babel_val_00674_001__00006-HkseAnWCgqk__bedroom__v26`, action=t pose, region=bedroom, regret=0.0000, predicted=candidate:7, SafeOracle=candidate:7.
- `candidate_miss_near_optimal`: `babel_val_08843_036__00643-ggNAcMh8JPT__dining_area__v06`, action=stand up, region=dining_area, regret=0.0100, predicted=stay, SafeOracle=candidate:4.
- `missed_move`: `babel_val_08758_002__00164-XfUxBGTFQQb__living_room__v20`, action=move up/down incline, region=living_room, regret=23.8803, predicted=stay, SafeOracle=candidate:11.
- `unnecessary_move`: `babel_val_07900_000__00164-XfUxBGTFQQb__bedroom__v19`, action=play instrument, region=bedroom, regret=17.6753, predicted=candidate:3, SafeOracle=stay.
- `wrong_candidate_high_regret`: `babel_val_06970_000__00087-YY8rqV6L6rf__living_room__v08`, action=play instrument, region=living_room, regret=19.5209, predicted=candidate:31, SafeOracle=candidate:23.
- `catastrophic_top1pct`: `babel_val_08758_002__00164-XfUxBGTFQQb__living_room__v20`, action=move up/down incline, region=living_room, regret=23.8803, predicted=stay, SafeOracle=candidate:11.

## Auxiliary Pairwise reference

- pairwise_mlp: Test StageC 61.45% Accuracy / 55.33% Macro-F1; regret mean 1.8018; aggregate headroom 70.83%.
- set_ranker: Test StageC 62.54% Accuracy / 56.37% Macro-F1; regret mean 1.6137; aggregate headroom 74.93%.
