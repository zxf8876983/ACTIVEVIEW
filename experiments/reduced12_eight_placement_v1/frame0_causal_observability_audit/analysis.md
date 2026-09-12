# Frame-0 Causal Observability Audit

## Protocol

- Moving Val contexts = 10,080
- Policy Test used = false
- training = none
- recognizer = frozen reduced12 ST-GCN feature cache + shared adapted head
- decision-time human pose = frame 0 only for Frame0SceneVisibility
- future motion frames used by Frame0 selector = false
- GT action used by selector = false
- candidate recognizer evidence used by selector = false
- action set = current/Stay + Stage-A legal candidates
- raycast = scene-only Habitat HM3D geometry to reconstructed world-space H36M17 joints
- tie rule = exact Stay tie wins; moving ties use smallest candidate viewpoint id

## Matched results

| Selector | Type | Accuracy | Macro-F1 | Move rate |
|---|---|---:|---:|---:|
| Stay | baseline | 0.302579 | 0.292976 | 0.000000 |
| Random legal | baseline | 0.365079 | 0.364567 | 0.843254 |
| RealPoseConfidence | non-causal reference | 0.513194 | 0.513235 | 0.871230 |
| Frame0SceneVisibility | causal privileged | 0.489782 | 0.485962 | 0.523611 |
| FullTemporalSceneVisibility | future-aware reference | 0.495833 | 0.492420 | 0.610020 |
| Historical Route-1 | learned task-aware | 0.509524 | 0.509806 | 0.972024 |
| GT-TrueLogP Oracle | privileged task-aware | 0.753175 | 0.755358 | 0.886409 |
| CandidateOnly AnyCorrect | coverage only | — | — | — |
| StayPlusCandidate AnyCorrect | coverage only | — | — | — |

Frame0SceneVisibility vs Random: +12.470pp Accuracy / +12.140pp Macro-F1.
Frame0SceneVisibility vs Historical Route-1 + Shared: -1.974pp Accuracy.
FullTemporalSceneVisibility - Frame0SceneVisibility: +0.605pp Accuracy / +0.646pp Macro-F1.

## Frame-0 pose confidence

Frame0PoseConfidence unavailable: archives store only a 30-frame aggregated viewpoint scalar, not per-frame keypoint confidence. No frame-0 value was fabricated.

## Occlusion stratification

Initial occlusion is defined by Stay frame-0 visibility (low occlusion = top tertile visibility; high occlusion = bottom tertile). See `occlusion_stratified_metrics.json` for Stay, Random, Frame0, FullTemporal and Historical results.

## Ranking and transition diagnostics

Frame0 vs FullTemporal selected-view agreement: 0.881448; top-3 overlap: 0.992460; Frame0 selected rank within FullTemporal mean/median: 1.374/1.000.
Frame0 vs FullTemporal candidate Spearman: 0.959321; within-context mean/median: 0.748594/0.983516.
Frame0 vs shared GT-TrueLogP candidate Spearman: 0.417600; within-context mean/median: 0.253482/0.272059; top1/top3 overlap: 0.260317/0.659325.

## Judgment

Frame-0 scene visibility reaches the strong causal threshold (at least 48% and +8pp over Random); keep a strict causal observability route and prioritize predicting scalar frame-0 quality before adding structured visibility.
Final decision: **KEEP strict causal observability**.
Observation quality is not recognition utility; the FullTemporal and GT-TrueLogP rows are privileged references, not deployable selectors.
No Policy Test was read, no model was trained, and no RGB/skeleton/DINO/runtime data were regenerated or modified.
