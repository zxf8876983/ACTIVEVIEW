# Known-Map + Current-Depth Movement-Aware NBV

Train record-holdout lambda selection and Moving Val evaluation only (10,080 contexts). Policy Test was not read.
Current depth is rendered only at Frame 0/current viewpoint. Raw depth and point clouds are transient; compact path/risk summaries are cached outside Git.

| Method | Acc | Macro-F1 | Mean path (m) | Median path (m) | Human clearance (m) | Depth-risk rate |
|---|---:|---:|---:|---:|---:|---:|
| StaticPrior | 0.549901 | 0.571990 | 2.8552 | 2.3554 | 1.2764 | 0.6099 |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 2.7876 | 2.2245 | 1.3671 | 0.5874 |
| RGBGlobal-Visibility + MapPath | 0.564881 | 0.581486 | 2.6633 | 2.1775 | 1.4389 | 0.5622 |
| RGBGlobal-Visibility + MapPath + CurrentDepth | 0.563690 | 0.580532 | 2.5673 | 1.7881 | 1.5130 | 0.5317 |
| GT SceneVisibility | 0.583929 | 0.598955 | 2.9047 | 2.4531 | 1.2248 | 0.6117 |
| GT SceneVisibility + MapPath | 0.570139 | 0.586700 | 2.0268 | 1.4168 | 1.7856 | 0.4643 |
| GT SceneVisibility + MapPath + Depth | 0.562599 | 0.580631 | 2.0106 | 1.4168 | 1.8368 | 0.4096 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | 2.9809 | 2.4531 | 1.4447 | 0.5723 |

## Decision summary
- Stage-A RGB→MapPath selected λ=0.10; mean path change=+4.46% and Accuracy change=-0.248pp. **PATH-COST NBV NOT USEFUL**.
- MapPath→MapPath+Depth switched 5.34% of selections; human-clearance relative change=+5.15%; depth-risk rate changed from 0.5622 to 0.5317. **MIXED DEPTH EVIDENCE**.
- Depth occupancy: 0.1625 of non-human obstacle points were in navmesh/free-space proxy cells; 0.1118 of obstacle points were attributable to the YOLO-bbox human footprint.
- High-occlusion subset contains 3271 contexts; details are in high_occlusion_metrics.json.

## Required scientific answers
1. Known-map path cost is useful only if it meets the preregistered distance/accuracy gate; the exact deltas are reported above.
2. Current depth changes decisions only through the compact transient occupancy and dynamic human footprint; disagreement and risk deltas are reported above.
3. Depth provides no candidate RGB/depth or future perception. A high human-attributable fraction means the static HM3D benchmark contains little non-human transient geometry.
4. High-occlusion movement results are kept separate and are not used to retune thresholds.
5. This remains an oracle/diagnostic audit for known Habitat geometry; no new model, map, skeleton, RGB or DINO was trained/generated.

```text
policy_test_used=false
training_used=false (Train is used only for prior and lambda holdout selection)
habitat_gt_scene_geometry_used=true
current_frame_depth_used_for_local_occupancy=true
future_candidate_rgb_used=false
future_candidate_skeleton_used_only_for_terminal_evaluation=true
deployable_rgb_selector=true; depth_context_is_current_only=true
```
