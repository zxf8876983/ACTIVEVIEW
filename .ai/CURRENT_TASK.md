# Frame-0 Alternative-View Visibility Predictor — completed 2026-09-12

Implemented and ran the first deployable, action-agnostic frame-0 visibility
predictor using Policy Train (46,324 contexts) and Moving Val (10,080
contexts). The predictor consumes only a strict current-view frame-0 RGB DINO
representation and Stage-A legal candidate geometry; scene-only Habitat
frame-0 H36M17 raycasts are supervision targets. Policy Test was not read.

The isolated runtime cache contains 56,404 current-view frame-0 RGB archives
and 16x768 DINO spatial tokens for Train/Val only. No candidate RGB/DINO,
future skeleton, action label, or recognizer output is a predictor input.
Target caches and checkpoints remain outside Git under the configured data
root.

Moving-Val results (Accuracy / Macro-F1):

- Stay: 0.302579 / 0.292976
- Random legal: 0.365079 / 0.364567
- GeometryOnly: 0.477579 / 0.479837
- RGBGlobal+Geometry: 0.494841 / 0.493881
- RGBSpatial+Geometry: 0.493849 / 0.494691
- Frame0SceneVisibility Oracle: 0.489782 / 0.485962
- Historical Route-1 + Shared: 0.509524 / 0.509806
- GT-TrueLogP Oracle: 0.753175 / 0.755358

RGBGlobal+Geometry is the best deployable branch by Accuracy and recovers
104.1% of the Random-to-frame-0-oracle Accuracy gain; RGBSpatial has the best
visibility MAE/ranking correlations but slightly lower downstream Accuracy.
The pre-registered decision is **STRONG KEEP** for the visibility-prediction
route. High-occlusion and transition details are in the experiment report.

Experiment report:
`experiments/reduced12_eight_placement_v1/frame0_visibility_predictor_v1/`

Task status: **CLEAN**. Next human decision: whether to integrate the best
frame-0 predictor into a frozen downstream policy experiment.
