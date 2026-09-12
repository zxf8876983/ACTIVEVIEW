# Frame-0 Task-Utility Predictor — completed 2026-09-13

Implemented and ran the causal Frame-0 Task-Utility Predictor on Policy Train
(46,324 contexts, 313 records) with Moving Val model selection/evaluation
(10,080 contexts, 105 records). Policy Test was not read. Predictor inputs are
strictly current frame-0 DINOv2 global tokens plus Stage-A legal candidate
geometry; no future observation, skeleton, recognizer output or GT action is
provided at inference.

The terminal recognizer is the frozen reduced12 ST-GCN encoder plus the frozen
shared adapted head. Existing frame-0 RGB/DINO and recognizer option caches
were reused; no new RGB, skeleton or DINO data was generated. Runtime
checkpoints remain outside Git.

Moving-Val Accuracy/Macro-F1:

- Stay: 0.302579 / 0.292976
- Random legal: 0.365079 / 0.364567
- RGBGlobal Visibility (fixed prior): 0.494841 / 0.493881
- GeometryOnly-TrueLogP: 0.479762 / 0.479889
- RGBGlobal-TrueLogP: 0.498214 / 0.495003
- RGBGlobal-Margin: 0.499306 / 0.499761
- RGBGlobal-TrueLogP + VisibilityAux: 0.506151 / 0.503722
- Historical Route-1 + Shared: 0.509524 / 0.509806
- GT-TrueLogP Oracle: 0.753175 / 0.755358

The best new branch is RGBGlobal-TrueLogP + VisibilityAux. Its gain over the
fixed visibility predictor is +1.131pp Accuracy, while RGBGlobal-TrueLogP is
only +1.845pp over GeometryOnly-TrueLogP and the auxiliary head adds +0.794pp.
The preregistered decision is **WEAK KEEP**: retain as a baseline without
adding complexity. The RGB-shuffle audit drops Accuracy from 0.498214 to
0.436111, indicating real frame-0 visual contribution. High-occlusion and
transition details are in the experiment JSON reports.

Experiment report:
`experiments/reduced12_eight_placement_v1/frame0_task_utility_predictor_v1/`

Task status: **CLEAN**. Next human decision: whether to move to the approved
short-prefix active-recognition direction; no follow-up experiment was
started automatically.
