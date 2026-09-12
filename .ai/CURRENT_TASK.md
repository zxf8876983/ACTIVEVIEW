# Prefix-5 task-utility selector — completed 2026-09-13

Implemented and ran the strict reduced12 Prefix-5 task-utility selector on
Policy Train and Moving Val. The action set is current/Stay plus the Stage-A
legal candidate pool. Stay uses current[0:30]; a candidate uses
current[0:5] + candidate[5:30]. This is a discrete-time view-switch
approximation. The frozen reduced12 ST-GCN and frozen shared head were not
modified, no perception artifacts were regenerated, and Policy Test was not
read.

Three small causal branches were trained for 12 epochs with 313-record,
16-context-per-record sampling and selected by Moving-Val mixed5 utility
loss: GeometryOnly-Mixed5, Prefix5+Geometry, and
Prefix5+RGBGlobal+Geometry. Moving-Val results (Accuracy/Macro-F1) were
0.422817/0.406814, 0.429266/0.411469, and 0.441667/0.424037 respectively.
Stay and Random-ShortPrefix5 were 0.302579/0.292976 and
0.342262/0.329111; the strict Mixed5 GT-TrueLogP oracle was
0.687599/0.671197 and GT-Margin/AnyCorrect was 0.720040/0.709917.

The best branch is below the pre-registered 0.52 kill threshold. Prefix
shuffle reduced Accuracy by 0.337pp; RGB shuffle reduced it by 2.163pp.
The route is therefore a negative result for this simple deployable selector;
no follow-up experiment was started automatically.

Report:
`experiments/reduced12_eight_placement_v1/prefix5_task_utility_predictor_v1/`

Task status: **CLEAN**. Runtime checkpoints remain outside Git under the
configured ActiveView data root.
