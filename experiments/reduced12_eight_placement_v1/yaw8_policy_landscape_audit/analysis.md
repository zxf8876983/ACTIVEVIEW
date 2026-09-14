# Yaw8 Recognizer × Policy Landscape Audit

Policy Train and Moving Val only; Policy Test was not opened.

## Main Moving-Val table

| Metric | Old ST-GCN | Yaw8 ST-GCN | Change |
|---|---:|---:|---:|
| Stay/current Acc | 0.302579 | 0.316865 | +1.429pp |
| Stay/current Macro-F1 | 0.292976 | 0.330391 | +3.741pp |
| Random legal Acc | 0.365079 | 0.383433 | +1.835pp |
| Random legal Macro-F1 | 0.364567 | 0.410639 | +4.607pp |
| StaticPrior Acc | 0.491567 | 0.530556 | +3.899pp |
| StaticPrior Macro-F1 | 0.491338 | 0.545988 | +5.465pp |
| GT-TrueLogP Oracle Acc | 0.753175 | 0.773611 | +2.044pp |
| GT-TrueLogP Oracle Macro-F1 | 0.755358 | 0.790086 | +3.473pp |
| GT-Margin Oracle Acc | 0.791964 | 0.777976 | -1.399pp |
| GT-Margin Oracle Macro-F1 | 0.800900 | 0.794831 | -0.607pp |
| AnyCorrect Coverage | 0.791964 | 0.777976 | -1.399pp |

Random→Oracle gap: old=0.388095, yaw8=0.390179; Static→Oracle gap: old=0.261607, yaw8=0.243056.
Relative-yaw best–worst accuracy gap: old=0.062544, yaw8=0.099997, reduction=-0.037453.
Viewpoint accuracy std: old=0.111740, yaw8=0.132129; eta² absolute-view=0.052660/0.072502, relative-yaw=0.001683/0.004539.
Global candidate GT-margin Spearman=0.557790; within-context mean/median=0.380182/0.500000.
Oracle GT-margin top-1 agreement between recognizers=0.380556; Train prior Spearman=0.974707, top-5 overlap=1.000000.

## Interpretation

**YAW ROBUSTNESS IMPROVED, BUT SUBSTANTIAL OCCLUSION-DRIVEN NBV HEADROOM REMAINS**.
Legal candidate accuracy change is +1.835pp; relative-yaw gap flattened=False; per-class drops larger than 10pp=['sit', 'bend'].

The old and Yaw8 candidates use the identical current/Stay + Stage-A legal action set. The Yaw8 checkpoint was frozen before Policy Val analysis; StaticPrior is computed independently from Policy Train candidate GT-Margins for each recognizer.

Promotion decision: **DO NOT PROMOTE YAW8 RECOGNIZER**. This audit does not train a selector. If promoted, the next required step is to rebuild Policy Train utility targets with frozen Yaw8 and retrain strict Frame0 NBV from scratch.

## Required scientific answers

1. Candidate recognition: legal-candidate micro Acc changes from 0.365079 to 0.383433; StaticPrior changes from 0.491567 to 0.530556.
2. Old viewpoint variation: absolute-view eta²=0.052660; relative-body-yaw eta²=0.001683.
3. Yaw8 relative-angle flattening: no (best–worst gap 0.099997).
4. Static prior remains strong; own-prior accuracy is 0.530556.
5. Static-prior ranking correlation is Spearman 0.974707, top-5 overlap 1.000000.
6. GT-TrueLogP Oracle / AnyCorrect changes 0.753175/0.791964 → 0.773611/0.777976.
7. Random→Oracle headroom changes 0.388095 → 0.390179; Static→Oracle changes 0.261607 → 0.243056.
8. High-occlusion Random/Static/Oracle Acc changes 0.269337/0.433507/0.638031 → 0.273617/0.453072/0.671354.
9. The old candidate utility landscape is substantially changed at the context level (GT-margin top-1 agreement and top-3 overlap are reported above), while the static viewpoint prior remains structurally similar.
10. A real active-view problem remains because Yaw8 Static/Random are well below its legal oracle; however, the strict Yaw8 promotion gate is not met because relative-yaw sensitivity increased and sit/bend show large random-candidate drops.

Flags: `policy_test_used=false`, `new_rgb_generated=false`, `new_skeleton_generated=false`, `yolo_rerun=false`, `videopose3d_rerun=false`, `stgcn_modified=false`, `moving_val_used_for_training=false`.
