# True-facing ST-GCN angle-prior re-audit — completed

Implemented and ran
`activeview/scripts/eval/analyze_reduced12_true_facing_stgcn_angle_reaudit.py`
with CUDA using the frozen Yaw8 ST-GCN encoder, Yaw8Fair shared head, existing
candidate cache, scene metadata, visibility cache and current-frame DINO cache.
No model was trained or modified, no RGB/skeleton was generated, and Policy
Test was not read.

The AMASS frame-0 body-facing convention is
`R_body_world = R_scene_yaw @ R_AMASS_root_frame0`, with local +Z as forward.
The old placement-yaw bin changed for 1,440/1,960 internal observations
(0.734694) and for 57,609/68,702 Moving legal candidates (0.838535).

Corrected internal angle accuracy is highest at 225° (0.706122) and lowest at
270° (0.653061). Train→Moving Spearman is 0.238095 for Q_acc versus angle
accuracy and 0.547619 for Q_margin versus mean margin, compared with historical
0.142857 and -0.333333. Moving selectors were TrueFacing-Acc 0.450099/0.477597
and TrueFacing-Margin 0.451587/0.483006 Accuracy/Macro-F1, below StaticPrior
0.549901/0.571990 and visibility references. The registered decision is
**KILL ST-GCN RELATIVE-ANGLE PRIOR**; the next diagnostic should be Clean-to-
Observed ST-GCN feature degradation rather than another angle prior.

The new reports are under
`experiments/reduced12_eight_placement_v1/true_facing_stgcn_angle_reaudit/`.
