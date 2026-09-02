# Current Task

## RGB Observation Dataset V1 completed

RGB Observation V1 has been generated and fully audited for the canonical
HM3D-train offline skeleton dataset. The source skeleton root remained
read-only. The separate MG08 RGB root contains 21 scenes, 82,320 records and
2,634,240 RGB views (`uint8`, `[32,256,256,3]`), all using fixed frame index
15. Completeness audit: missing 0, extra 0, invalid 0.

Generation used 16 Habitat workers per scene after explicit user
authorization. No YOLO, VideoPose3D, ST-GCN, skeleton regeneration or policy
experiment was run. The external output is documented in
`docs/results/rgb_observation_v1.md` and summarized by
`docs/results/rgb_observation_v1_summary.json`.

## EXP024 — DINOv2 RGB Context Utility Regression Pilot

EXP024 has completed its authorized Train→Val run. It used only the existing
MG08 RGB Observation V1 records for visited Stage-D `s0`/`s1` viewpoints,
with frozen DINOv2 ViT-B/14 global CLS embeddings and a trainable RGB
projector/regression head. The deduplicated cache contains 58,266 Train and
19,484 Val observations (77,750 union), with no future-candidate RGB access.

Val results: EXP024 Accuracy 0.657682, Macro-F1 0.609503, mean regret
1.414353, P90 regret 5.431195 and headroom 0.782434. Relative to frozen
EXP022 (0.659898 Accuracy / 1.416495 mean regret), EXP024 reduced mean regret
by 0.002142 but reduced Accuracy by 0.002216; decision is INCONCLUSIVE as a
global-CLS RGB pilot. Pearson utility correlation was 0.428491 and Spearman
0.252335. Candidate identity mismatch was zero.

Runtime cache and experiment outputs remain external under
`/home/zxf/MG08/robot/ActiveView/`; Test was not read. No Stage A/B/C artifact,
RGB dataset, perception pipeline or Habitat rendering was modified.

## EXP025 — DINOv2 Spatial RGB Utility Regression

EXP025 completed its fixed Train-to-Val run using only visited `s0`/`s1` RGB
observations. The 77,750-observation spatial patch cache is external under
`/home/zxf/MG08/robot/ActiveView/`; no p2/p3 RGB was accessed. Val Accuracy was
0.659898, Macro-F1 0.607331, mean regret 1.378650, P90 5.330633 and headroom
0.784692. Relative to EXP024, mean regret improved by 0.0357029 but Macro-F1
declined by 0.002172; decision INCONCLUSIVE. Candidate mismatch was zero.

## EXP026 — Spatial RGB-D Utility Regression

EXP026 completed its fixed Train-to-Val run. Metric depth was rendered only
for the same 77,750 visited `s0`/`s1` observations at frame 15, with 16
workers and four-camera batching; compact cache is external under
`/home/zxf/MG08/robot/ActiveView/`. Val Accuracy was 0.657325, Macro-F1
0.606847, mean regret 1.407385, P90 5.423377 and headroom 0.783227.
Relative to EXP025, mean regret worsened by 0.0287345 and Accuracy declined
by 0.002574; decision INCONCLUSIVE. Candidate mismatch was zero.

## EXP027 — Spatial-RGB Oracle Behavior Cloning

EXP027 completed its authorized fixed 30-epoch Train-only three-action
behavior-cloning run and one Val evaluation. It used only visited s0/s1
EXP025 spatial RGB cache and legal Stage-D skeleton/delta/candidate geometry;
the frozen Stage C-v0 first action and p1 remained unchanged. Train oracle
labels exactly matched the frozen Fixed-first Second-Step Oracle.

Val EXP027 BC: Accuracy 0.657039, Macro-F1 0.608384, mean regret 1.486694,
P90 regret 5.751852 and headroom 0.769306. It underperformed EXP014,
EXP023 and EXP025; decision is INCONCLUSIVE as a diagnostic negative result.
Three-way imitation accuracy was 0.493841 and binary Move/Stay accuracy was
0.644837. Harmful moves numbered 2,442 and missed beneficial oracle moves
1,803. Test remained locked and no upstream or perception artifact changed.

## EXP028 — Oracle Action Predictability / Representation Sufficiency Audit

EXP028 completed its authorized Val-only diagnostic over the frozen Stage-D
eligible population (29,133 Train / 9,742 Val). The observable vector used
Train-only normalization and a Train-only cosine nearest-neighbor index over
normalized current observations, legal candidate geometry and visited s0/s1
spatial RGB. No policy was trained and no Test was read.

Val NN 3-way agreement was 0.454527 (1-NN), 0.449497 (5-NN), 0.451139
(10-NN), and 0.444570 (25-NN); 25-NN mean three-way entropy was 0.870910.
EXP027 three-way imitation accuracy increased from 0.4730 in the smallest
margin bin to 0.5623 for margin >=2.0, while the >=1.0 subset reached
0.5376. Cross-motion oracle action switch rate for matching quantized
scene/region/geometry contexts was 0.551969 on Val. The audit is
**INCONCLUSIVE** pending human scientific interpretation (Case A/B/C); it
does not establish intrinsic unpredictability. Full summary is in
`experiments/stage_d/EXP028_oracle_action_predictability/`.

Test remains locked. Do not start EXP029 automatically.

## EXP041–EXP044 candidate-observation world-model campaign

The EXP014 evaluator gate was rechecked successfully (Accuracy 0.6582540931,
Macro-F1 0.6101526052).  A lazy candidate-conditioned perceived-skeleton
world-model module and Train/Val-only campaign entry point were added together
with EXP041–EXP044 experiment records and focused tests.  The source/target
identity audit passes for 29,133 Train and 9,742 Val contexts (32 unique
viewpoints per archive, missing/duplicate/mismatch counts all zero).  The
external conda `habitat` now exposes an RTX 4090 (CUDA 12.4); fixed EXP042
A/B/C training and EXP043 frozen-ST-GCN Val diagnostics completed. EXP044 is
explicitly skipped because a validated recurrent multi-step trajectory
evaluator/history-rollout entry point is not implemented. Test, Habitat
rendering, perception and ST-GCN retraining were not performed.
