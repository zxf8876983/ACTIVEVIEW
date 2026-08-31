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

## Status

EXP025 and EXP026 are complete and ready for human scientific review. Test
remained locked; no Stage A/B/C artifact, skeleton/RGB dataset, perception
pipeline or ST-GCN artifact was modified. Do not start EXP027 automatically.
