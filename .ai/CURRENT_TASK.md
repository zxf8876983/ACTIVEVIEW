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

## Status

EXP024 is complete and ready for human scientific review. Do not start
EXP025 or add spatial RGB features automatically.
