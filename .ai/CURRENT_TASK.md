# Structured observability privileged audit — completed 2026-09-13

Implemented and ran the final Train/Val-only reduced12 frame-0 structured
observability audit. The exact action set was current/Stay plus the Stage-A
legal candidate pool over 46,324 Policy Train contexts and 10,080 Moving-Val
contexts. Frozen reduced12 ST-GCN/shared head were unchanged; Policy Test and
new perception artifacts were not read or generated.

The audit regenerated exact frame-0, 17-joint, scene-only Habitat raycast
visibility caches (Train/Val) and verified that their per-candidate mean is
bit-identical to the existing scalar frame-0 cache (max absolute difference
0). Five 12-epoch record-balanced MLP branches were trained with 313 records
× 16 contexts/epoch, seed 42, AdamW, and SmoothL1 + 0.5 listwise utility
loss.

Moving-Val Accuracy/Macro-F1:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| ScalarVisibility+Geometry | 0.520139 | 0.520467 |
| StructuredVisibility17+Geometry | 0.512401 | 0.510476 |
| CurrentPose+Geometry | 0.500000 | 0.499949 |
| CurrentPose+StructuredVisibility17+Geometry | 0.522421 | 0.520503 |
| GTAction+CurrentPose+StructuredVisibility17+Geometry | 0.520833 | 0.518689 |
| GT-TrueLogP Oracle | 0.753175 | 0.755358 |

Structured visibility alone is -0.774pp versus the scalar baseline; adding
current frame-0 pose recovers only +1.002pp, and the GT-action branch adds no
further gain. The residual Oracle→GT-action branch gap is 23.234pp Accuracy,
which triggers the preregistered kill rule. The conclusion is to stop the
pre-action structured observability selector family and redirect effort to
future-recognizer evidence / sequential information acquisition rather than
more visibility predictors.

Report and implementation:
`experiments/reduced12_eight_placement_v1/structured_observability_final_audit/`
and
`activeview/scripts/experiments/run_reduced12_structured_observability_final_audit.py`.

Task status: **CLEAN**. No follow-up method was started automatically.
