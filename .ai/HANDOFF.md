# Handoff

Status: CLEAN

EXP025, EXP026, EXP027 and EXP028 are complete. Runtime caches, checkpoints and predictions
are under `/home/zxf/MG08/robot/ActiveView/`; source summaries are under
`experiments/stage_d/EXP025_dinov2_spatial_rgb/` and
`experiments/stage_d/EXP026_spatial_rgbd_utility_regression/`.

EXP025 used 77,750 unique visited `s0`/`s1` RGB observations (58,266 Train;
19,484 Val) with frozen DINOv2 spatial tokens. Val: Accuracy 0.659898,
Macro-F1 0.607331, mean regret 1.378650, P90 5.330633, headroom 0.784692.
EXP026 used the same keys with frame-15 Habitat metric depth and 16 workers.
Val: Accuracy 0.657325, Macro-F1 0.606847, mean regret 1.407385, P90
5.423377, headroom 0.783227. Candidate identity mismatches were zero in both.

EXP027 used only visited s0/s1 spatial RGB and legal Stage-D features with
frozen Stage C-v0 first-step behavior. Val Accuracy 0.657039, Macro-F1
0.608384, mean regret 1.486694 and P90 5.751852; decision INCONCLUSIVE.
Test remained locked. No Stage A/B/C artifact, skeleton/RGB dataset,
perception pipeline or ST-GCN artifact was modified.

EXP028 was a Val-only oracle predictability audit using a Train-only cosine
nearest-neighbor index over normalized current observations, legal candidate
geometry and visited s0/s1 spatial RGB. It found low 25-NN agreement (0.444570),
high mean three-way neighborhood entropy (0.870910), and Val cross-motion
oracle-action switching (0.551969) for quantized context groups. Decision is
INCONCLUSIVE pending human review; no policy was trained and no Test was read.
See `experiments/stage_d/EXP028_oracle_action_predictability/`.
Do not start EXP029 automatically.
