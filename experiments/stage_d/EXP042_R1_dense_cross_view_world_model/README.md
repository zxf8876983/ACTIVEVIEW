# EXP042-R1 — Dense Cross-View World Model

Train/Val-only rerun of candidate-observation world models with all 32
viewpoints as targets. Metrics use corrected temporal axis (axis 1) and frozen
H36M-17 bone edges. No Test, perception regeneration, or GT pose input is used.

Variants A–D differ by belief/RGB/residual context; E/F add a fixed 0.10 KL loss
to a frozen ST-GCN teacher (recorded under EXP045).
