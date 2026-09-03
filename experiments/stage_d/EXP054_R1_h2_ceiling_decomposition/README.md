# EXP054-R1 — True-Future Matched-JR Geometry Correction

This rerun changes only the Train matched-Joint-Revision geometry: after the
frozen first-step move to `v1`, Train rebuilds the legal candidate order and
candidate descriptors centered at `v1`, matching the Val protocol. Candidate
recognition remains archived true ST-GCN `true_logp`; the first action, model
architecture, target and optimizer are unchanged. Test, H3, perception and
Habitat rendering are not used.
