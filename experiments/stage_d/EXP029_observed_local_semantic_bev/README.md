# EXP029 — Observed Local Semantic BEV Sufficiency Audit

Val-only representation audit. The fixed first-step protocol and all Stage-D
semantics remain frozen. Only semantic/depth observations from visited `s0`
and `s1` are rendered; `p2`/`p3`, Test and unobserved scene maps are never
accessed. The representation appends a deterministic 15-channel `[15,80,80]`
BEV (10×10 pooling) to the legal EXP028 compact representation.

The Habitat smoke gate is executed before the 4-worker Train/Val cache build
(the host cannot safely keep 16 annotated Habitat simulators resident).
Large BEV caches and checkpoints live under `ACTIVEVIEW_DATA_ROOT` and are not
tracked by Git.
