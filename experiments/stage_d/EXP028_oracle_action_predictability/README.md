# EXP028 — Oracle Action Predictability / Representation Sufficiency Audit

This is a Val-only, analysis-only audit. It uses frozen Stage-D eligible
episodes and EXP027 predictions to measure whether the Fixed-first Oracle
actions are locally predictable from legal current observations, candidate
geometry and visited s0/s1 spatial RGB embeddings. Train vectors alone form the
cosine nearest-neighbor index; Train statistics are applied to Val. No policy
is trained, no Test rows are read, and future-candidate RGB/depth/skeleton are
never accessed.

Fixed oracle actions are `argmax([0, U2(p2), U2(p3)])` with the frozen cache
order tie rule. Margin bins and context quantization are fixed in source:
`[0,.05), [.05,.1), [.1,.25), [.25,.5), [.5,1), [1,2), [2,+inf)` and
`round(geometry, 3)`, respectively.

Run with `bash run.sh` from the repository root. Runtime inputs and RGB cache
remain external under the configured data roots.
