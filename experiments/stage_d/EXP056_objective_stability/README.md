# EXP056 — Paired 3-Seed Objective Stability

This experiment compares the frozen `_JointRevision` architecture with the
original first-correct single-action CE objective and EXP055's multi-positive
set objective.  Seeds are fixed to 42, 43 and 44; each objective is trained for
20 Train epochs and evaluated once with the real-observation H2 closed-loop
evaluator.  No Test split, perception regeneration, Habitat rendering, or
additional model variant is used.

Run `run.sh` only after the Conda `habitat` CUDA preflight. Checkpoints are
written under the external runtime data root; compact metrics and manifests are
tracked in this directory.
