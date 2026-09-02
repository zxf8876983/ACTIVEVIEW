# EXP051-R1 — Genuine Closed-Loop Counterfactual Revision

This preregistered run is blocked at the mandatory frozen-model integrity gate.
The EXP050 Joint Revision checkpoint is not present in the repository or in the
configured runtime artifact roots.  EXP051-R1 must not retrain that model, so
history-shift and H=2 rollout phases were not executed.

The existing EXP051 blocked record is preserved unchanged.  Once the exact
EXP050 checkpoint and its configuration are restored, rerun `run.sh`; no RGB,
Habitat, Test, or perception substitute is accepted.
