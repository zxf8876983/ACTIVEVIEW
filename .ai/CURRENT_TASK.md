# Current Task

## Reduced14 per-class recognizability audit — completed 2026-09-08

Implemented and ran a Val-only audit for the reduced14 eight-placement
protocol. The frozen reduced14 ST-GCN was evaluated on the independent 270
sample development-Val tensor with CUDA RTX 4090. ActiveView metrics used the
existing 19,440 Val utility contexts and 144,720 legal candidate pairs; the
existing moving-Val counterfactual cache was checked for ID/label alignment.

The report is stored in
`experiments/reduced14_eight_placement_v1/per_class_recognizability_audit/`.
The audit reports all 14 class-level clean recognizability, s0/legal/oracle
viewpoint metrics, transparent A/B/C diagnostic groups and the requested
special-action checks. Potential removal candidates are diagnostic only; no
taxonomy change was made. No model was trained, no data was regenerated, and
Test was not read.

The task is complete; do not start another experiment automatically.
