# Stage C-v0 failure-analysis snapshot

This directory is a review snapshot generated from the frozen ACTIVEVIEW v11.5
runtime artifacts. It is not an additional training or evaluation dataset.

Source artifacts were read from the configured runtime root
(`/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/`): accepted
Stage A Episodes, Stage B utility labels, Stage C feature JSONL, Set Ranker and
Pairwise prediction JSONL, evaluation summaries, and the passed Stage C
validator report. The analysis covered 13,774 Test Episodes and 194 independent
motion records. No Habitat, YOLO, VideoPose3D, retraining, or upstream artifact
generation was run.

## Files

- `stage_c_failure_report.md`: human-readable scientific report.
- `record_failure_table.csv`: one row per independent motion record.
- `figures/`: five diagnostic plots used by the report.

The complete machine-readable summary and per-Episode table are retained in
the configured runtime directory `datasets/policy_v11_5/stage_c/failure_analysis/`
and are intentionally not tracked in Git.

The reproducible entry point is
`activeview/scripts/analyze_stage_c_failures.py`. Its default output is the
runtime directory `stage_c/failure_analysis/`; this directory contains the
versioned review copy only. Thresholds such as p75/p90/p95/p99 regret are
computed from the frozen Test distribution at runtime and are recorded in the
JSON summary.

The analysis is descriptive and uses repeated Episode observations carefully:
record-level aggregation is reported separately, and no significance claim is
made from treating 13,774 Episodes as IID samples. Stage D and any Stage C-v1
model decision remain out of scope.
