# Stage C-v1 Experiments

Each `EXPxxx` tests one primary scientific change. Keep a small, readable
record in Git:

- `README.md` — question, hypothesis, frozen components and interpretation;
- `config.yaml` — parameters that affect execution;
- `run.sh` — exact Train→Val command;
- `baseline.json` — compact baseline when needed;
- `result.json` and `analysis.md` — written after the experiment runs.

Large checkpoints, predictions and logs belong under `ACTIVEVIEW_DATA_ROOT`.
During development use Train and Val only. Run Test only after the final
method has been explicitly selected. Negative results remain recorded.
