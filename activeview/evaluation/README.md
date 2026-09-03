# Evaluation layer

`evaluation/` is the shared, protocol-level evaluation implementation.

- `metrics.py` provides pure classification, regret and movement summaries;
- `evaluator.py` builds trajectories, fixed-first oracles and method summaries.

The command-line wrappers under `activeview/scripts/eval/` select a frozen
split and write experiment records without changing model or data artifacts.
