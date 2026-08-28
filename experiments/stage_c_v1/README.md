# Stage C-v1 Controlled Experiments

This directory contains immutable, human-reviewed experiment source records.
`EXP001_gap_aware_ranking` is the first controlled record and is currently
`PLANNED`; creating the record does not authorize starting or running it.
Every later experiment requires its own explicit human authorization.

## Standard lifecycle

```bash
# 1. Create a PLANNED record
python -m activeview.scripts.create_experiment \
  --stage stage_c_v1 \
  --name <name> \
  --hypothesis "<one falsifiable hypothesis>" \
  --core-change "<one approved core change>"

# 2. Human review of hypothesis.md and config.yaml

# 3. Start after review and a clean Git tree
python -m activeview.scripts.start_experiment --experiment EXPxxx

# 4. Run approved Train + Val only; write val_metrics.json and analysis.json

# 5. Write the scientific conclusion

# 6. Validate and finalize
python -m activeview.scripts.validate_experiment --experiment EXPxxx
python -m activeview.scripts.finalize_experiment \
  --experiment EXPxxx --decision ACCEPT

# 7. Freeze the accepted candidate, then commit this tracked transition
python -m activeview.scripts.freeze_final_candidate --experiment EXPxxx
git add experiments/stage_c_v1/EXPxxx_* && git commit
```

Normal Stage C-v1 work must not run Test. Test remains locked until the
accepted candidate is frozen and committed, then explicitly authorized by a
human from a clean tree:

```bash
python -m activeview.scripts.authorize_final_test \
  --experiment EXPxxx --confirm-final-model-frozen
```

Authorization requires `FINAL_FROZEN + ACCEPT`, a passing validator, a clean
working tree, and a commit/configuration lock. It writes only an external
runtime authorization artifact; tracked manifest and registry files are not
modified. The artifact is then checked fail-closed by
`activeview.research.test_gate`; `--allow-test` alone cannot bypass it.

Failed and negative experiments remain in their immutable directories and in
the registry. Never delete or reuse an `EXPxxx` directory. Do not start the
next experiment automatically after one completes.
