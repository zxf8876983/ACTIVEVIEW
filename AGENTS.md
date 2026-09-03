# ACTIVEVIEW Agent Guide

ACTIVEVIEW is a research codebase for active viewpoint selection in indoor
elderly action recognition. The repository favors scientific correctness,
reproducibility, readability and fast iteration over production-platform
engineering:

```text
scientific correctness > reproducibility > readability > iteration speed > engineering robustness
```

## Reading order

At the start of a task read:

1. `AGENTS.md`;
2. `.ai/PROJECT_STATE.md`;
3. `.ai/CURRENT_TASK.md`;
4. the relevant experiment `README.md`/`config.yaml` when an experiment is involved;
5. only the directly relevant source files.

Read `.ai/RESEARCH_LOG.md` when historical experiment context is needed and
`.ai/HANDOFF.md` only when resuming an interrupted task. Historical documents
under `docs/archive/` are not default context.

## Source and data boundaries

- `activeview/` is the sole production and research source package. Never
  create copied version trees such as `ea_avs_mvp_v12/` or `activeview_v2/`.
- Runtime datasets, checkpoints, RGB/depth, skeletons, caches and predictions
  belong under `ACTIVEVIEW_DATA_ROOT` (default `../../data/ActiveView/`), not Git.
- Habitat assets are read from `ACTIVEVIEW_HABITAT_DATA_ROOT` (the configured
  `robot/DATA/` directory). Never scan undeclared disks, especially
  `/home/zxf/MG08/`.
- Resolve paths through `activeview/core/paths.py`; do not hardcode new machine
  paths or create repository symlinks to runtime data.

## Runtime environment and GPU policy

- The user authorizes project tasks to run the Conda `habitat` environment
  outside the Codex sandbox when GPU access or runtime artifacts require it.
  Such external execution remains scoped to this repository's approved
  experiments and must follow the checks below.
- All project Python commands must use the Conda `habitat` environment; do not
  use the system Python. Prefer
  `/home/zxf/anaconda3/envs/habitat/bin/python` or
  `conda run -n habitat python ...`. Do not assume `conda activate` persists
  between shell commands.
- This project is GPU-based. Before any training, evaluation, inference or
  experiment, run `nvidia-smi` and verify CUDA with:
  `/home/zxf/anaconda3/envs/habitat/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`.
- Use CUDA whenever it is available. Do not silently fall back to CPU for a
  PyTorch workload. If CUDA is expected but unavailable, stop and report the
  problem instead of running the workload.
- Before launching a long experiment, verify the Conda/Python environment, GPU
  availability and selected GPU, and print the Python executable and PyTorch
  CUDA status.

## Frozen v11.5 scientific invariants

Do not change these definitions without explicit user authorization:

- RGB-only 256×256 → YOLO26n-Pose → VideoPose3D → H36M-17 normalization →
  frozen ST-GCN;
- 16 selected action classes, 21 HM3D-train scenes, four semantic regions and
  32 candidate viewpoints;
- record split `train/val/test = 589/197/194`;
- estimated skeletons only; no AMASS/SMPL GT joints to ST-GCN;
- no future candidate RGB/depth, labels or post-hoc predictions in a decision;
- accepted Stage A/B/C-v0 artifacts, feature schema and utility definitions;
- candidate pool, baseline definitions, metrics and randomization protocol.

During Stage C-v1 development use Train and Val only. Do not use Test for
architecture, loss, sampler or hyperparameter selection. Run final Test only
when the user explicitly states that the final method is frozen and requests
the evaluation.

## Research-first coding principles

- Prefer a direct function or CLI argument when a feature serves one
  experiment. Do not add managers, services, registries, state machines,
  generic plugin systems or provenance/authorization frameworks unless the
  user explicitly requests them.
- Make one core scientific change per experiment. Keep the experiment record
  under `experiments/stage_c_v1/<EXP>/` with a README, config, run script,
  result and analysis. Keep large runtime artifacts outside Git.
- Preserve negative results. Never silently change a frozen protocol to make a
  metric look better, and never overwrite an earlier experiment.
- Comments and identifiers use English; explanations to the user use Chinese.
- Keep code readable, but do not refactor stable scientific code solely
  because a file is long. Prefer local, minimal changes over architectural
  cleanup. Add type hints, avoid mutable defaults and bare `except`, and
  follow standard import ordering.

## Validation

Run only checks relevant to the change and report commands honestly:

```bash
python -m compileall -q activeview tests
python -m pytest -q tests/unit tests/integration
```

Scientific validators may be run read-only when their frozen runtime artifacts
are available:

```bash
python -m activeview.scripts.eval.validate_policy_episodes
python -m activeview.scripts.eval.validate_utility_labels
python -m activeview.scripts.eval.validate_initial_policy_data
```

Do not claim Habitat/GPU/data-heavy validation that was not run. If a required
dependency or artifact is unavailable, report `NOT RUN` and the reason.

## Context maintenance

- Update `.ai/PROJECT_STATE.md` only when the scientific phase or canonical
  protocol changes.
- Overwrite `.ai/CURRENT_TASK.md` when the current task changes.
- Append meaningful experiment outcomes to `.ai/RESEARCH_LOG.md`.
- Update `.ai/HANDOFF.md` only for interruption or handoff; a clean completed
  task should leave it concise and marked `CLEAN`.

At completion, summarize modified files, validation, runtime state and the
next human decision. Stop when the requested task is complete; do not start a
new experiment automatically.
