# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository contains versioned EA-AVS MVP prototypes for **Elderly Action Active View Selection**: lightweight Habitat-Sim experiments for active robot viewpoint selection around an abstract human skeleton. The work is intentionally scoped to geometric/viewpoint-selection simulation, not RL training, real humanoids, pose-estimation models, ROS, Unity, or large-scale dataset generation.

The current progression is:

- `ea_avs_mvp/`: MVP0.1 baseline that samples candidate viewpoints, scores geometric keypoint visibility, compares Fixed/Random/Nearest/Ours, and writes metrics/debug artifacts.
- `ea_avs_mvp_v2/`: v2.0 formalizes the key research constraint: candidate selection uses only pre-move prediction (`*_pred`), while rendered post-move evaluation produces `*_true` metrics.
- `ea_avs_mvp_v3/`: v3.0 extends v2.0 with multi-pose skeletons, human yaw/orientation modeling, and action-part-weighted scoring.

The markdown design documents at the repository root are authoritative for intended scope and constraints:

- `EA_AVS_MVP01_Project_Document.md`
- `EA_AVS_MVP20_Code_Generation_Document.md`
- `EA_AVS_MVP30_Code_Generation_Document.md`

## Environment and Dependencies

There is currently no `pyproject.toml`, `requirements.txt`, or test configuration. The code imports these runtime dependencies directly:

- `habitat_sim`
- `numpy`
- `yaml` / PyYAML
- `PIL` / Pillow

The checked-in VS Code settings indicate a Conda-managed Python environment. Run commands from inside the corresponding version directory so script-relative config paths resolve cleanly.

The default configs contain absolute Habitat scene/navmesh paths under `/home/zxf/WorkSpace/code/code/robot/habitat-sim/...`; verify or edit these paths before running on another machine.

## Common Commands

Run MVP0.1:

```bash
cd ea_avs_mvp
python scripts/run_mvp_visibility.py \
  --config configs/mvp_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/test_run
```

Run MVP v2.0:

```bash
cd ea_avs_mvp_v2
python scripts/run_mvp20_visibility.py \
  --config configs/mvp20_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/mvp20_test_run
```

Run MVP v3.0:

```bash
cd ea_avs_mvp_v3
python scripts/run_mvp30_action_aware.py \
  --config configs/mvp30_action_aware.yaml \
  --episodes 20 \
  --output-dir outputs/mvp30_test_run
```

Quick smoke run for any version: use `--episodes 1` and a temporary output directory.

```bash
cd ea_avs_mvp_v3
python scripts/run_mvp30_action_aware.py \
  --config configs/mvp30_action_aware.yaml \
  --episodes 1 \
  --output-dir outputs/smoke_test
```

Lightweight syntax validation when Habitat is unavailable:

```bash
python -m compileall ea_avs_mvp ea_avs_mvp_v2 ea_avs_mvp_v3
```

There are no repository tests at the moment. If tests are added later, document the exact full-suite and single-test commands here.

## Architecture Overview

Each MVP version follows the same script-driven architecture:

1. A top-level script in `scripts/` parses `--config`, `--episodes`, and `--output-dir`.
2. `config.py` loads and validates YAML configuration.
3. `habitat_runner.py` wraps Habitat-Sim scene loading, navmesh/pathfinder access, point sampling, geodesic distance, and `render_at()`.
4. `skeleton.py` / `action_pose_library.py` create abstract 3D human keypoints.
5. `candidate_sampler.py` creates local candidate robot camera poses around the human and filters them through navigability/geodesic constraints.
6. Evaluator modules score candidate views.
7. `policies.py` selects views for Fixed, Random, Nearest, and Ours.
8. `metrics.py` and `visualization.py` write CSV, JSONL/debug JSON, and rendered images.

Important version-specific differences:

- MVP0.1 uses one `ViewpointEvaluator` and a single score `Q`.
- v2.0 splits scoring into `PredictiveEvaluator.score_view_pred()` and `TrueEvaluator.score_view_true()`; policy selection must use `Q_pred`, and rendered observations must only affect true evaluation after selection.
- v3.0 keeps the v2.0 pred/true separation and adds `pose_type`, `human_yaw`, orientation scoring, action-part weights, and metrics such as `S_action_part_pred`, `S_orient_pred`, `S_action_part_true`, and `S_orient_true`.

## Critical Research Constraints

Preserve these constraints when modifying the code:

- Candidate selection must not call `render_at()` or use future RGB/depth observations.
- Policies, especially `OursPolicy`, must select only from pre-move predicted scores (`Q_pred` in v2/v3).
- True metrics are for post-selection evaluation only and must not influence policy decisions.
- `OursPolicy` must compare the current view with valid candidates and allow staying in place.
- v3.0 remains one-shot active re-observation, not multi-step navigation or RL.

## Outputs

Experiment runs write under the selected output directory:

- `metrics.csv`: per-policy metric rows.
- `episodes.jsonl`: per-episode summaries/failures.
- `images/`: rendered RGB images when enabled.
- `debug/`: candidate/debug JSON files.

Avoid committing generated experiment outputs unless they are intentionally curated examples.
