# Repository Consolidation Audit

Date: 2026-08-29

## Baseline and protocol boundary

- Pre-consolidation branch: `main`
- Pre-consolidation commit: `a2935fd177bee15eca4a40b896db5907d0e937d1`
- Working-tree changes present before this task: untracked `.learnings/`, `notes.md`, and `task_plan.md`; these user-owned scratch files are not part of the consolidation.
- The controlled-research context files `.ai/RESEARCH_PLAN.md`, `.ai/RESEARCH_LOG.md`, and `.ai/REJECTED_IDEAS.md` are absent. They are not fabricated by this structural task; the absence is recorded for follow-up.
- Accepted Stage A/B/C artifacts and runtime provenance paths remain external under `ACTIVEVIEW_DATA_ROOT` and are not regenerated or rewritten.

## Version-directory audit

The repository contains the following tracked source trees before migration:

| Tree | Tracked files | Role | Consolidation decision |
|---|---:|---|---|
| `ea_avs_mvp/` | 226 | MVP0.1 historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v2/` | 146 | MVP2 historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v3/` | 152 | MVP3 historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v4/` | 21 | Historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v5/` | 31 | Historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v6/` | 42 | Historical RGB-D implementation | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v7/` | 65 | Historical motion pipeline | Remove after migrating the motion modules required by v11 |
| `ea_avs_mvp_v8/` | 43 | Historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v9/` | 90 | Historical prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v10/` | 189 | Historical candidate/entropy prototype | Remove from working tree; recoverable from Git history |
| `ea_avs_mvp_v11/` | 97 | Current selected16 + Stage A/B/C implementation | Rename to `activeview/` |

## Dependency findings

The v11 package is otherwise internally organized, but
`dataset/babel_clean_dataset_generator.py` imports:

```text
ea_avs_mvp_v7.motion.amass_loader.NormalizedMotion
ea_avs_mvp_v7.motion.motion_converter.MotionConverter
```

Those modules depend only on their sibling `joint_mapping.py` and standard/
Habitat runtime libraries. They are formal v11 runtime dependencies, not dead
historical code, so `motion/{__init__,amass_loader,joint_mapping,motion_converter}.py`
will be moved into `activeview/motion/` and imports will be rewritten. The v7
`motion_player.py` and v7-only scripts/tests have no v11 callers and will not
be migrated.

No v11 Python module imports v1–v6, v8, v9, or v10. References to those trees
outside the active package are historical reports, historical READMEs, and
old source trees slated for removal. Active docs and command examples will be
updated to `activeview/...`; historical reports retain their factual version
names and are not treated as executable paths.

## Active v11 contents to preserve

The current package contains the accepted action-recognition, perception,
dataset, active-view, Stage A/B/C, validation, visualization, and CLI modules.
The tracked VideoPose3D license/demo images and the tracked pose-estimator
binary are retained as package assets. The tracked
`ea_avs_mvp_v11/outputs/v11_visualization/utility_prediction_evaluation.png`
is a generated result and will not be treated as production source; it is
excluded from the migrated package and remains recoverable in Git history.

## Tests and entry points

All tests currently live under `ea_avs_mvp_v11/tests/` and import the v11
package. They will move to repository-level `tests/unit/` and
`tests/integration/`. Python-bound CLI scripts remain package-bound under
`activeview/scripts/`; no root operational shell script needs relocation.

## Root-file classification

- Keep `docs/`, `.ai/`, `AGENTS.md`, `README.md`, and current v11 protocol and
  failure-analysis snapshots.
- Keep root historical specification/report Markdown as read-only research
  history; they may mention v1–v11 but are not active commands.
- Keep tracked root `results/*.json` as historical result records; generated
  `outputs/` and `tmp/` are ignored/untracked and are not migrated.
- Do not move runtime datasets, checkpoints, RGB/depth, skeletons, feature
  caches, or prediction JSONL into Git; `.gitignore` already excludes them.
- Do not add an archive/legacy copy: Git history and the recovery tag are the
  archive.

## Executed consolidation and validation

1. Recovery tag `pre-activeview-consolidation` was created at the baseline
   commit.
2. `ea_avs_mvp_v11/` was renamed to `activeview/`; the required AMASS loader,
   joint mapping and Habitat motion converter moved from v7 into
   `activeview/motion/`. The obsolete v7 `MotionPlayer` export was removed.
3. Tests moved to repository-level `tests/unit/` and `tests/integration/`; all
   executable imports and active CLI examples now use `activeview`.
4. v1–v10 source trees were removed from the working tree. Generated caches
   left by deleted trees were removed; no archive copy was created.
5. Because the persisted Stage A summary contains the intentionally excluded
   legacy minival scene, `validate_stage_a.py` now filters that scene from the
   canonical current scene audit while preserving raw provenance in the
   summary. This does not change Episode/Stage A data.

Validation performed after migration:

- `python -m compileall -q activeview tests`: PASS.
- Import smoke for `activeview`, path helpers, ST-GCN and Stage C analysis:
  PASS.
- `pytest -q tests/unit tests/integration`: 58 passed, 2 skipped.
- Stage B validator against the frozen external artifact: `passed=true`,
  zero errors.
- Stage C validator against the frozen external artifact: `passed=true`,
  zero errors.
- Stage A static/NPZ audit: all Episode, NPZ and tuple-coverage checks pass;
  canonical scene audit passes after filtering the known legacy minival ID;
  the final full cached-skeleton audit also reports all integrity checks true.

No model was trained, no dataset or Habitat observation was regenerated, and
no new Test evaluation or tuning was performed.
