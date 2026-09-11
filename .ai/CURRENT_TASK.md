# Current Task

## Context synchronization — completed 2026-09-11

No experiment is currently running.  The latest request was to synchronize the
research goal, frozen protocol, current assets and scientific progress into the
AI context; this update changes context documentation only.

## Current research question

For the reduced12 + eight-placement ActiveView protocol, determine whether the
remaining HAR utility gap is dominated by motion, scene/placement, viewpoint
geometry, or sequential reachability.  The latest evidence should guide a
human-approved decision between non-greedy sequential information acquisition
and further multimodal utility modeling.  Do not start either direction
automatically.

## Frozen current protocol

- Taxonomy (12): `walk`, `sit`, `stand up`, `bend`, `crawl`, `stumble`, `clap`,
  `throw`, `kick`, `knock`, `punch`, `touching face`.
- Official Train cap: 300 records/class; Official Val cap: 50 records/class;
  seed 42.
- ActiveView policy records-only split: 313 Train / 105 Val / 0 Test.
- 20 HM3D-train scenes, eight furniture-anchored placements/scene, 32 legal
  candidate viewpoints/placement.
- Frozen reduced12 ST-GCN, Stage-B utility, Stage-C-v0 and existing
  counterfactual/Stage-D caches are the only runtime assets used by the latest
  audits.
- Current diagnostics are Val-only (10,080 moving contexts); policy Test is
  not read.  No training or perception/data regeneration was performed.

## Latest completed diagnostics

### Utility source decomposition

On 68,702 legal candidate samples (143 scene/placement groups), body-relative
azimuth alignment was numerically confirmed.  Same-motion utility-map
Spearman mean/median is 0.268/0.400, versus 0.147/0.188 for matched
different-motion maps.  Same scene/placement consistency is 0.148 across
mixed actions and 0.281 for action-matched pairs.  Leave-one-sample-out
explained variance is 0.479 motion-only, 0.104 scene-only and 0.559 additive
motion+scene, leaving 0.441 interaction residual.  Motion+scene explained
variance decreases from 0.728 at 1.5 m to 0.313 at 3.0 m.

### Reachability / sequential oracle curve

Candidate-only privileged Accuracy/Macro-F1 are 0.454266/0.444782 at H1,
0.538790/0.529863 at K1, 0.597222/0.591727 at K2, 0.654762/0.650562 at K3,
0.688393/0.683594 at K4, 0.701984/0.697100 at K5, 0.708234/0.702974 at K6,
and 0.709623/0.704598 for the full candidate set.  Greedy privileged local
search reaches only 0.558234 Accuracy after four steps.  Correct-view basins
have mean largest-component size 2.23 nodes and 76.3% of correct nodes in
that component; 29.04% of contexts have no reachable correct candidate.

### Interpretation

The current conclusion is **strong motion×scene×view interaction**, with
motion as a secondary stable source.  One-shot scalar utility prediction and
monotonic greedy refinement are insufficient to explain the full oracle
ceiling.  A sequential information-acquisition protocol is scientifically
motivated, but it must be non-greedy/global enough to handle fragmented
correct basins.  This is a diagnosis, not an authorization to modify the
frozen method.

## Reproducibility boundaries

- `test_used=false` for all latest diagnostics.
- No formal WM-E/JR/ST-GCN checkpoint was modified.
- No RGB, skeleton, DINO, Habitat or perception data was regenerated.
- Runtime data/checkpoints remain external under `ACTIVEVIEW_DATA_ROOT`.

## Report locations

- `experiments/reduced12_eight_placement_v1/utility_source_decomposition/`
- `experiments/reduced12_eight_placement_v1/khop_oracle_curve/`
- `experiments/reduced12_eight_placement_v1/overnight_nbv_diagnosis/`

Task status: **CLEAN**.  Await explicit approval before any new experiment.
