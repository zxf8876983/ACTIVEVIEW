# ACTIVEVIEW Current Task

## Phase 0 — Controlled Research Agent Infrastructure (2026-08-29) — awaiting final human review

Research infrastructure is implemented and lifecycle-hardened under `activeview/research/` with
immutable `EXPxxx` source directories, external runtime directories, a
monotonic CSV registry, frozen-foundation provenance, lifecycle validation,
and a fail-closed final-Test authorization gate. The registry is initialized
and empty; no real `EXP001` exists. No Stage C-v1 scientific experiment,
training, data regeneration, or new Test evaluation has started.

The latest hardening adds start-time run/config/hypothesis/command locks,
recursive frozen-artifact re-hashing, controlled config Test-lock validation,
canonical nested-manifest Test-gate validation, and rollback/integration
regressions. Next human decision: review the infrastructure and, separately,
design and explicitly authorize `EXP001`.

## Repository consolidation (2026-08-29) — completed

Repository consolidation completed from pre-consolidation commit
`a2935fd177bee15eca4a40b896db5907d0e937d1`, protected by tag
`pre-activeview-consolidation`. The sole active package is `activeview/`;
v1–v10 source trees have been removed from the working tree and remain
recoverable from Git history. The required controlled-research files
`.ai/RESEARCH_PLAN.md`, `.ai/RESEARCH_LOG.md`, and `.ai/REJECTED_IDEAS.md`
were absent and were not fabricated. This task does not retrain, regenerate
data, alter accepted Stage A/B/C artifacts, or run Test evaluation. The
post-consolidation validation results are recorded in
`docs/repository_consolidation_audit.md`.

## Status

**STAGE C IMPLEMENTED / READY FOR SCIENTIFIC REVIEW** — v11.5 canonical selected16 data, frozen ST-GCN, accepted Stage A/B artifacts, and Stage C current-conditioned utility predictors are documented below. No generation or evaluation process is currently running.

## Current truth

- Active source: `activeview/`.
- Train/Val data: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` (3,240/980, `(N,3,30,17,1)`).
- Frozen checkpoint: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`.
- Active pose chain: RGB-only 256×256 → Ultralytics YOLO26n-Pose → VideoPose3D → camera-to-gravity/YZ conversion → root/scale/yaw-only → ST-GCN.
- Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` only.
- Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.
- Offline data root: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/` with `hm3d-minival/` and `hm3d-train/` scene-set folders.

## ST-GCN protocol

The fixed 16-class mapping is official-150 audited 14 classes plus `lie` and `stumble`; `fall` is not active. BABEL `train.json`/`val.json` are split directly, with single-label filtering, strict `num_frames > 30`, conflicting source-interval removal, official caps 400/100, auxiliary classes uncapped, and seed 42. ST-GCN receives only estimated H36M-17 skeletons. Training uses tempered oversampling and class-weighted cross entropy; `ReduceLROnPlateau` and early stopping monitor deterministic full-Train loss only. The frozen checkpoint is the final stopped-epoch model; Val is evaluated once post-training for upper-bound diagnosis and is never used for checkpoint selection or policy training.

Stage A policy records use canonical `train/val/test = 6:2:2`; the persisted
split `summary.json -> split_ratios` is the single source consumed by the
Episode builder.

The split loader validates summary counts, unique record count, per-class
counts, and canonical ratios against the actual split JSON files. Stage A also
audits target scene IDs versus successfully used scene IDs, so a failed scene
cannot silently disappear from the expected coverage set.

## Offline strategy protocol

Each semantic scene/region has one furniture-based human placement and 32 candidate viewpoints (radii 1.5/2.0/2.5/3.0 m × eight azimuths). Offline generation uses four COLOR cameras per worker, RGB-to-skeleton inference, and stores skeleton/confidence plus scene ID, navmesh path, placement, raw/snapped/actual agent positions, rotations, navigability, placement-referenced reachability and costs. No RGB/Depth is saved. The schema is `semantic-region-offline-v2`; candidate metadata is `semantic-region-v2`.

The placement reachability flag is static metadata only. During sequential evaluation, Habitat reloads the navmesh and recomputes paths from the robot's current position to every pending candidate before selecting. Current policies are `NoMove`, `Fixed`, `Random`, `Nearest`, and hindsight candidate-pool `Oracle`. Stage C adds learned `PairwiseUtilityMLP` and permutation-equivariant `SetUtilityRanker`; Stage D and Habitat online learned-policy evaluation remain out of scope.

## Current data status

- HM3D-minival: `offline/hm3d-minival/00800-TEEsavR23oF/` is a legacy v11 scene and is excluded from the current strategy-evaluation scene set.
- HM3D-train: 21-scene selection is recorded in `offline/hm3d-train/dataset_summary.json`; all 21 scene folders have complete 980×4×32 manifests. `00592-CthA7sQNTPK` and `00643-ggNAcMh8JPT` have been rotation-audited against the exact offline render state. `00643` required metadata refresh only (`npz_changed=0`); no generation process remains active.
- Stage A policy split has been regenerated with canonical `train/val/test = 6:2:2`: 589/197/194 records, 980 unique records total. Episodes were rebuilt from existing offline caches: 41,819/13,987/13,774 train/val/test Episodes (69,580 total) for the 21 current HM3D-train scenes. The legacy minival scene `00800-TEEsavR23oF` is intentionally outside the current evaluation protocol.
- Dynamic, random-start and grid-start evaluation outputs remain under `results/` with their corresponding caches under `datasets/strategy_eval_cache/`.

## Canonical entry points

```text
scripts/prepare_selected16_manifests.py
scripts/generate_selected16_habitat_dataset.py
scripts/generate_selected16_habitat_parallel.py
scripts/train_selected16_habitat_stgcn.py
scripts/generate_semantic_region_candidate_metadata.py
scripts/generate_semantic_region_offline_views.py
scripts/generate_hm3d_train_four_region_offline.py
scripts/evaluate_semantic_region_offline.py
scripts/evaluate_hm3d_train_dynamic_reachability.py
scripts/evaluate_hm3d_train_random_initializations.py
scripts/evaluate_hm3d_train_grid_initializations.py
```

## Invariants

1. Never use historical datasets/checkpoints as current defaults.
2. Never feed AMASS/SMPL GT joints into ST-GCN.
3. Keep RGB-only, YOLO26n-Pose, VideoPose3D, 30 frames, H36M-17 and yaw-only alignment.
4. Keep `fall` excluded and `lie`/`stumble` included unless the user changes the protocol.
5. Do not scan `/home/zxf/MG08/` or any undeclared scene root.
6. Do not let future candidate RGB, labels or post-hoc ST-GCN predictions enter an executable policy decision.

## Stage A acceptance

The final Episode JSONL is audited after serialization with
`activeview/scripts/validate_stage_a.py`. The audit checks split isolation,
current/candidate validity, record-local cached skeleton paths, finite geometry
and costs, episode uniqueness, NPZ/Episode geometry correspondence, and
recursive future-perception leakage fields. Individual non-finite skeleton
viewpoints are allowed when they are not selected; only the final
current/candidate IDs must be finite. The default command also validates every
referenced cached NPZ archive. Real Habitat NavMesh
ShortestPath verification is an explicit second step:

```bash
conda run --no-capture-output -n habitat python -m activeview.scripts.validate_stage_a --verify-habitat
```

Unit tests do not replace this Habitat integration check. If the Habitat
dependency or scene assets are unavailable, the result must be reported as
`NOT RUN`, not as a passed Stage A acceptance.

Latest 6:2:2 acceptance outputs:

- Static + NPZ: `/home/zxf/WorkSpace/code/data/ActiveView/results/stage_a_validate_static_622_clean.json`
- Real Habitat ShortestPath: `/home/zxf/WorkSpace/code/data/ActiveView/results/stage_a_validate_habitat_622_clean.json`
- Both reports pass split, Episode, NPZ and tuple coverage checks. Habitat
  reports `path_failures=[]` for 69,580 Episodes across 21 current scenes.
  The raw summary still lists the legacy minival directory in its historical
  target list, so its scene audit reports `all_target_scenes_used=false`; this
  is not a missing scene in the current 21-scene protocol.

## Stage B offline utility labels (completed and frozen)

Stage B was implemented from the accepted Stage A Episodes and existing
estimated-skeleton NPZ archives. The builder uses a frozen ST-GCN in
`eval()`/inference mode, computes utilities with direct `log_softmax`, caches
predictions by archive path, and writes only compact diagnostics and oracle
labels (no RGB, skeleton, logits or probability arrays). Output is isolated
under:

`/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_b/`

Generated counts are 41,819 / 13,987 / 13,774 Episodes and 306,869 /
102,637 / 101,074 candidate pairs for train/val/test respectively (69,580
Episodes and 510,580 pairs total). The canonical policy split is 589/197/194
and all 21 HM3D-train scenes are recorded; the legacy `00800-TEEsavR23oF`
scene is excluded.

`stage_b_summary.json` contains NoMove, CandidateOracle and SafeOracle metrics,
headroom distributions, source hashes and protocol definitions. The
independent validator passed with zero duplicate IDs/pairs, zero missing or
unexpected Episodes, and zero record errors:

`/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_b/validation_report.json`

The validator log is saved at
`/home/zxf/WorkSpace/code/data/ActiveView/results/stage_b_validate.log`.

## Stage C current-conditioned utility prediction (implemented; ready for scientific review)

Stage C consumes only accepted Stage A/B artifacts. The current-only input is
the frozen ST-GCN 256-D feature, 16-D current log-probabilities, entropy,
top-1/top-2 margin and pose confidence (275-D). Candidate input is 11-D
geometry only: snapped displacement expressed in the current agent's yaw frame,
distance/azimuth/path features and snapped placement radii. Candidate
perception, labels, skeletons, viewpoint IDs and utilities are excluded from
model inputs; body yaw and movement penalties are not used. Training is record-balanced with SmoothL1 plus stay-inclusive
listwise ranking, and checkpoints are selected by validation recognition
Macro-F1. Test is reported once for final diagnosis only.

Artifacts:

- Feature cache and schema: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/`
- Pairwise checkpoint: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stage_c/pairwise_mlp_best.pth` (selected epoch 29, 142,785 parameters)
- Set ranker checkpoint: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stage_c/set_ranker_best.pth` (selected epoch 46, 407,745 parameters)
- Combined summary: `stage_c/stage_c_summary.json`
- Independent validator: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/validation_report.json` (`passed=true`)

Validation re-computes all saved Val/Test metric trees from prediction JSONL,
independently reselects each candidate/Stay action from predicted utilities,
checks Stage B recognition outcomes and regret, canonical 589/197/194 split
counts, Stage A/B/feature provenance hashes, finite feature schemas and feature
counts. Stage D, Habitat
re-rendering and online learned-policy evaluation have not started.

The validator additionally recomputes the complete metrics tree with a
`1e-7` absolute tolerance, cross-checks every candidate geodesic against Stage
A, verifies Stage A summary and all three Episode-file SHA-256 hashes, and
uses mutually exclusive near-zero/positive/negative utility bins. Rescue and
degradation now include both overall and conditional rates. The validator also
independently loads `datasets/policy_v11_5/splits/` through the canonical
policy-split loader and requires frozen 589/197/194 counts in the split JSON,
Stage A summary, and Stage B summary; mutually consistent but non-canonical
counts are rejected.

Final offline Stage C results are recorded in
`datasets/policy_v11_5/stage_c/stage_c_summary.json`: NoMove is
41.25%/37.13% and 41.27%/38.18% Accuracy/Macro-F1 on Val/Test; Pairwise is
63.27%/57.75% and 61.45%/55.33%; Set Ranker is 64.91%/59.80% and
62.54%/56.37%; SafeOracle is 85.85%/81.85% and 84.49%/81.11%.
Set Ranker Test mean regret is 1.614 (median 0.0075, p90 6.143) and
aggregate positive-headroom capture is 74.93%; Pairwise is 1.802
(0.0108, 6.582) and 70.83%. These are offline diagnostics only; Stage D
Habitat online learned-policy evaluation has not started, and no multi-seed
confidence interval is available.

## Stage C-v0 failure analysis (2026-08-28)

The frozen Stage C-v0 Test artifacts were analyzed read-only with
`activeview/scripts/analyze_stage_c_failures.py`. The analysis consumed the
accepted Stage A/B JSONL, Stage C feature cache, Set Ranker and Pairwise
prediction JSONL, evaluation summaries, and the passed Stage C validator
report. It did not regenerate observations, rerun pose estimation, retrain a
model, or modify any accepted upstream artifact.

The output is stored outside the source tree at
`/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/failure_analysis/`
and the review snapshot is versioned under
`docs/results/stage_c_failure_analysis/`. It contains the machine-readable
summary, episode/record CSV tables, and five diagnostic figures. Test coverage
is 13,774 Episodes from 194 independent motion records. Regret is highly
right-skewed (median 0.00746, p90 6.143, mean 1.614); 10.00% of Episodes are
above p90 and the top 10% of motion records account for 40.20% of top-5%
catastrophic Episodes (4.02x the uniform 10% baseline), so failures are
substantially but not exclusively concentrated in difficult records.

The dominant taxonomy is wrong-candidate high utility loss (32.44%), followed
by correct SafeOracle action (29.85%) and missed move (21.95%). Exact candidate
hit is 33.93%, while aggregate positive-headroom capture is 74.93%; the miss
gap analysis shows both near-equivalent misses and materially harmful misses.
The hardest classes are `lie`, `play instrument`, `stumble`, and `knock` by Set
accuracy/regret. Region differences are descriptive and do not establish
unseen-scene generalization. State correlations are weak (entropy Spearman
rho=+0.184; margin=-0.180; pose confidence=+0.023). Symmetric-geometry
ambiguity is only modestly enriched in high-regret Episodes, so evidence for
perceived body orientation is weak/inconclusive. The corrected diagnostic
matches pairs by candidate radius and reports an explicit 1.21x enrichment
ratio. The report prioritizes
long-tail hard-example handling and better current/candidate representation as
hypotheses for later review; no Stage C-v1 or Stage D decision was made. The
candidate-set gap bins are now a mutually exclusive partition, geometry
statistics distinguish CandidateOracle from SafeOracle move geometry, and the
symmetric-geometry diagnostic reports an explicit enrichment ratio (1.21x).

Status: Stage C-v0 failure analysis completed; results ready for scientific review.
